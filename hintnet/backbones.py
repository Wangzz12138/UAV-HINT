"""Frozen ActionCLIP input encoder for the HINT-Net reconstruction.

Only pretrained ActionCLIP checkpoints containing both ``model_state_dict`` and
``fusion_model_state_dict`` are accepted.  A plain OpenAI CLIP checkpoint does
not contain the trained temporal Transf module and is intentionally rejected.

The paper does not identify which intermediate encoder features become tokens.
Here each enabled video stream contributes the original ActionCLIP Transf
pooled video embedding; contextual text contributes its CLIP EOT embedding.
``visual_token_mode='frames'`` instead retains the contextualized temporal
features immediately before the original mean pooling. Both choices are
explicit reconstruction settings, not claims about unreleased author code.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _strip_parallel_prefix(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    result = {}
    for key, value in state.items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key in result:
            raise ValueError(f"Duplicate checkpoint parameter after prefix removal: {key}")
        result[key] = value
    return result


class FrozenActionCLIP(nn.Module):
    """Encode synchronized global/local clips and contextual descriptions.

    Args:
        checkpoint: Local ActionCLIP checkpoint, including trained Transf weights.
        num_frames: Expected clip length. Paper leaves this setting unspecified.
        use_global/use_local/use_text: Input modality ablation switches.
        visual_token_mode: ``pooled`` (one token/stream) or ``frames`` (T/stream).
        normalize: Apply the original ActionCLIP feature L2 normalization.
        image_batch_size: Optional chunk size for the frozen image encoder.
        disable_cudnn: Use native convolution for the frozen visual encoder.
            Defaults to True because repeated cuDNN Conv2d outputs were found
            unstable on the current workstation; scoped to each encoder call.
        input_resolution: Optional square input size, divisible by the ViT patch
            size. Non-native sizes interpolate spatial positional embeddings
            during the forward; pretrained parameters and the CLS position stay
            unchanged. None preserves the checkpoint's native resolution.

    Inputs must already use CLIP RGB normalization. Returns tokens of shape
    [B,S,512] and an all-valid boolean mask of shape [B,S]. Frozen modules always
    remain in evaluation mode, including when a parent calls ``train()``.
    """

    output_dim = 512

    def __init__(
        self,
        checkpoint: str | Path,
        num_frames: int = 8,
        use_global: bool = True,
        use_local: bool = True,
        use_text: bool = True,
        visual_token_mode: str = "pooled",
        normalize: bool = True,
        image_batch_size: int = 0,
        disable_cudnn: bool = True,
        input_resolution: int | None = None,
    ) -> None:
        super().__init__()
        if not any((use_global, use_local, use_text)):
            raise ValueError("At least one ActionCLIP input modality must be enabled")
        if visual_token_mode not in ("pooled", "frames"):
            raise ValueError("visual_token_mode must be 'pooled' or 'frames'")
        if num_frames < 1 or image_batch_size < 0:
            raise ValueError("num_frames must be positive and image_batch_size nonnegative")
        if input_resolution is not None and (type(input_resolution) is not int or input_resolution < 1):
            raise ValueError("input_resolution must be a positive integer or None")
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ActionCLIP pretrained checkpoint not found: {checkpoint}")

        # Dependencies and BPE vocabulary are needed only for raw-input mode.
        from .vendor.actionclip.model import build_model
        from .vendor.actionclip.simple_tokenizer import SimpleTokenizer
        from .vendor.actionclip.visual_prompt import visual_prompt

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        required = ("model_state_dict", "fusion_model_state_dict")
        if not isinstance(payload, Mapping) or any(key not in payload for key in required):
            raise ValueError(
                "Expected an ActionCLIP checkpoint containing model_state_dict and "
                "fusion_model_state_dict; a plain CLIP checkpoint is insufficient"
            )
        clip_state = _strip_parallel_prefix(payload["model_state_dict"])
        fusion_state = _strip_parallel_prefix(payload["fusion_model_state_dict"])
        # The optimizer state in released checkpoints can be much larger than
        # the model; release its references before constructing the modules.
        del payload
        if "visual.proj" not in clip_state:
            raise ValueError("HINT-Net requires ActionCLIP ViT-B/16 or ViT-B/32 weights")
        if clip_state["visual.conv1.weight"].shape[-1] not in (16, 32):
            raise ValueError("Unsupported ActionCLIP patch size; expected ViT-B/16 or ViT-B/32")
        patch_size = clip_state["visual.conv1.weight"].shape[-1]
        if input_resolution is not None and input_resolution % patch_size:
            raise ValueError(f"input_resolution must be divisible by the ViT patch size {patch_size}")
        if clip_state["text_projection"].shape[1] != self.output_dim:
            raise ValueError("HINT-Net expects the 512-dimensional ActionCLIP embedding space")
        if "frame_position_embeddings.weight" not in fusion_state:
            raise ValueError("The checkpoint does not contain the required Transf temporal head")
        if num_frames > fusion_state["frame_position_embeddings.weight"].shape[0]:
            raise ValueError("num_frames exceeds the pretrained temporal position embedding capacity")

        # build_model and load_state_dict both load strictly; no random fallback.
        self.clip = build_model(dict(clip_state), T=num_frames).float()
        self.fusion = visual_prompt("Transf", clip_state, num_frames).float()
        self.fusion.load_state_dict(fusion_state, strict=True)
        self.tokenizer = SimpleTokenizer() if use_text else None
        self.checkpoint_path = str(checkpoint.resolve())
        self.input_resolution = self.clip.visual.input_resolution if input_resolution is None else input_resolution
        self.num_frames = num_frames
        self.use_global = use_global
        self.use_local = use_local
        self.use_text = use_text
        self.visual_token_mode = visual_token_mode
        self.normalize = normalize
        self.image_batch_size = image_batch_size
        self.disable_cudnn = disable_cudnn
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenActionCLIP:
        return super().train(False)

    def _encode_video(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != self.num_frames or video.shape[2] != 3:
            raise ValueError(f"Expected [B,{self.num_frames},3,H,W] clip; received {tuple(video.shape)}")
        resolution = self.input_resolution
        if tuple(video.shape[-2:]) != (resolution, resolution):
            raise ValueError(f"Configured ActionCLIP expects {resolution}x{resolution} input frames")
        batch, frames = video.shape[:2]
        flat = video.reshape(batch * frames, *video.shape[2:])
        chunk_size = self.image_batch_size or flat.shape[0]
        convolution_context = torch.backends.cudnn.flags(enabled=False) if self.disable_cudnn else nullcontext()
        with convolution_context:
            encoded = torch.cat([self.clip.encode_image(x) for x in flat.split(chunk_size)], dim=0)
        encoded = encoded.reshape(batch, frames, -1)
        if self.visual_token_mode == "pooled":
            return self.fusion(encoded).unsqueeze(1)
        # Exact pre-pooling computation of ActionCLIP's original Transf branch.
        positions = torch.arange(frames, device=encoded.device).unsqueeze(0).expand(batch, -1)
        temporal = encoded + self.fusion.frame_position_embeddings(positions)
        temporal = self.fusion.transformer(temporal.permute(1, 0, 2)).permute(1, 0, 2)
        return temporal.to(encoded.dtype) + encoded

    def _tokenize(self, text: Sequence[str], device: torch.device) -> Tensor:
        context_length = self.clip.context_length
        start = self.tokenizer.encoder["<|startoftext|>"]
        end = self.tokenizer.encoder["<|endoftext|>"]
        result = torch.zeros(len(text), context_length, dtype=torch.long, device=device)
        for index, description in enumerate(text):
            if not isinstance(description, str):
                raise TypeError("Each contextual description must be a string")
            encoded = [start, *self.tokenizer.encode(description), end]
            if len(encoded) > context_length:
                raise ValueError(
                    f"Description {index} needs {len(encoded)} tokens; ActionCLIP allows "
                    f"{context_length}. Shorten the contextual description explicitly."
                )
            result[index, :len(encoded)] = torch.tensor(encoded, dtype=torch.long, device=device)
        return result

    @torch.no_grad()
    def forward(
        self,
        video: Tensor | None = None,
        local_video: Tensor | None = None,
        text: Sequence[str] | None = None,
    ) -> tuple[Tensor, Tensor]:
        tokens = []
        if self.use_global:
            if video is None:
                raise ValueError("Global video is enabled but no video tensor was supplied")
            tokens.append(self._encode_video(video))
        if self.use_local:
            if local_video is None:
                raise ValueError("Local video is enabled but no local_video tensor was supplied")
            tokens.append(self._encode_video(local_video))
        if self.use_text:
            if text is None or isinstance(text, str) or len(text) == 0:
                raise ValueError("Text is enabled: provide one contextual string per sample")
            device = self.clip.token_embedding.weight.device
            tokens.append(self.clip.encode_text(self._tokenize(text, device)).unsqueeze(1))
        if not tokens or len({value.shape[0] for value in tokens}) != 1:
            raise ValueError("All enabled input modalities must have the same nonzero batch size")
        combined = torch.cat(tokens, dim=1).float()
        if self.normalize:
            combined = F.normalize(combined, dim=-1)
        return combined, torch.ones(combined.shape[:2], dtype=torch.bool, device=combined.device)
