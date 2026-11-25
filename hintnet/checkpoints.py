"""Portable trainable-only weights with exact frozen-decoder identity checks.

The small release checkpoint contains learned queries, projection, prefix and
classification layers. It intentionally does not contain the frozen T5 or
ActionCLIP weights. A SHA256 over every frozen decoder tensor prevents silently
using another pretrained decoder when the small checkpoint is restored.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch


ADAPTER_FORMAT = "hintnet_adapter_v1"
T5_MODEL = "google-t5/t5-base"
T5_REVISION = "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1"
ACTIONCLIP_SHA256 = "0acfe0359eb73c02e93f0b1ee4761e7099747d15fe305aa833776f6681c8aa5b"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash names, exact shapes/dtypes and contiguous CPU tensor bytes."""
    if not state:
        raise ValueError("Cannot fingerprint an empty state dictionary")
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected a tensor at {name}")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        # uint8 views support bfloat16 too, without lossy dtype conversion.
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_decoder_state(model):
    return {name: value for name, value in model.state_dict().items() if name.startswith("decoder.")}


def trainable_state(model):
    names = {name for name, value in model.named_parameters() if value.requires_grad}
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in sorted(names)}


def load_adapter_state(model, payload):
    """Validate and load *every* trainable tensor, retaining real pretrained T5.

    Construct ``model`` through the normal pretrained loading branch before
    calling this function. Passing decoder_config to a random T5Stack and then
    loading only an adapter is explicitly detected by the fingerprint check.
    """
    if payload.get("format") != ADAPTER_FORMAT or payload.get("format_version") != 1:
        raise ValueError("Unsupported portable HINT-Net adapter format")
    weights = payload.get("model")
    if not isinstance(weights, Mapping):
        raise ValueError("Adapter has no tensor state dictionary")
    expected = {name for name, value in model.named_parameters() if value.requires_grad}
    supplied = set(weights)
    if supplied != expected:
        raise ValueError(f"Adapter parameter mismatch: missing={sorted(expected-supplied)}, unexpected={sorted(supplied-expected)}")
    state = model.state_dict()
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.shape != state[name].shape or value.dtype != state[name].dtype:
            raise ValueError(f"Adapter shape or dtype mismatch for {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Adapter has nonfinite values at {name}")
    actual_decoder = tensor_state_sha256(frozen_decoder_state(model))
    if actual_decoder != payload.get("source", {}).get("frozen_decoder_sha256"):
        raise ValueError("Frozen T5 decoder fingerprint differs from the decoder used to train this adapter")
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or set(missing) != set(state) - expected:
        raise ValueError(f"Adapter restoration mismatch: missing={missing}, unexpected={unexpected}")
    if any(not name.startswith(("decoder.", "backbone.")) for name in missing):
        raise ValueError("Adapter omitted a non-pretrained model tensor")
    return {"trainable_tensors": len(expected),
            "trainable_parameters": sum(value.numel() for value in weights.values()),
            "frozen_decoder_sha256": actual_decoder}
