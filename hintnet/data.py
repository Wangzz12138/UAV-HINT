"""Instance clips and frozen-encoder token caches for UAV-HINT.

The supplied paper does not include the annotation export, official video split,
or label IDs. This module uses an explicit JSONL interchange format; it never
silently invents those resources. See ``tools/prepare_manifest.py --help``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor
from torch.utils.data import Dataset


LEVELS = ("action", "short_term", "long_term")
# Reproduction IDs, in the displayed order of paper Fig. 2, not official IDs.
LABELS = {
    "action": ["Sitting", "Looking around", "Running", "Using phone", "Riding", "Walking", "Standing"],
    "short_term": ["Keeping", "Entering a building", "Searching", "Crossing intersection", "Talking", "Turning", "Approaching someone", "Waiting"],
    "long_term": ["Daily routine", "Going somewhere", "Social interaction", "Undetermined"],
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TEXT_FIELDS = ("scene", "time", "weather", "age", "gender", "state")


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array or one JSON object per line, with useful line errors."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path}: JSON manifest must contain a list")
    else:
        records = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc.msg}") from exc
    if not records:
        raise ValueError(f"{path}: empty manifest")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object")
        for key in ("video_id", "instance_id"):
            if key not in record or record[key] is None or str(record[key]) == "":
                raise ValueError(f"{path}: record {index} needs {key}")
    return records


def validate_manifest_splits(manifest: str | Path | Sequence[str | Path]) -> dict[str, int]:
    """Reject video-level leakage within or across manifest files.

    Split fields are required. Counts are unique videos, rather than instances.
    Official video IDs must be supplied by the dataset owner; no random split is
    created here.
    """
    paths = [manifest] if isinstance(manifest, (str, Path)) else list(manifest)
    assignments: dict[str, str] = {}
    for path in paths:
        for record in read_manifest(path):
            split = record.get("split")
            if split not in ("train", "val", "test"):
                raise ValueError(f"video {record['video_id']}: split must be train, val, or test")
            video_id = str(record["video_id"])
            if video_id in assignments and assignments[video_id] != split:
                raise ValueError(f"video leakage: {video_id!r} appears in {assignments[video_id]!r} and {split!r}")
            assignments[video_id] = split
    return {split: sum(value == split for value in assignments.values()) for split in ("train", "val", "test")}


def load_label_names(path: str | Path | Mapping | None = None) -> dict[str, list[str]]:
    labels = LABELS if path is None else (path if isinstance(path, Mapping)
             else json.loads(Path(path).read_text(encoding="utf-8")))
    result = {}
    for level in LEVELS:
        values = labels.get(level)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v.strip() for v in values):
            raise ValueError(f"label vocabulary {level!r} must be a nonempty list of names")
        if len(set(v.casefold() for v in values)) != len(values):
            raise ValueError(f"duplicate label names at {level}")
        result[level] = list(values)
    return result


def encode_labels(labels: Any, vocabulary: Mapping[str, Sequence[str]] = LABELS) -> Tensor:
    if isinstance(labels, Mapping):
        if any(level not in labels for level in LEVELS):
            raise ValueError("labels must contain action, short_term, and long_term")
        values = [labels[level] for level in LEVELS]
    elif isinstance(labels, (list, tuple)) and len(labels) == 3:
        values = list(labels)
    else:
        raise ValueError("labels must be a three-element list or an object with the three semantic levels")
    encoded = []
    for level, value in zip(LEVELS, values):
        names = vocabulary[level]
        if isinstance(value, str):
            lookup = {name.strip().casefold(): i for i, name in enumerate(names)}
            key = value.strip().casefold()
            if key not in lookup:
                raise ValueError(f"unknown {level} label: {value!r}")
            value = lookup[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(names):
            raise ValueError(f"{level} label ID must be in [0, {len(names) - 1}], got {value!r}")
        encoded.append(value)
    return torch.tensor(encoded, dtype=torch.long)


def format_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        # Only context fields enter the encoder; labels/other annotation keys do not.
        return "; ".join(f"{key}: {value[key]}" for key in TEXT_FIELDS if key in value and value[key] is not None)
    raise ValueError("text must be a string or a dictionary of contextual tags")


def sample_frame_indices(length: int, count: int, training: bool = False) -> list[int]:
    if length <= 0 or count <= 0:
        raise ValueError("clip length and num_frames must be positive")
    if not training or length < count:
        return np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()
    boundaries = np.linspace(0, length, count + 1).astype(int)
    return [int(torch.randint(int(a), max(int(a) + 1, int(b)), ()).item()) for a, b in zip(boundaries[:-1], boundaries[1:])]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _validate_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("each bbox must contain four pixel coordinates [x1, y1, x2, y2]")
    try:
        box = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox coordinates must be numeric") from exc
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"bbox must be finite with x2>x1 and y2>y1, got {value!r}")
    return box


class UAVHintDataset(Dataset):
    """Load one observed instance clip per manifest record.

    Raw input records require ``frames`` (ordered image paths) and ``bboxes``
    (one xyxy pixel box per frame) when local input is enabled. Cached records
    require ``tokens`` pointing to a .pt tensor [S,D] or dictionary containing
    ``tokens`` or any of ``global``, ``local``, and ``text`` token tensors [S,D].
    ``text`` in a cache dictionary is a tensor, while manifest text is a string
    or contextual tag dictionary. Token files are loaded with weights_only=True.
    """

    def __init__(self, manifest: str | Path, root: str | Path | None = None,
                 split: str | None = None, num_frames: int = 8, image_size: int = 224,
                 training: bool = False, input_mode: str = "video",
                 labels_path: str | Path | Mapping | None = None, use_global: bool = True,
                 use_local: bool = True, require_labels: bool = True,
                 token_dim: int = 512):
        self.manifest = Path(manifest)
        self.root = Path(root) if root is not None else self.manifest.resolve().parent
        self.vocabulary = load_label_names(labels_path)
        self.num_frames, self.image_size = num_frames, image_size
        self.training, self.input_mode = training, input_mode
        self.use_global, self.use_local = use_global, use_local
        self.require_labels, self.token_dim = require_labels, token_dim
        if num_frames <= 0 or image_size <= 0:
            raise ValueError("num_frames and image_size must be positive")
        if input_mode not in ("video", "tokens"):
            raise ValueError("input_mode must be 'video' or 'tokens'")
        if not use_global and not use_local and input_mode == "video":
            raise ValueError("video mode requires at least one of global/local input")
        all_records = read_manifest(self.manifest)
        if any("split" in record for record in all_records) or split is not None:
            validate_manifest_splits(self.manifest)
        if split is not None and split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.records = [record for record in all_records if split is None or record.get("split") == split]
        if not self.records:
            raise ValueError(f"no samples for split {split!r} in {self.manifest}")
        for record in self.records:
            self._validate_record(record)
        self.mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def _validate_record(self, record: Mapping[str, Any]) -> None:
        if self.require_labels or "labels" in record:
            encode_labels(record.get("labels"), self.vocabulary)
        format_text(record.get("text", record.get("tags")))
        if self.input_mode == "tokens":
            if not isinstance(record.get("tokens"), str) or not record["tokens"]:
                raise ValueError("token mode requires a tokens path in every record")
            return
        frames = record.get("frames")
        if not isinstance(frames, list) or not frames or not all(isinstance(f, str) and f for f in frames):
            raise ValueError("video mode requires a nonempty frames list of image paths")
        if self.use_local:
            boxes = record.get("bboxes")
            if not isinstance(boxes, list) or len(boxes) != len(frames):
                raise ValueError("bboxes must have one xyxy box per frame for local input")
            for box in boxes:
                _validate_bbox(box)

    def __len__(self) -> int:
        return len(self.records)

    def _image_tensor(self, image: Image.Image, flip: bool) -> Tensor:
        if flip:
            image = ImageOps.mirror(image)
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
        return (tensor - self.mean) / self.std

    def _load_tokens(self, path: Path, source_frame_count: int | None = None) -> tuple[Tensor, Tensor]:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mask = None
        if isinstance(payload, Mapping):
            mask = payload.get("token_mask")
            if "tokens" in payload:
                tokens = payload["tokens"]
            else:
                # Cache ordering is global, local, text; configurable upstream.
                keys = [key for key, enabled in (("global", self.use_global), ("local", self.use_local), ("text", True)) if enabled and key in payload]
                if not keys:
                    raise ValueError(f"{path}: cache needs tokens or global/local/text tensors")
                blocks = [payload[key] for key in keys]
                if not all(isinstance(t, Tensor) and t.ndim == 2 for t in blocks):
                    raise ValueError(f"{path}: each cached modality must be a [S,D] tensor")
                tokens = torch.cat(blocks, dim=0)
            if "flipped_tokens" in payload:
                flipped = payload["flipped_tokens"]
                if not isinstance(tokens, Tensor) or not isinstance(flipped, Tensor) or flipped.shape != tokens.shape:
                    raise ValueError(f"{path}: flipped_tokens must have the same shape as tokens")
                if not torch.isfinite(flipped).all():
                    raise ValueError(f"{path}: flipped_tokens contain non-finite values")
                if self.training:
                    if source_frame_count != self.num_frames:
                        raise ValueError("Augmentation caches require exactly num_frames source frames; temporal sampling cannot be cached as two variants")
                    flip = bool(torch.rand(()) < 0.5)
                    # Raw video loading draws the flip bit first, then one
                    # singleton randint per temporal stratum even for T==N.
                    # Preserve this RNG stream as well as the augmentation.
                    sample_frame_indices(source_frame_count, self.num_frames, True)
                    if flip:
                        tokens = flipped
        else:
            tokens = payload
        if not isinstance(tokens, Tensor) or tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[1] != self.token_dim:
            raise ValueError(f"{path}: expected nonempty tokens [S,{self.token_dim}]")
        if not torch.isfinite(tokens).all():
            raise ValueError(f"{path}: cached tokens contain non-finite values")
        tokens = tokens.float()
        if mask is None:
            mask = torch.ones(tokens.shape[0], dtype=torch.bool)
        if not isinstance(mask, Tensor) or mask.shape != (tokens.shape[0],):
            raise ValueError(f"{path}: token_mask must have shape [S]")
        mask = mask.bool()
        if not mask.any():
            raise ValueError(f"{path}: token_mask cannot hide every token")
        return tokens, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample = {
            "video_id": str(record["video_id"]),
            "instance_id": str(record["instance_id"]),
            "text": format_text(record.get("text", record.get("tags"))),
            "labels": encode_labels(record["labels"], self.vocabulary) if "labels" in record else torch.full((3,), -1, dtype=torch.long),
        }
        if self.input_mode == "tokens":
            sample["tokens"], sample["token_mask"] = self._load_tokens(
                _resolve(self.root, record["tokens"]), source_frame_count=len(record.get("frames", [])))
            return sample
        flip = self.training and bool(torch.rand(()) < 0.5)
        indices = sample_frame_indices(len(record["frames"]), self.num_frames, self.training)
        global_frames, local_frames = [], []
        for frame_index in indices:
            path = _resolve(self.root, record["frames"][frame_index])
            with Image.open(path) as source:
                image = source.convert("RGB")
            if self.use_global:
                global_frames.append(self._image_tensor(image, flip))
            if self.use_local:
                x1, y1, x2, y2 = _validate_bbox(record["bboxes"][frame_index])
                w, h = image.size
                crop_box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(w, math.ceil(x2)), min(h, math.ceil(y2)))
                if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                    raise ValueError(f"{path}: bbox lies outside the frame: {record['bboxes'][frame_index]}")
                local_frames.append(self._image_tensor(image.crop(crop_box), flip))
        if global_frames:
            sample["video"] = torch.stack(global_frames)
        if local_frames:
            sample["local_video"] = torch.stack(local_frames)
        return sample


def collate_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack clips and pad variable-length cached tokens (True means valid)."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch = {key: [sample[key] for sample in samples] for key in ("video_id", "instance_id", "text")}
    batch["labels"] = torch.stack([sample["labels"] for sample in samples])
    for key in ("video", "local_video"):
        if key in samples[0]:
            batch[key] = torch.stack([sample[key] for sample in samples])
    if "tokens" in samples[0]:
        max_length = max(sample["tokens"].shape[0] for sample in samples)
        first = samples[0]["tokens"]
        batch["tokens"] = first.new_zeros(len(samples), max_length, first.shape[1])
        batch["token_mask"] = torch.zeros(len(samples), max_length, dtype=torch.bool)
        for index, sample in enumerate(samples):
            length = sample["tokens"].shape[0]
            batch["tokens"][index, :length] = sample["tokens"]
            batch["token_mask"][index, :length] = sample["token_mask"]
    return batch
