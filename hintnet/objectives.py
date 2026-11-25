"""Classification and complete-triplet logical supervision (paper equations 5–7)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TASKS = ("action", "short_term", "long_term")


def _compatibility_tensor(value, shape: Optional[Sequence[int]] = None) -> Tensor:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("compatibility must be a rectangular three-dimensional 0/1 tensor") from exc
    if tensor.ndim != 3 or any(size <= 0 for size in tensor.shape):
        raise ValueError("compatibility must have three nonempty dimensions (action, short_term, long_term)")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"compatibility shape must be {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.is_complex() or not bool(torch.all((tensor == 0) | (tensor == 1))):
        raise ValueError("compatibility entries must be booleans or exactly 0/1")
    return tensor.to(dtype=torch.bool)


def load_compatibility(path: str | Path, num_classes=(7, 8, 4)) -> Tensor:
    """Read an explicit compatibility definition; no default semantic graph is inferred.

    JSON accepts ``{"valid_triplets": [[action_id, short_id, long_id], ...]}``
    or ``{"compatibility": [[[0, 1, ...], ...], ...]}``. IDs are zero-based.
    Extra provenance fields are allowed, but exactly one representation is required.
    """
    if len(num_classes) != 3 or any(type(size) is not int or size <= 0 for size in num_classes):
        raise ValueError("num_classes must contain three positive integers")
    with Path(path).open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("compatibility JSON must be an object with valid_triplets or compatibility")
    representations = [name for name in ("valid_triplets", "compatibility") if name in document]
    if len(representations) != 1:
        raise ValueError("provide exactly one of valid_triplets or compatibility in the JSON object")
    if "compatibility" in document:
        return _compatibility_tensor(document["compatibility"], num_classes)
    triplets = document["valid_triplets"]
    if not isinstance(triplets, list):
        raise ValueError("valid_triplets must be a list of three integer class IDs")
    tensor = torch.zeros(tuple(num_classes), dtype=torch.bool)
    for index, triplet in enumerate(triplets):
        if not isinstance(triplet, list) or len(triplet) != 3 or any(type(value) is not int for value in triplet):
            raise ValueError(f"valid_triplets[{index}] must contain three integer class IDs")
        if any(value < 0 or value >= size for value, size in zip(triplet, num_classes)):
            raise ValueError(f"valid_triplets[{index}]={triplet} is outside class ranges {tuple(num_classes)}")
        tensor[tuple(triplet)] = True
    return tensor


class HierarchicalLoss(nn.Module):
    """Summed three-task CE plus mean probability mass on invalid triples.

    ``compatibility`` is deliberately required when logical supervision is active.
    It may be omitted only with ``lvp_weight=0``. Each forward returns scalar tensors
    ``total``, ``classification`` and ``lvp`` suitable for backward and logging.
    """

    def __init__(self, compatibility: Optional[Tensor], lvp_weight: float = 5.0):
        super().__init__()
        if isinstance(lvp_weight, bool) or not math.isfinite(lvp_weight) or lvp_weight < 0:
            raise ValueError("lvp_weight must be finite and nonnegative")
        if compatibility is None and lvp_weight != 0:
            raise ValueError("an explicit compatibility tensor is required when lvp_weight is nonzero")
        compatibility = None if compatibility is None else _compatibility_tensor(compatibility).detach().clone()
        self.register_buffer("compatibility", compatibility)
        self.lvp_weight = float(lvp_weight)

    def forward(self, logits: Mapping[str, Tensor], labels: Tensor) -> dict[str, Tensor]:
        missing = [task for task in TASKS if task not in logits]
        if missing:
            raise ValueError(f"missing logits for tasks: {missing}")
        first = logits[TASKS[0]]
        if not isinstance(first, Tensor) or first.ndim != 2 or first.shape[0] == 0:
            raise ValueError("task logits must have shape [nonempty batch, classes]")
        batch = first.shape[0]
        if not isinstance(labels, Tensor) or labels.shape != (batch, 3):
            raise ValueError(f"labels must be a tensor with shape ({batch}, 3)")
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("labels must contain integer class IDs")
        if labels.device != first.device:
            raise ValueError("labels and logits must be on the same device")
        probabilities = []
        losses = []
        for index, task in enumerate(TASKS):
            values = logits[task]
            if not isinstance(values, Tensor) or values.ndim != 2 or values.shape[0] != batch or values.shape[1] == 0:
                raise ValueError(f"{task} logits must have shape [{batch}, nonempty classes]")
            if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
                raise ValueError(f"{task} logits must contain finite floating-point values")
            if values.device != first.device:
                raise ValueError("all task logits must be on the same device")
            if bool(((labels[:, index] < 0) | (labels[:, index] >= values.shape[1])).any()):
                raise ValueError(f"labels for {task} are outside [0, {values.shape[1]})")
            if self.compatibility is not None and values.shape[1] != self.compatibility.shape[index]:
                raise ValueError(f"{task} logits have {values.shape[1]} classes, compatibility expects {self.compatibility.shape[index]}")
            # Float32 avoids underflow in the joint probability under mixed precision.
            probabilities.append(values.float().softmax(dim=-1))
            losses.append(F.cross_entropy(values.float(), labels[:, index].long()))
        classification = torch.stack(losses).sum()
        if self.compatibility is None:
            lvp = classification.new_zeros(())
        else:
            invalid = (~self.compatibility).to(device=first.device, dtype=probabilities[0].dtype)
            lvp = torch.einsum("ba,bs,bl,asl->b", *probabilities, invalid).mean()
        total = classification + self.lvp_weight * lvp
        return {"total": total, "classification": classification, "lvp": lvp}
