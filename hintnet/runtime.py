"""Shared reproducible configuration, loading, and evaluation helpers."""
import copy
import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (LEVELS, UAVHintDataset, collate_batch, load_label_names,
                   validate_manifest_splits)
from .metrics import evaluate_predictions
from .model import HINTNet, ModelConfig
from .objectives import load_compatibility


DEFAULT_CONFIG = {
    "model": ModelConfig().to_dict(),
    "data": {"input_mode": "video", "num_frames": 8, "image_size": 336,
             "use_global": True, "use_local": True},
    "backbone": {"checkpoint": None, "use_text": True,
                 "visual_token_mode": "pooled", "image_batch_size": 0,
                 "normalize": True, "disable_cudnn": True,
                 "initialize_for_cached_inputs": False},
    "training": {"epochs": 50, "batch_size": 8, "lr": 5e-5,
                 "weight_decay": 0.01, "warmup_ratio": 0.1,
                 "num_workers": 4, "seed": 42, "grad_clip": 1.0,
                 "amp": False, "cpu_threads": 4,
                 "validate": True, "checkpoint_every": 1},
    "manifest": None, "root": None, "labels": None,
    "compatibility": None, "lvp_weight": 5.0,
}


def merge_config(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key not in result:
            raise ValueError(f"Unknown config field: {key}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    return merge_config(DEFAULT_CONFIG, json.loads(Path(path).read_text(encoding="utf-8")))


def seed_everything(seed, cpu_threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def choose_device(value):
    device = torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def make_dataset(config, split, training=False, require_labels=True, label_names=None):
    if not config["manifest"]:
        raise ValueError("Provide a manifest in config or with --manifest")
    validate_manifest_splits(config["manifest"])
    return UAVHintDataset(
        config["manifest"], root=config["root"], split=split,
        labels_path=label_names if label_names is not None else config["labels"], training=training,
        require_labels=require_labels, token_dim=config["model"]["feature_dim"],
        **config["data"],
    )


def make_loader(dataset, config, device, training=False, generator=None):
    return DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=training,
        num_workers=config["training"]["num_workers"], collate_fn=collate_batch,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker,
        generator=generator, drop_last=False,
    )


def make_model(config, device, decoder_config=None):
    backbone = None
    backbone_options = dict(config["backbone"])
    initialize_for_cached_inputs = backbone_options.pop("initialize_for_cached_inputs", False)
    if config["data"]["input_mode"] == "video" or initialize_for_cached_inputs:
        from .backbones import FrozenActionCLIP
        if not config["backbone"]["checkpoint"]:
            raise ValueError("Raw input requires --actionclip-checkpoint or backbone.checkpoint")
        if config["model"]["feature_dim"] != 512:
            raise ValueError("Real ActionCLIP input requires feature_dim=512")
        backbone = FrozenActionCLIP(
            num_frames=config["data"]["num_frames"],
            input_resolution=config["data"]["image_size"],
            use_global=config["data"]["use_global"],
            use_local=config["data"]["use_local"], **backbone_options)
    model = HINTNet(ModelConfig(**config["model"]), backbone=backbone,
                    decoder_config=decoder_config).to(device)
    return model


def get_compatibility(config):
    if config["compatibility"]:
        return load_compatibility(config["compatibility"], tuple(config["model"]["num_classes"]))
    if config["lvp_weight"] != 0:
        raise ValueError("LVP requires an explicit compatibility JSON; use lvp_weight=0 only for an ablation")
    return None


def model_inputs(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key in ("tokens", "token_mask", "video", "local_video", "text")}


@torch.no_grad()
def evaluate(model, loader, device, compatibility=None, collect_predictions=False):
    model.eval()
    probabilities = {level: [] for level in LEVELS}
    all_labels, rows = [], []
    for batch in loader:
        logits = model(**model_inputs(batch, device))
        scores = {name: value.float().softmax(-1).cpu() for name, value in logits.items()}
        for name, value in scores.items():
            probabilities[name].append(value)
        if "labels" in batch:
            all_labels.append(batch["labels"])
        if collect_predictions:
            for index, (video, instance) in enumerate(zip(batch["video_id"], batch["instance_id"])):
                row = {"video_id": video, "instance_id": instance,
                       "predictions": {name: int(value[index].argmax()) for name, value in scores.items()},
                       "probabilities": {name: value[index].tolist() for name, value in scores.items()}}
                if "labels" in batch and (batch["labels"][index] >= 0).all():
                    row["labels"] = batch["labels"][index].tolist()
                rows.append(row)
    if not any(probabilities.values()):
        raise ValueError("Evaluation split is empty")
    metrics = {}
    if all_labels:
        labels = torch.cat(all_labels)
        labeled = (labels >= 0).all(dim=1)
        if labeled.any():
            metrics = evaluate_predictions(
                {name: torch.cat(value)[labeled] for name, value in probabilities.items()},
                labels[labeled], compatibility)
    return metrics, rows


def load_saved_model(checkpoint, device, overrides=None):
    try:
        # The release adapter contains only ordinary metadata and tensor state.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Legacy training snapshots also carry NumPy RNG/optimizer state.
        # As with the original loader, use only your own trusted full snapshots.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported HINT-Net checkpoint format")
    config = copy.deepcopy(payload["config"])
    if overrides:
        config = merge_config(config, overrides)
    if payload.get("format") == "hintnet_adapter_v1":
        from .checkpoints import file_sha256, load_adapter_state
        if config["data"]["input_mode"] == "video" or config["backbone"].get("initialize_for_cached_inputs", False):
            expected = payload["source"].get("actionclip_checkpoint_sha256")
            if not expected or file_sha256(config["backbone"]["checkpoint"]) != expected:
                raise ValueError("ActionCLIP checkpoint fingerprint differs from the encoder used to train this adapter")
        # Do not instantiate a random decoder from decoder_config for a small
        # adapter: unlike a full snapshot, it contains no frozen T5 tensors.
        model = make_model(config, device)
        payload["adapter_validation"] = load_adapter_state(model, payload)
        return model, config, payload
    model = make_model(config, device, decoder_config=payload["decoder_config"])
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if any(not key.startswith("backbone.") for key in missing) or unexpected:
        raise ValueError(f"Checkpoint parameter mismatch: missing={missing}, unexpected={unexpected}")
    return model, config, payload


def learning_rate_factor(step, total_steps, warmup_steps):
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps > 1 and step >= total_steps - 1:
        return 0.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return 0.5 * (1 + math.cos(math.pi * min(1, max(0, progress))))


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
