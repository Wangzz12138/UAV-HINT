#!/usr/bin/env python3
"""Export a final-50, 336 HINT-Net snapshot as portable trainable-only weights.

No dataset, original machine path, optimizer or RNG state is written. The real
frozen T5 decoder must be supplied when loading the adapter, and its exact
tensor fingerprint is checked automatically by hintnet.runtime.load_saved_model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from hintnet.checkpoints import (ACTIONCLIP_SHA256, ADAPTER_FORMAT, T5_MODEL,
                                T5_REVISION, frozen_decoder_state,
                                tensor_state_sha256, trainable_state)
from hintnet.model import HINTNet, ModelConfig


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_adapter(checkpoint, output):
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source.get("format_version") != 1 or source.get("epoch") != 49:
        raise ValueError("Expected a format-version1 final epoch50 checkpoint")
    config = deepcopy(source["config"])
    if (config["data"]["image_size"] != 336 or config["model"]["random_decoder"]
            or not config["model"]["use_hqr"] or not config["model"]["use_ptu"]
            or config["training"]["epochs"] != 50):
        raise ValueError("Export requires the real-pretrained full HQR/PTU 336 model trained for50 epochs")
    if file_sha256(config["backbone"]["checkpoint"]) != ACTIONCLIP_SHA256:
        raise ValueError("The source encoder must be the released ActionCLIP ViT-B/16 8-frame pretrained checkpoint")
    model = HINTNet(ModelConfig(**config["model"]), backbone=None, decoder_config=source["decoder_config"])
    model.load_state_dict(source["model"], strict=True)
    weights = trainable_state(model)
    decoder_hash = tensor_state_sha256(frozen_decoder_state(model))
    if len(weights) != 20 or sum(value.numel() for value in weights.values()) != 2087731:
        raise ValueError("Expected the complete HQR/PTU HINT-Net trainable state")
    config["model"].update(t5_model=T5_MODEL, t5_revision=T5_REVISION, local_files_only=False)
    config["manifest"] = "data/manifest.jsonl"
    config["root"] = None
    config["labels"] = "configs/labels.json"
    config["compatibility"] = "configs/compatibility_81.json"
    config["data"]["input_mode"] = "tokens"
    config["backbone"]["checkpoint"] = "pretrained/actionclip_vit_b16_8f.pt"
    config["backbone"]["initialize_for_cached_inputs"] = False
    config["training"]["num_workers"] = 0
    payload = {"format": ADAPTER_FORMAT, "format_version": 1,
               "epoch": 49, "global_step": source["global_step"], "config": config,
               "model": weights, "label_names": source["label_names"],
               "compatibility": source["compatibility"].detach().cpu(),
               "checkpoint_selection": "fixed_final_epoch50",
               "source": {"training_seed": config["training"]["seed"], "training_epochs": 50,
                          "input_resolution": 336, "original_checkpoint_sha256": file_sha256(checkpoint),
                          "frozen_decoder_sha256": decoder_hash, "t5_model": T5_MODEL,
                          "t5_revision": T5_REVISION, "actionclip_checkpoint_sha256": ACTIONCLIP_SHA256,
                          "trainable_tensor_count": len(weights), "trainable_parameters": 2087731,
                          "note": "Frozen pretrained weights excluded. Exact frozen decoder identity is verified during loading. Inference-only adapter; use a full training snapshot for optimizer/RNG resume."}}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    torch.save(payload, output)
    metadata = {"format": ADAPTER_FORMAT, "file": output.name, "bytes": output.stat().st_size,
                "sha256": file_sha256(output), "source": payload["source"],
                "label_names": payload["label_names"]}
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_adapter(args.checkpoint, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
