#!/usr/bin/env python3
"""Create deterministic SYNTHETIC cached features for pipeline verification.

These features, labels, video IDs, and compatibility relations are artificial.
Metrics produced from them are not UAV-HINT benchmark results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo/data"))
    parser.add_argument("--feature-dim", type=int, default=512,
                        help="cached feature dimension; use 32 for the smallest CPU smoke run")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if args.feature_dim < 19 or args.feature_dim % 4:
        parser.error("--feature-dim must be >=19 and divisible by four for the smoke model")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    compatibility_path = output / "compatibility.json"
    generator = torch.Generator().manual_seed(args.seed)
    valid_triplets = [[action, short, (action + short) % 4] for action in range(7) for short in range(8)]
    records = []
    for split in ("train", "val", "test"):
        for index, (action, short, long) in enumerate(valid_triplets):
            # Each level occupies an independent subspace; the remaining channels
            # and token noise are synthetic. No image encoder is represented.
            latent = torch.zeros(args.feature_dim)
            latent[action] = 2.0
            latent[7 + short] = 2.0
            latent[15 + long] = 2.0
            length = 4 + index % 6
            tokens = latent.unsqueeze(0) + 0.05 * torch.randn(length, args.feature_dim, generator=generator)
            relative_path = Path("tokens") / split / f"{index:03d}.pt"
            token_path = output / relative_path
            token_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tokens, token_path)
            records.append({
                "video_id": f"SYNTHETIC_{split}_video_{index:03d}",
                "instance_id": "synthetic_person_0",
                "split": split,
                "tokens": relative_path.as_posix(),
                "text": "Synthetic demonstration features; no real video or human observations.",
                "labels": {"action": action, "short_term": short, "long_term": long},
                "synthetic": True,
            })
    manifest_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    compatibility_path.write_text(json.dumps({
        "valid_triplets": valid_triplets,
        "metadata": {
            "synthetic": True,
            "source": "Artificial demo rule: long_term = (action + short_term) modulo 4",
            "warning": "This is not the official UAV-HINT compatibility tensor or benchmark data.",
        },
    }, indent=2) + "\n", encoding="utf-8")
    config = {
        "manifest": str(manifest_path),
        "root": str(output),
        "labels": str(Path(__file__).resolve().parents[1] / "configs" / "labels.json"),
        "compatibility": str(compatibility_path),
        "lvp_weight": 0.2,
        "model": {
            "feature_dim": args.feature_dim,
            "num_queries": 2,
            "num_prefix": 2,
            "cross_attention_heads": 4,
            "dropout": 0.0,
            "num_classes": [7, 8, 4],
            "use_hqr": True,
            "use_ptu": True,
            "random_decoder": True,
            "decoder_dim": 48,
            "decoder_layers": 6,
            "decoder_heads": 4,
            "decoder_ff_dim": 96,
        },
        "data": {"input_mode": "tokens", "num_frames": 8, "image_size": 336},
        "backbone": {},
        "training": {
            "epochs": args.epochs,
            "batch_size": 8,
            "num_workers": 0,
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "seed": args.seed,
        },
    }
    config_path = output / "smoke.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "demo_summary.json").write_text(json.dumps({
        "synthetic": True,
        "purpose": "Pipeline smoke test with random frozen decoder; not paper reproduction metrics",
        "instances_per_split": 56,
        "generator_seed": args.seed,
        "feature_dim": args.feature_dim,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"synthetic": True, "manifest": str(manifest_path),
                      "compatibility": str(compatibility_path), "config": str(config_path),
                      "feature_dim": args.feature_dim,
                      "instances_per_split": 56}, indent=2))


if __name__ == "__main__":
    main()
