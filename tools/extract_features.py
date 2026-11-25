#!/usr/bin/env python3
"""Cache deterministic frozen ActionCLIP tokens and write a portable manifest.

The resulting cache uses uniform evaluation sampling and no flip augmentation.
Training from this cache therefore differs from raw-video training with temporal
sampling and spatial augmentation. No labels are used to compute the features.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from hintnet.backbones import FrozenActionCLIP
from hintnet.data import UAVHintDataset, collate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None, help="Root of source frame paths (default: source manifest directory)")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--visual-token-mode", choices=("pooled", "frames"), default="pooled")
    parser.add_argument("--image-batch-size", type=int, default=0)
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--no-text", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--enable-cudnn", action="store_true", help="Enable cuDNN visual convolution (disabled by default for repeatability on this workstation)")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing cache and output manifest")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        parser.error("batch-size/image-size must be positive and workers nonnegative")
    if args.no_global and args.no_local:
        parser.error("Feature extraction requires at least one video stream")

    source_manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_manifest = (args.output_manifest or (output_dir / "manifest.jsonl")).expanduser().resolve()
    if source_manifest == output_manifest:
        parser.error("Output manifest must differ from the source manifest")
    dataset = UAVHintDataset(
        source_manifest,
        root=args.root.expanduser().resolve() if args.root else None,
        split=args.split,
        num_frames=args.num_frames,
        image_size=args.image_size,
        training=False,
        input_mode="video",
        labels_path=args.labels,
        use_global=not args.no_global,
        use_local=not args.no_local,
        require_labels=False,
    )
    token_dir = output_dir / "tokens"
    token_paths = [token_dir / f"{index:08d}.pt" for index in range(len(dataset))]
    if not args.overwrite:
        existing = next((path for path in [output_manifest, *token_paths] if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output already exists: {existing}; choose a new output directory or use --overwrite")
    encoder = FrozenActionCLIP(
        args.checkpoint,
        num_frames=args.num_frames,
        input_resolution=args.image_size,
        use_global=not args.no_global,
        use_local=not args.no_local,
        use_text=not args.no_text,
        visual_token_mode=args.visual_token_mode,
        normalize=not args.no_normalize,
        image_batch_size=args.image_batch_size,
        disable_cudnn=not args.enable_cudnn,
    ).to(args.device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=collate_batch, pin_memory=str(args.device).startswith("cuda"))
    provenance = {
        "encoder": "ActionCLIP",
        "checkpoint": encoder.checkpoint_path,
        "num_frames": args.num_frames,
        "image_size": encoder.input_resolution,
        "pretrained_image_size": encoder.clip.visual.input_resolution,
        "visual_position_embedding": ("original pretrained grid unchanged" if encoder.input_resolution == encoder.clip.visual.input_resolution else
                                      "bicubic spatial interpolation, align_corners=False; CLS and registered pretrained parameters unchanged"),
        "visual_token_mode": args.visual_token_mode,
        "normalized": not args.no_normalize,
        "global": not args.no_global,
        "local": not args.no_local,
        "text": not args.no_text,
        "sampling": "uniform_deterministic",
        "spatial_augmentation": False,
        "cudnn_disabled": not args.enable_cudnn,
    }
    token_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = output_manifest.with_name(output_manifest.name + ".tmp")
    offset = 0
    try:
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            for batch in loader:
                videos = {key: batch[key].to(args.device, non_blocking=True)
                          for key in ("video", "local_video") if key in batch}
                tokens, masks = encoder(**videos, text=batch["text"])
                for local_index in range(tokens.shape[0]):
                    sample_index = offset + local_index
                    token_path = token_paths[sample_index]
                    temporary_token = token_path.with_suffix(".pt.tmp")
                    torch.save({"tokens": tokens[local_index].cpu(), "token_mask": masks[local_index].cpu()}, temporary_token)
                    temporary_token.replace(token_path)
                    record = dict(dataset.records[sample_index])
                    # Rebase retained frame paths so the new manifest is also
                    # meaningful outside the source manifest's directory.
                    record["frames"] = [str((dataset.root / path).resolve()) for path in record["frames"]]
                    record["tokens"] = os.path.relpath(token_path, output_manifest.parent)
                    record["feature_cache"] = provenance
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                offset += tokens.shape[0]
                print(f"Cached {offset}/{len(dataset)} samples", flush=True)
        temporary_manifest.replace(output_manifest)
    except BaseException:
        temporary_manifest.unlink(missing_ok=True)
        raise
    print(json.dumps({"manifest": str(output_manifest), "samples": offset, "provenance": provenance}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
