#!/usr/bin/env python3
"""Cache both exact frozen-ActionCLIP variants for fixed-frame observations.

The variants are the original observation and the existing raw pipeline's
synchronized horizontal PIL flip BEFORE bicubic resizing. Text never changes.
All observations must already contain exactly num_frames frames: this cache
does not replace stochastic temporal sampling on longer source sequences.

Each .pt contains tokens [S,512], flipped_tokens [S,512], token_mask [S].
Evaluation always uses tokens. A training loader may select flipped_tokens with
the same 0.5 probability as raw training. This tool does not alter that loader.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, Dataset

from hintnet.backbones import FrozenActionCLIP
from hintnet.data import UAVHintDataset, collate_batch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FlippedObservationDataset(UAVHintDataset):
    """Use the raw preprocessing function with its pre-resize flip forced on."""

    def _image_tensor(self, image, flip):
        return super()._image_tensor(image, True)


class PairedObservationDataset(Dataset):
    def __init__(self, original: UAVHintDataset, flipped: UAVHintDataset):
        self.original, self.flipped = original, flipped

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        return self.original[index], self.flipped[index]


def collate_variants(samples):
    return collate_batch([sample[0] for sample in samples]), collate_batch([sample[1] for sample in samples])


def encoder_inputs(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items() if key in ("video", "local_video", "text")}


def validation_indices(dataset, requested):
    if requested:
        indices = sorted(set(int(value.strip()) for value in requested.split(",")))
    else:
        first_by_scene = {}
        for index, record in enumerate(dataset.records):
            context = record.get("text", record.get("tags"))
            scene = str(context.get("scene", "unknown")) if isinstance(context, Mapping) else "unknown"
            first_by_scene.setdefault(scene, index)
        indices = sorted(set([*first_by_scene.values(), len(dataset) - 1]))
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("validation indices must refer to existing dataset rows")
    return indices


def validate_cache(encoder, raw_training, original, flipped, token_paths, indices, device, tolerance):
    checks = []
    maximum = 0.0
    for index in indices:
        payload = torch.load(token_paths[index], map_location="cpu", weights_only=True)
        row = {"index": index, "video_id": str(original.records[index]["video_id"]),
               "instance_id": str(original.records[index]["instance_id"])}
        for variant, draw, key, reference_dataset in (("original", 0.75, "tokens", original),
                                                      ("flipped", 0.25, "flipped_tokens", flipped)):
            # Run the actual raw TRAINING branch, including its singleton
            # torch.randint calls, while explicitly selecting the flip outcome.
            with patch("hintnet.data.torch.rand", return_value=torch.tensor(draw)):
                sample = raw_training[index]
            reference = reference_dataset[index]
            pixel_differences = {name: float((sample[name] - reference[name]).abs().max())
                                 for name in ("video", "local_video") if name in sample}
            if sample["text"] != reference["text"] or any(pixel_differences.values()):
                raise AssertionError(f"Cached preprocessing differs from raw {variant} branch: {row}")
            tokens, mask = encoder(**encoder_inputs(collate_batch([sample]), device))
            if tokens.requires_grad or not torch.equal(mask.cpu()[0], payload["token_mask"]):
                raise AssertionError("Invalid frozen encoder gradient state or cached mask")
            difference = float((tokens[0].cpu() - payload[key]).abs().max())
            if difference > tolerance:
                raise AssertionError(f"Raw/cache {variant} difference {difference} exceeds {tolerance}: {row}")
            row[f"{variant}_raw_pixel_max_abs"] = pixel_differences
            row[f"{variant}_tokens_max_abs"] = difference
            maximum = max(maximum, difference)
        checks.append(row)
    frozen = not encoder.training and all(not parameter.requires_grad and parameter.grad is None
                                         for parameter in encoder.parameters())
    if not frozen:
        raise AssertionError("Encoder must remain frozen, in evaluation mode, with no parameter gradients")
    return {"status": "passed", "tolerance": tolerance, "maximum_token_absolute_difference": maximum,
            "raw_pipeline": "training=True; force torch.rand flip outcome; pre-resize PIL mirror",
            "frozen_eval_no_parameter_gradients": frozen, "checked_rows": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/prepared/manifest.jsonl"))
    parser.add_argument("--config", type=Path, default=None, help="Read the run's raw data/backbone settings")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/actionclip_augmented_cache"))
    parser.add_argument("--output-manifest", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--validation-indices", default=None, help="Comma-separated rows; default: one per scene and last row")
    parser.add_argument("--validation-tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.image_batch_size < 1:
        parser.error("batch sizes must be positive and num-workers nonnegative")
    if not 0 < args.validation_tolerance <= 1e-6:
        parser.error("validation-tolerance must be positive and at most 1e-6")
    started = time.perf_counter()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    data_config, backbone_config = config.get("data", {}), config.get("backbone", {})
    num_frames = args.num_frames or data_config.get("num_frames", 8)
    checkpoint = Path(args.checkpoint or backbone_config.get("checkpoint")
                      or "pretrained/actionclip_vit_b16_8f.pt").expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_manifest = (args.output_manifest or output_dir / "manifest.jsonl").expanduser().resolve()
    if manifest == output_manifest:
        parser.error("Output manifest must differ from the original manifest")
    options = dict(root=args.root or config.get("root"), num_frames=num_frames,
                   image_size=data_config.get("image_size", 336), input_mode="video",
                   labels_path=args.labels or config.get("labels"), require_labels=False,
                   use_global=data_config.get("use_global", True), use_local=data_config.get("use_local", True))
    original = UAVHintDataset(manifest, training=False, **options)
    flipped = FlippedObservationDataset(manifest, training=False, **options)
    raw_training = UAVHintDataset(manifest, training=True, **options)
    if any(len(record["frames"]) != num_frames for record in original.records):
        raise ValueError("Exact two-variant caching requires len(frames) == num_frames in EVERY record")
    normalize = backbone_config.get("normalize", True)
    disable_cudnn = backbone_config.get("disable_cudnn", True)
    if not normalize or not disable_cudnn:
        raise ValueError("This verified cache requires normalized tokens and the stable native-convolution encoder path")
    token_dir = output_dir / "tokens"
    token_paths = [token_dir / f"{index:08d}.pt" for index in range(len(original))]
    validation_path = output_dir / "cache_validation.json"
    provenance_path = output_dir / "cache_provenance.json"
    if not args.overwrite:
        existing = next((path for path in [output_manifest, validation_path, provenance_path, *token_paths] if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Output exists: {existing}; use another output directory or explicit --overwrite")
    indices = validation_indices(original, args.validation_indices)
    encoder = FrozenActionCLIP(checkpoint, num_frames=num_frames,
                               input_resolution=options["image_size"],
                               use_global=options["use_global"], use_local=options["use_local"],
                               use_text=backbone_config.get("use_text", True),
                               visual_token_mode=backbone_config.get("visual_token_mode", "pooled"),
                               normalize=True, disable_cudnn=True,
                               image_batch_size=args.image_batch_size).to(args.device).eval()
    provenance = {"format_version": 1, "encoder": "ActionCLIP", "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256_file(checkpoint), "source_manifest": str(manifest),
                  "source_manifest_sha256": sha256_file(manifest), "samples": len(original),
                  "num_frames": num_frames, "all_records_have_exact_fixed_frame_count": True,
                  "image_size": options["image_size"], "normalized": True, "cudnn_disabled": True,
                  "pretrained_image_size": encoder.clip.visual.input_resolution,
                  "visual_position_embedding": ("original pretrained grid unchanged" if encoder.input_resolution == encoder.clip.visual.input_resolution else
                                                "bicubic spatial interpolation, align_corners=False; CLS and registered pretrained parameters unchanged"),
                  "visual_token_mode": encoder.visual_token_mode,
                  "global": encoder.use_global, "local": encoder.use_local, "text": encoder.use_text,
                  "text_unchanged_between_variants": True, "image_batch_size": args.image_batch_size,
                  "spatial_augmentation": "synchronized horizontal PIL flip before bicubic resize",
                  "augmentation_variants": ["original", "synchronized_horizontal_flip"],
                  "training_flip_probability": 0.5, "evaluation_variant": "original",
                  "temporal_sampling": "all existing fixed observation frames; no resampling",
                  "labels_used_by_encoder": False}
    loader = DataLoader(PairedObservationDataset(original, flipped), batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, collate_fn=collate_variants,
                        pin_memory=str(args.device).startswith("cuda"))
    token_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = output_manifest.with_name(output_manifest.name + ".tmp")
    offset = 0
    try:
        with temporary_manifest.open("w", encoding="utf-8") as handle:
            for ordinary_batch, flipped_batch in loader:
                if ordinary_batch["text"] != flipped_batch["text"]:
                    raise AssertionError("Flip augmentation must not change contextual text")
                tokens, masks = encoder(**encoder_inputs(ordinary_batch, args.device))
                flipped_tokens, flipped_masks = encoder(**encoder_inputs(flipped_batch, args.device))
                if not torch.equal(masks, flipped_masks) or not bool(torch.isfinite(tokens).all() and torch.isfinite(flipped_tokens).all()):
                    raise AssertionError("Non-finite tokens or variant mask mismatch")
                for local_index in range(tokens.shape[0]):
                    index = offset + local_index
                    path = token_paths[index]
                    temporary_token = path.with_suffix(".pt.tmp")
                    torch.save({"tokens": tokens[local_index].cpu(),
                                "flipped_tokens": flipped_tokens[local_index].cpu(),
                                "token_mask": masks[local_index].cpu()}, temporary_token)
                    temporary_token.replace(path)
                    record = dict(original.records[index])
                    record["frames"] = [str((original.root / value).resolve()) for value in record["frames"]]
                    record["tokens"] = os.path.relpath(path, output_manifest.parent)
                    record["feature_cache"] = provenance
                    record["augmentation_cache"] = {"format_version": 1, "original_key": "tokens",
                                                     "flipped_key": "flipped_tokens", "mask_key": "token_mask",
                                                     "flip_application": "PIL before bicubic resize",
                                                     "training_flip_probability": 0.5, "evaluation_variant": "original",
                                                     "num_frames": num_frames,
                                                     "rng_contract": "One Bernoulli torch.rand draw followed by num_frames singleton torch.randint draws during training"}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                offset += tokens.shape[0]
                print(f"Cached original + PIL-flipped {offset}/{len(original)} observations; elapsed {time.perf_counter()-started:.1f}s", flush=True)
        validation = validate_cache(encoder, raw_training, original, flipped, token_paths, indices,
                                    args.device, args.validation_tolerance)
        validation.update({"samples_cached": offset, "source_manifest": str(manifest),
                           "cache_manifest": str(output_manifest),
                           "elapsed_seconds": round(time.perf_counter() - started, 3)})
        validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        temporary_manifest.replace(output_manifest)
    except BaseException:
        temporary_manifest.unlink(missing_ok=True)
        raise
    print(json.dumps({"manifest": str(output_manifest), "validation": str(validation_path),
                      "samples": offset, "maximum_raw_cache_token_difference": validation["maximum_token_absolute_difference"],
                      "elapsed_seconds": validation["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
