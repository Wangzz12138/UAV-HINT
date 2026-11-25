#!/usr/bin/env python3
"""Train HINT-Net on UAV-HINT raw clips or frozen ActionCLIP token caches."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from hintnet.data import LEVELS, load_label_names
from hintnet.objectives import HierarchicalLoss
from hintnet.runtime import (atomic_save, choose_device, evaluate, get_compatibility,
                             learning_rate_factor, load_config, load_saved_model,
                             make_dataset, make_loader, make_model, model_inputs,
                             seed_everything)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hintnet_336.json")
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--compatibility")
    parser.add_argument("--actionclip-checkpoint")
    parser.add_argument("--t5-model")
    parser.add_argument("--local-files-only", action="store_true", help="load T5 only from local files/cache")
    parser.add_argument("--output", help="run directory; defaults to outputs/hintnet or the resumed run directory")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lvp-weight", type=float)
    parser.add_argument("--input-mode", choices=("video", "tokens"))
    parser.add_argument("--no-hqr", action="store_true")
    parser.add_argument("--no-ptu", action="store_true")
    parser.add_argument("--no-validation", action="store_true", help="train all train records, never evaluate or select on test; use final epoch")
    parser.add_argument("--checkpoint-every", type=int, help="save last checkpoint every N epochs and at the final epoch")
    parser.add_argument("--resume", help="resume at an epoch boundary using the saved run config")
    args = parser.parse_args()
    device = choose_device(args.device)
    if args.resume:
        model, config, saved = load_saved_model(args.resume, device)
        if saved.get("format") == "hintnet_adapter_v1":
            raise ValueError("Adapter checkpoints are for inference. Resume training from a full last.pt snapshot containing optimizer and RNG states.")
        # Changing the run schedule or data on resume invalidates its optimizer
        # and scheduler state; start a new run for those changes.
        disallowed = (args.manifest, args.root, args.compatibility, args.actionclip_checkpoint,
                      args.t5_model, args.epochs, args.batch_size, args.num_workers,
                      args.lvp_weight, args.input_mode, args.no_hqr, args.no_ptu,
                      args.no_validation, args.checkpoint_every, args.local_files_only)
        if any(value is not None and value is not False for value in disallowed):
            raise ValueError("--resume uses the saved configuration; omit training/data overrides")
    else:
        config = load_config(args.config)
        for key in ("manifest", "root", "compatibility", "lvp_weight"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        for key in ("epochs", "batch_size", "num_workers"):
            if getattr(args, key) is not None:
                config["training"][key] = getattr(args, key)
        if args.actionclip_checkpoint:
            config["backbone"]["checkpoint"] = args.actionclip_checkpoint
        if args.t5_model:
            config["model"]["t5_model"] = args.t5_model
        if args.local_files_only:
            config["model"]["local_files_only"] = True
        if args.input_mode:
            config["data"]["input_mode"] = args.input_mode
        if args.no_hqr:
            config["model"]["use_hqr"] = False
        if args.no_ptu:
            config["model"]["use_ptu"] = False
        if args.no_validation:
            config["training"]["validate"] = False
        if args.checkpoint_every is not None:
            config["training"]["checkpoint_every"] = args.checkpoint_every
        model, saved = None, None
    training = config["training"]
    if training["epochs"] < 1 or training["batch_size"] < 1 or training["num_workers"] < 0:
        raise ValueError("epochs/batch_size must be positive, num_workers nonnegative")
    if not 0 <= training["warmup_ratio"] < 1 or training["lr"] <= 0:
        raise ValueError("warmup_ratio must be in [0,1) and learning rate positive")
    validate = training.get("validate", True)
    checkpoint_every = training.get("checkpoint_every", 1)
    if not isinstance(validate, bool) or type(checkpoint_every) is not int or checkpoint_every < 1:
        raise ValueError("training.validate must be boolean and checkpoint_every a positive integer")
    seed_everything(training["seed"], training["cpu_threads"])
    # Persist absolute resource paths so saved runs work from another directory.
    for key in ("manifest", "root", "labels", "compatibility"):
        if config[key]:
            config[key] = str(Path(config[key]).resolve())
    if config["backbone"]["checkpoint"]:
        config["backbone"]["checkpoint"] = str(Path(config["backbone"]["checkpoint"]).resolve())
    names = load_label_names(saved["label_names"] if saved else config["labels"])
    if tuple(len(names[level]) for level in LEVELS) != tuple(config["model"]["num_classes"]):
        raise ValueError("Label vocabulary sizes must match model.num_classes")
    compatibility = saved["compatibility"] if saved else get_compatibility(config)
    train_data = make_dataset(config, "train", training=True, label_names=names)
    val_data = make_dataset(config, "val", label_names=names) if validate else None
    # Known annotations must agree with the supplied compatibility and ID order.
    if compatibility is not None:
        for dataset in (train_data, val_data):
            if dataset is None:
                continue
            for record in dataset.records:
                from hintnet.data import encode_labels
                labels = encode_labels(record["labels"], names)
                if not compatibility[tuple(labels.tolist())]:
                    raise ValueError(f"Ground truth is excluded by compatibility: {record['video_id']}/{record['instance_id']}")
    generator = torch.Generator().manual_seed(training["seed"])
    train_loader = make_loader(train_data, config, device, training=True, generator=generator)
    val_loader = make_loader(val_data, config, device) if val_data is not None else None
    if model is None:
        model = make_model(config, device)
    criterion = HierarchicalLoss(compatibility, config["lvp_weight"]).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=training["lr"], weight_decay=training["weight_decay"])
    total_steps = training["epochs"] * len(train_loader)
    warmup_steps = int(total_steps * training["warmup_ratio"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: learning_rate_factor(step, total_steps, warmup_steps))
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch, best, global_step = 0, -1.0, 0
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start_epoch, best, global_step = saved["epoch"] + 1, saved["best_long_term_mAP"], saved["global_step"]
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        if device.type == "cuda" and saved["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
        generator.set_state(saved["loader_rng"])
    output = Path(args.output or (Path(args.resume).resolve().parent if args.resume else "outputs/hintnet"))
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.pt").exists():
        raise FileExistsError(f"Run already exists at {output}; choose another --output or use --resume")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"device": str(device), "train_instances": len(train_data),
                      "val_instances": len(val_data) if val_data is not None else 0,
                      "validation_enabled": validate,
                      "checkpoint_selection": "validation_long_term_mAP" if validate else "final_epoch",
                      "random_decoder": config["model"]["random_decoder"],
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, ensure_ascii=False), flush=True)
    for epoch in range(start_epoch, training["epochs"]):
        model.train()
        sums = {name: 0.0 for name in ("total", "classification", "lvp")}
        count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(**model_inputs(batch, device))
            # Logical joint probabilities are evaluated in float32 outside AMP.
            losses = criterion(logits, batch["labels"].to(device))
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), training["grad_clip"], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            size = batch["labels"].shape[0]
            count += size
            for name in sums:
                sums[name] += float(losses[name].detach()) * size
        metrics = evaluate(model, val_loader, device, compatibility)[0] if validate else None
        improved = validate and metrics["long_term_mAP"] > best
        if validate:
            best = max(best, metrics["long_term_mAP"])
        row = {"epoch": epoch + 1, "global_step": global_step,
               "lr": optimizer.param_groups[0]["lr"],
               "train_loss": {name: value / count for name, value in sums.items()},
               "val": metrics}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        save_checkpoint = (epoch + 1) % checkpoint_every == 0 or epoch + 1 == training["epochs"] or improved
        if save_checkpoint:
            state = {"format_version": 1, "epoch": epoch, "global_step": global_step,
                 "config": config, "decoder_config": model.decoder.stack.config.to_dict(),
                 "model": {key: value for key, value in model.state_dict().items() if not key.startswith("backbone.")},
                 "compatibility": compatibility, "label_names": names,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "best_long_term_mAP": best,
                 "checkpoint_selection": "validation_long_term_mAP" if validate else "final_epoch",
                 "loader_rng": generator.get_state(),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                         "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
            atomic_save(state, output / "last.pt")
            if improved:
                atomic_save(state, output / "best.pt")
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
