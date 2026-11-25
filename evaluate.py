#!/usr/bin/env python3
"""Evaluate HINT-Net mAP, accuracy, macro-F1, per-class AP and paper S_cons."""
import argparse
import csv
import json
from pathlib import Path

from hintnet.runtime import (choose_device, evaluate, load_saved_model,
                             make_dataset, make_loader, seed_everything)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--predictions", help="optional instance prediction JSONL output")
    parser.add_argument("--per-label-output", help="optional per-label AP/F1/support CSV output")
    parser.add_argument("--predict-only", action="store_true", help="allow records without labels")
    parser.add_argument("--input-mode", choices=("video", "tokens"))
    parser.add_argument("--actionclip-checkpoint", help="needed if evaluating a token-trained model on raw inputs")
    parser.add_argument("--t5-model", help="optional local T5-base directory or Hugging Face model ID")
    parser.add_argument("--local-files-only", action="store_true", help="load T5 only from local files/cache")
    args = parser.parse_args()
    overrides = {}
    for name in ("manifest", "root"):
        if getattr(args, name):
            overrides[name] = getattr(args, name)
    if args.manifest and not args.root:
        overrides["root"] = None
    if args.input_mode:
        overrides["data"] = {"input_mode": args.input_mode}
    if args.actionclip_checkpoint:
        overrides["backbone"] = {"checkpoint": args.actionclip_checkpoint}
    if args.t5_model:
        overrides.setdefault("model", {})["t5_model"] = args.t5_model
    if args.local_files_only:
        overrides.setdefault("model", {})["local_files_only"] = True
    for name in ("batch_size", "num_workers"):
        if getattr(args, name) is not None:
            overrides.setdefault("training", {})[name] = getattr(args, name)
    device = choose_device(args.device)
    model, config, saved = load_saved_model(args.checkpoint, device, overrides)
    seed_everything(config["training"]["seed"], config["training"]["cpu_threads"])
    dataset = make_dataset(config, args.split, require_labels=not args.predict_only,
                           label_names=saved["label_names"])
    loader = make_loader(dataset, config, device)
    metrics, predictions = evaluate(model, loader, device, saved["compatibility"], collect_predictions=bool(args.predictions))
    for row in predictions:
        row["prediction_names"] = {name: saved["label_names"][name][index] for name, index in row["predictions"].items()}
    training_checkpoint_policy = saved.get("checkpoint_selection", "validation_long_term_mAP")
    is_final_checkpoint = (Path(args.checkpoint).name == "last.pt"
                           and saved.get("epoch", -1) + 1 == config["training"]["epochs"])
    result = {"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()),
              "random_decoder": config["model"]["random_decoder"],
              "checkpoint_epoch": saved.get("epoch", -1) + 1,
              "checkpoint_selection": "final_epoch" if is_final_checkpoint else training_checkpoint_policy,
              "training_checkpoint_policy": training_checkpoint_policy,
              "input_mode": config["data"]["input_mode"],
              "num_instances": len(dataset), "metrics": metrics,
              "label_names": saved["label_names"],
              "metric_definitions": {
                  "mAP": "Macro mean of one-vs-rest class AP from softmax scores; absent-positive AP=0",
                  "accuracy": "Independent argmax accuracy for each semantic level",
                  "macro_f1": "Unweighted mean F1 over every declared class; zero_division=0",
                  "S_cons": "Paper Sec.3.5 Eqs.(1)-(2): mean M[argmax(action),argmax(short_term),argmax(long_term)]",
                  "triplet_accuracy": "Fraction with all three argmax labels correct; distinct from S_cons",
                  "means": "Unweighted means over the three semantic levels; all fractions use [0,1]",
              }}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions:
        path = Path(args.predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    if args.per_label_output:
        if not metrics:
            raise ValueError("Per-label metrics require at least one fully labeled evaluation sample")
        path = Path(args.per_label_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["level", "label_id", "label", "AP", "AP_percent", "F1", "F1_percent", "support"])
            writer.writeheader()
            for level, names in saved["label_names"].items():
                for index, name in enumerate(names):
                    ap = metrics[f"{level}_class_AP"][index]
                    f1 = metrics[f"{level}_class_F1"][index]
                    writer.writerow({"level": level, "label_id": index, "label": name,
                                     "AP": ap, "AP_percent": 100 * ap,
                                     "F1": f1, "F1_percent": 100 * f1,
                                     "support": metrics[f"{level}_class_support"][index]})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
