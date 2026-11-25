#!/usr/bin/env python3
"""Validate/normalize explicit annotation records without inventing official splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hintnet.data import UAVHintDataset, encode_labels, format_text, load_label_names, read_manifest, validate_manifest_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Input is JSONL or a JSON array of records with video_id, instance_id, "
        "split, frames, bboxes, text, and labels; tokens can replace frames/bboxes. "
        "Use --split-file for official video assignments: "
        "{\"train\":[\"video1\"],\"val\":[\"video2\"],\"test\":[\"video3\"]}. "
        "The tool does not infer label semantics or convert arbitrary Label Studio exports."
    ))
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, help="normalized output JSONL; omit to only validate")
    parser.add_argument("--split-file", type=Path, help="official video split assignment JSON")
    parser.add_argument("--labels", type=Path, help="custom ordered names for the three levels")
    parser.add_argument("--root", type=Path, help="base path for images/caches; default source directory")
    parser.add_argument("--input-mode", choices=("video", "tokens"), default="video")
    parser.add_argument("--global-only", action="store_true", help="allow records without bounding boxes")
    parser.add_argument("--check-files", action="store_true", help="verify every referenced file exists")
    args = parser.parse_args()
    records = read_manifest(args.source)
    labels = load_label_names(args.labels)
    if args.split_file:
        raw_assignments = json.loads(args.split_file.read_text(encoding="utf-8"))
        assignments = {}
        for split in ("train", "val", "test"):
            videos = raw_assignments.get(split)
            if not isinstance(videos, list):
                raise ValueError(f"split file requires a list for {split}")
            for video in videos:
                video = str(video)
                if video in assignments:
                    raise ValueError(f"duplicate video {video!r} in split assignment")
                assignments[video] = split
        for record in records:
            video = str(record["video_id"])
            if video not in assignments:
                raise ValueError(f"video {video!r} is absent from the supplied split file")
            if "split" in record and record["split"] != assignments[video]:
                raise ValueError(f"video {video!r}: record split contradicts split file")
            record["split"] = assignments[video]
    for record in records:
        values = encode_labels(record["labels"], labels).tolist()
        record["labels"] = dict(zip(("action", "short_term", "long_term"), values))
        record["text"] = format_text(record.get("text", record.get("tags")))
    # Validate via a temporary normalized manifest before writing the requested
    # output, so an invalid dataset never overwrites a user's prior output.
    import tempfile
    base = args.root.resolve() if args.root else args.source.resolve().parent
    with tempfile.TemporaryDirectory(prefix="hint-manifest-") as temporary:
        temporary_path = Path(temporary) / "manifest.jsonl"
        temporary_path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        counts = validate_manifest_splits(temporary_path)
        dataset = UAVHintDataset(temporary_path, root=base, labels_path=args.labels,
                                 input_mode=args.input_mode, use_local=not args.global_only)
        if args.check_files:
            for record in records:
                references = [record["tokens"]] if args.input_mode == "tokens" else record["frames"]
                for reference in references:
                    path = Path(reference)
                    path = path if path.is_absolute() else base / path
                    if not path.is_file():
                        raise FileNotFoundError(path)
        if args.output:
            # Moving the manifest must not change the interpretation of its paths.
            # Save absolute references whenever output moves to a new directory.
            if args.output.resolve().parent != base:
                for record in records:
                    if args.input_mode == "tokens":
                        path = Path(record["tokens"])
                        record["tokens"] = str(path if path.is_absolute() else base / path)
                    else:
                        record["frames"] = [str(Path(path) if Path(path).is_absolute() else base / path) for path in record["frames"]]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
        print(json.dumps({"instances": len(dataset), "videos": counts, "output": str(args.output) if args.output else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
