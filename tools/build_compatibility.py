#!/usr/bin/env python3
"""Build the user-requested compatibility support from ALL complete raw tracks.

The scope deliberately includes train, validation and test labels. This is an
empirical full-dataset graph, not an independent official semantic constraint.
Only exact observed triples are valid; adjacent edges are never closed into new
triples. Counts use complete person tracks once, not frames or choice rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hintnet.data import LABELS, LEVELS, encode_labels, load_label_names, read_manifest
from tools.convert_uav_hint import parse_annotations


SPLITS = ("train", "val", "test")
CSV_FIELDS = (
    "action_id", "action_name", "short_term_id", "short_term_name",
    "long_term_id", "long_term_name", "valid", "count_total",
    "count_train", "count_val", "count_test",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def build_compatibility(annotations: Path, manifest: Path, splits: Path,
                        labels: Path) -> tuple[dict, list[dict]]:
    """Validate exact raw-track/manifest joins, then count all three splits."""
    annotations, manifest, splits, labels = map(Path, (annotations, manifest, splits, labels))
    vocabulary = load_label_names(labels)
    if vocabulary != LABELS:
        raise ValueError("label names/order must match the canonical IDs used by the raw annotation parser")
    tracks, audit = parse_annotations(annotations)
    source_tracks = {}
    for track in tracks:
        key = (str(track["video_id"]), str(track["instance_id"]))
        if key in source_tracks:
            raise ValueError(f"duplicate complete raw track: {key}")
        source_tracks[key] = track

    split_document = json.loads(splits.read_text(encoding="utf-8"))
    assignments = {}
    for split in SPLITS:
        values = split_document.get(split)
        if not isinstance(values, list):
            raise ValueError(f"split file requires a list for {split}")
        for video_id in values:
            video_id = str(video_id)
            if video_id in assignments:
                raise ValueError(f"duplicate video in split file: {video_id}")
            assignments[video_id] = split
    if set(assignments) != set(audit["video_names"]):
        raise ValueError("split assignments must cover exactly every video ID in the raw annotation export")

    records = read_manifest(manifest)
    manifest_tracks = {}
    for record in records:
        key = (str(record["video_id"]), str(record["instance_id"]))
        if key in manifest_tracks:
            raise ValueError(f"duplicate manifest track: {key}; statistics require one record per raw person track")
        manifest_tracks[key] = record
    if set(manifest_tracks) != set(source_tracks):
        missing = sorted(set(source_tracks) - set(manifest_tracks))
        extra = sorted(set(manifest_tracks) - set(source_tracks))
        raise ValueError(f"manifest/raw track IDs differ: missing={missing}, extra={extra}")

    total_counts = Counter()
    split_counts = {split: Counter() for split in SPLITS}
    split_tracks = Counter()
    split_videos = defaultdict(set)
    for key, track in source_tracks.items():
        record = manifest_tracks[key]
        source_labels = tuple(encode_labels(track["labels"], vocabulary).tolist())
        manifest_labels = tuple(encode_labels(record.get("labels"), vocabulary).tolist())
        if source_labels != manifest_labels:
            raise ValueError(f"manifest/raw labels differ for {key}: raw={source_labels}, manifest={manifest_labels}")
        split = assignments[key[0]]
        if record.get("split") != split:
            raise ValueError(f"manifest/split assignment differs for {key}: {record.get('split')!r} vs {split!r}")
        total_counts[source_labels] += 1
        split_counts[split][source_labels] += 1
        split_tracks[split] += 1
        split_videos[split].add(key[0])

    shape = [len(vocabulary[level]) for level in LEVELS]
    valid_triplets = sorted(total_counts)
    possible = shape[0] * shape[1] * shape[2]
    held_out_only = [triplet for triplet in valid_triplets if not split_counts["train"][triplet]]
    rows = []
    for action, short, long in product(*(range(size) for size in shape)):
        triple = (action, short, long)
        rows.append({
            "action_id": action, "action_name": vocabulary["action"][action],
            "short_term_id": short, "short_term_name": vocabulary["short_term"][short],
            "long_term_id": long, "long_term_name": vocabulary["long_term"][long],
            "valid": int(total_counts[triple] > 0), "count_total": total_counts[triple],
            **{f"count_{split}": split_counts[split][triple] for split in SPLITS},
        })
    source_provenance = {
        "annotations": _source(annotations), "manifest": _source(manifest),
        "splits": _source(splits), "labels": _source(labels),
    }
    split_metadata_path = splits.with_suffix(".metadata.json")
    if split_metadata_path.is_file():
        source_provenance["split_metadata"] = _source(split_metadata_path)
    document = {
        "valid_triplets": [list(triplet) for triplet in valid_triplets],
        "metadata": {
            "user_requested": True,
            "kind": "empirical_full_dataset_triplet_support",
            "scope": "all_complete_raw_annotation_tracks_in_train_val_test",
            "included_splits": list(SPLITS),
            "counting_unit": "one complete raw annotated person track",
            "official_semantic_graph": False,
            "independent_of_validation_test_labels": False,
            "definition": "M[a,s,l]=1 exactly when that complete triplet is observed in the supplied full raw annotation export; violation mask=1-M",
            "no_adjacent_edge_closure": True,
            "interpretation": "Unobserved combinations are assigned zero by this empirical definition; absence does not independently establish semantic impossibility.",
            "label_order": list(LEVELS), "label_names": vocabulary,
            "shape": shape, "possible_triplets": possible,
            "valid_triplets": len(valid_triplets), "violation_triplets": possible - len(valid_triplets),
            "raw_tasks": audit["tasks"], "raw_box_tracks": audit["box_tracks"],
            "complete_tracks": len(source_tracks), "manifest_tracks": len(manifest_tracks),
            "raw_manifest_ids_and_labels_match": True,
            "skipped_tracks": audit["skipped_tracks"],
            "complete_tracks_per_split": {split: split_tracks[split] for split in SPLITS},
            "all_split_video_counts": {split: sum(value == split for value in assignments.values()) for split in SPLITS},
            "complete_track_videos_per_split": {split: len(split_videos[split]) for split in SPLITS},
            "observed_triplets_per_split": {split: len(split_counts[split]) for split in SPLITS},
            "held_out_only_triplets_count": len(held_out_only),
            "held_out_only_triplets": [list(triplet) for triplet in held_out_only],
            "source_label_aliases": audit["label_aliases"],
            "draft_annotations_used": False,
            "provenance": source_provenance,
        },
        "triplet_counts": [
            {"triplet": list(triplet), "total": total_counts[triplet],
             **{split: split_counts[split][triplet] for split in SPLITS}}
            for triplet in valid_triplets
        ],
    }
    return document, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("datasets/prepared/manifest.jsonl"))
    parser.add_argument("--splits", type=Path, default=Path("datasets/prepared/splits.json"))
    parser.add_argument("--labels", type=Path, default=Path("configs/labels.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared/compatibility_full_dataset.json"))
    parser.add_argument("--csv", type=Path, help="defaults to the output JSON path with a .csv suffix")
    args = parser.parse_args()
    document, rows = build_compatibility(args.annotations, args.manifest, args.splits, args.labels)
    csv_path = args.csv if args.csv is not None else args.output.with_suffix(".csv")
    if csv_path.resolve() == args.output.resolve():
        parser.error("JSON and CSV output paths must differ")
    protected = {path.resolve() for path in (args.annotations, args.manifest, args.splits, args.labels)}
    if args.output.resolve() in protected or csv_path.resolve() in protected:
        parser.error("output paths cannot overwrite a source input")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = document["metadata"]
    print(json.dumps({"json": str(args.output.resolve()), "csv": str(csv_path.resolve()),
                      "complete_tracks": metadata["complete_tracks"], "shape": metadata["shape"],
                      "valid_triplets": metadata["valid_triplets"], "violation_triplets": metadata["violation_triplets"],
                      "complete_tracks_per_split": metadata["complete_tracks_per_split"],
                      "observed_triplets_per_split": metadata["observed_triplets_per_split"],
                      "held_out_only_triplets_count": metadata["held_out_only_triplets_count"],
                      "user_requested": True, "includes_validation_and_test_labels": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
