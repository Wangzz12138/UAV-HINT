#!/usr/bin/env python3
"""Convert the supplied UAV-HINT Label Studio video annotations to raw clips.

No official benchmark split is invented. Supply official assignments with
--split-file, or explicitly opt into a reproducible provisional split. The
default observation protocol uses only the first half of each annotated track;
this is a reproduction choice because the paper supplies no observation horizon.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hintnet.data import LEVELS, encode_labels


FIELD_MAP = {"current_action": "action", "short_term_intention": "short_term",
             "high_level_intention": "long_term"}
ALIASES = {"short_term": {"talking to someone": "Talking"},
           "long_term": {"goal-directed movement": "Going somewhere"}}
CONTEXT_FIELDS = ("scene", "time", "weather", "age", "gender", "state")


def single_choice(result: dict[str, Any]) -> str:
    choices = result["value"].get("choices", [])
    if len(choices) != 1 or not isinstance(choices[0], str):
        raise ValueError(f"region {result.get('id')}, {result.get('from_name')}: expected exactly one text choice")
    return choices[0]


def video_basename(task: dict[str, Any]) -> str:
    raw = task.get("file_upload") or task.get("data", {}).get("video")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"task {task.get('id')}: missing video filename")
    name = Path(unquote(urlparse(raw).path)).name
    return re.sub(r"^[0-9a-fA-F]{8}-", "", name)


def parse_annotations(path: Path) -> tuple[list[dict], dict]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("annotation export must be a nonempty JSON task list")
    tracks, skipped = [], []
    counts = Counter()
    contexts = {field: Counter() for field in CONTEXT_FIELDS}
    raw_label_counts = {field: Counter() for field in FIELD_MAP}
    video_names = {}
    for task in tasks:
        basename = video_basename(task)
        video_id = Path(basename).stem
        if video_id in video_names:
            raise ValueError(f"duplicate task video ID: {video_id}")
        video_names[video_id] = basename
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled", False)]
        if len(annotations) != 1:
            raise ValueError(f"task {task.get('id')}: exactly one non-cancelled annotation is required, got {len(annotations)}")
        annotation = annotations[0]
        groups = defaultdict(dict)
        for result in annotation.get("result", []):
            field = result["from_name"]
            group = groups[result["id"]]
            if field in group:
                raise ValueError(f"task {task['id']}: duplicate region field {result['id']}/{field}")
            group[field] = result
        scene_tags = {}
        for group in groups.values():
            if "box" not in group:
                for field in ("scene", "time", "weather"):
                    if field in group:
                        value = single_choice(group[field])
                        if field in scene_tags and scene_tags[field] != value:
                            raise ValueError(f"task {task['id']}: conflicting {field} tags")
                        scene_tags[field] = value
                        contexts[field][value] += 1
        for region_id, group in groups.items():
            if "box" not in group:
                continue
            counts["box_tracks"] += 1
            box_result = group["box"]
            if box_result.get("type") != "videorectangle":
                raise ValueError(f"task {task['id']}, {region_id}: box is not videorectangle")
            value = box_result["value"]
            sequence = value.get("sequence", [])
            missing = [field for field in FIELD_MAP if field not in group]
            if not sequence or missing:
                skipped.append({"task_id": task["id"], "video_id": video_id,
                                "instance_id": region_id, "reason": "empty_bbox_sequence" if not sequence else "missing_hierarchical_labels",
                                "missing_fields": missing, "bbox_keyframes": len(sequence)})
                continue
            labels, source_labels = {}, {}
            for source, target in FIELD_MAP.items():
                source_labels[source] = single_choice(group[source])
                raw_label_counts[source][source_labels[source]] += 1
                labels[target] = ALIASES.get(target, {}).get(source_labels[source].casefold(), source_labels[source])
                if group[source]["value"].get("sequence") != sequence:
                    raise ValueError(f"task {task['id']}, {region_id}: {source} has a different temporal sequence from its box")
            encoded = encode_labels(labels).tolist()
            tags = dict(scene_tags)
            for field in ("age", "gender", "state"):
                if field in group:
                    tags[field] = single_choice(group[field])
                    contexts[field][tags[field]] += 1
            previous = -float("inf")
            rates = set()
            for point in sequence:
                timestamp = float(point["time"])
                coordinates = [float(point[key]) for key in ("x", "y", "width", "height")]
                if not math.isfinite(timestamp) or timestamp <= previous:
                    raise ValueError(f"task {task['id']}, {region_id}: non-increasing bbox timestamps")
                previous = timestamp
                if not all(math.isfinite(v) for v in coordinates) or min(coordinates[2:]) <= 0:
                    raise ValueError(f"task {task['id']}, {region_id}: malformed bbox")
                if point.get("rotation", 0) != 0:
                    raise ValueError("rotated boxes require an explicitly defined crop policy")
                if not point.get("enabled", True):
                    raise ValueError("disabled track segments require an explicitly defined visibility policy")
                if timestamp > 0:
                    rates.add(round(float(point["frame"]) / timestamp, 6))
            counts["usable_tracks"] += 1
            counts["bbox_keyframes_usable"] += len(sequence)
            counts["single_keyframe_tracks"] += len(sequence) == 1
            counts["endpoint_exceeds_declared_duration"] += sequence[-1]["time"] > value["duration"]
            tracks.append({
                "video_id": video_id, "video_name": basename, "instance_id": region_id,
                "task_id": task["id"], "annotation_id": annotation["id"],
                "duration": float(value["duration"]), "annotation_frames_count": int(value["framesCount"]),
                "sequence": sequence, "labels": dict(zip(LEVELS, encoded)),
                "source_labels": source_labels, "text": tags,
                "annotation_frame_time_rates": sorted(rates),
            })
    summary = {
        "annotation_source": str(path.resolve()),
        "annotation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tasks": len(tasks), "videos": len(video_names),
        **dict(counts), "skipped_tracks": skipped,
        "video_names": video_names,
        "raw_label_counts": {field: dict(counter) for field, counter in raw_label_counts.items()},
        "context_counts": {field: dict(counter) for field, counter in contexts.items()},
        "label_aliases": ALIASES,
        "paper_discrepancies": [
            "This annotation export has 353 complete labeled tracks; paper reports 413 human instances.",
            "Explicit bbox timestamps use time=frame/6 (6 FPS); paper describes annotation at 5 FPS.",
            "Export labels 'talking to someone' and 'goal-directed movement' are explicitly mapped to paper labels 'Talking' and 'Going somewhere'.",
        ],
    }
    return tracks, summary


def assign_splits(video_ids, *, split_file=None, provisional=False, seed=42, counts=(164, 47, 23)):
    video_ids = sorted(video_ids)
    if split_file is not None:
        payload = json.loads(Path(split_file).read_text(encoding="utf-8"))
        assignments = {}
        for split in ("train", "val", "test"):
            if not isinstance(payload.get(split), list):
                raise ValueError(f"split file needs a list for {split}")
            for video in payload[split]:
                video = str(video)
                if video in assignments:
                    raise ValueError(f"duplicate video {video!r} in split file")
                assignments[video] = split
        if set(assignments) != set(video_ids):
            raise ValueError(f"split IDs must match annotation video stems: missing={sorted(set(video_ids) - set(assignments))}, extra={sorted(set(assignments) - set(video_ids))}")
        supplied_metadata = payload.get("_metadata", payload.get("metadata", {}))
        sidecar = Path(split_file).with_suffix(".metadata.json")
        if not supplied_metadata and sidecar.is_file():
            supplied_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        return assignments, {"source": str(Path(split_file).resolve()),
                             "supplied_metadata": supplied_metadata,
                             "note": "Externally supplied assignment; official status is not asserted by this converter."}
    if not provisional:
        raise ValueError("supply --split-file or explicitly choose --provisional-split")
    if len(counts) != 3 or min(counts) < 0 or sum(counts) != len(video_ids):
        raise ValueError(f"--split-counts must be three nonnegative integers summing to {len(video_ids)} videos")
    random.Random(seed).shuffle(video_ids)
    assignments, offset = {}, 0
    for split, count in zip(("train", "val", "test"), counts):
        for video in video_ids[offset:offset + count]:
            assignments[video] = split
        offset += count
    return assignments, {"official": False, "source": "seeded provisional video split",
                         "seed": seed, "counts": list(counts),
                         "note": "Generated reproducible assignment, not original paper video IDs. Default counts follow the user-requested 7:2:1 ratio for 234 videos."}


def interpolate_box(sequence, timestamp, width, height):
    """Linear interpolation between enabled keyframes; hold endpoints.

    The final enabled keyframe persists to clip end, which matters for stationary
    tracks represented by only one keyframe. Pixel observations still end at the
    selected causal cutoff; interpolation only reconstructs supplied annotations.
    """
    times = [float(point["time"]) for point in sequence]
    right = bisect_right(times, timestamp)
    first, second = sequence[max(0, right - 1)], sequence[min(right, len(sequence) - 1)]
    denominator = float(second["time"]) - float(first["time"])
    weight = 0 if denominator <= 0 else min(1, max(0, (timestamp - float(first["time"])) / denominator))
    values = [float(first[key]) * (1 - weight) + float(second[key]) * weight for key in ("x", "y", "width", "height")]
    x, y, w, h = values
    box = [max(0.0, x * width / 100), max(0.0, y * height / 100),
           min(float(width), (x + w) * width / 100), min(float(height), (y + h) * height / 100)]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("interpolated bbox lies outside the video")
    return box


def probe_video(path, ffprobe, cache_path=None):
    path = Path(path)
    stat = path.stat()
    identity = {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache_path is not None and Path(cache_path).is_file():
        cached = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if cached.get("source") == identity:
            return cached["video_info"]
    result = subprocess.run([
        ffprobe, "-v", "error", "-threads", "1", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "stream=width,height,duration:frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    if any("best_effort_timestamp_time" not in frame for frame in payload["frames"]):
        raise ValueError(f"{path}: frame PTS missing; native frame index alignment cannot be guaranteed")
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]]
    if not timestamps or timestamps != sorted(timestamps):
        raise ValueError(f"{path}: primary video frame timestamps are missing or unordered")
    origin = timestamps[0]
    timestamps = [time - origin for time in timestamps]
    info = {"width": int(stream["width"]), "height": int(stream["height"]),
            "timestamps": timestamps, "pts_origin": origin,
            "duration": float(stream.get("duration", timestamps[-1]))}
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"source": identity, "video_info": info}) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
    return info


def observed_frame_indices(timestamps, requested):
    """Choose the most recent available frame without crossing each cutoff."""
    if not timestamps or timestamps != sorted(timestamps):
        raise ValueError("native frame timestamps must be nonempty and ordered")
    if any(not math.isfinite(time) or time < timestamps[0] - 1e-9 for time in requested):
        raise ValueError("requested observation precedes the first available frame")
    return [bisect_right(timestamps, time + 1e-9) - 1 for time in requested]


def extract_video_frames(video_path, output, indices, ffmpeg):
    output.mkdir(parents=True, exist_ok=True)
    pending = [index for index in sorted(set(indices)) if not (output / f"{index:08d}.jpg").is_file()]
    if not pending:
        return
    selection = "+".join(f"eq(n\\,{index})" for index in pending)
    with tempfile.TemporaryDirectory(prefix="decode-", dir=output) as temporary:
        result = subprocess.run([
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-threads", "1",
            "-i", str(video_path), "-map", "0:v:0", "-vf", f"select={selection}",
            "-vsync", "0", "-frames:v", str(len(pending)), "-q:v", "2", "-threads", "1", str(Path(temporary) / "%08d.jpg"),
        ], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {result.stderr[-2000:]}")
        decoded = sorted(Path(temporary).glob("*.jpg"))
        if len(decoded) != len(pending):
            raise ValueError(f"{video_path}: requested {len(pending)} frames but decoded {len(decoded)}")
        for index, source in zip(pending, decoded):
            source.replace(output / f"{index:08d}.jpg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--videos", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared"))
    splits = parser.add_mutually_exclusive_group()
    splits.add_argument("--split-file", type=Path)
    splits.add_argument("--provisional-split", action="store_true")
    parser.add_argument("--split-counts", nargs=3, type=int, default=(164, 47, 23), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--observation-ratio", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit-videos-per-split", type=int)
    parser.add_argument("--max-videos", type=int, help="bounded subset balanced across the supplied splits for a quick integration test")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.num_frames < 1 or not 0 < args.observation_ratio <= 1 or args.workers < 1:
        parser.error("num-frames/workers must be positive and observation-ratio in (0,1]")
    if args.workers > 16:
        parser.error("use at most sixteen video workers; each ffmpeg/probe decode uses one thread")
    if args.max_videos is not None and args.limit_videos_per_split is not None:
        parser.error("choose only one of --max-videos or --limit-videos-per-split")
    tracks, summary = parse_annotations(args.annotations)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        (output / "annotation_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in summary.items() if key not in ("video_names", "raw_label_counts", "context_counts", "skipped_tracks")}, ensure_ascii=False, indent=2))
        return
    if not args.videos or not args.ffmpeg or not args.ffprobe:
        parser.error("conversion requires --videos, --ffmpeg and --ffprobe (or binaries on PATH)")
    assignments, split_metadata = assign_splits(summary["video_names"], split_file=args.split_file,
                                                provisional=args.provisional_split, seed=args.seed, counts=args.split_counts)
    selected_videos = set(assignments)
    if args.limit_videos_per_split is not None:
        if args.limit_videos_per_split < 1:
            parser.error("limit-videos-per-split must be positive")
        selected_videos = {video for split in ("train", "val", "test") for video in sorted(v for v, s in assignments.items() if s == split)[:args.limit_videos_per_split]}
    elif args.max_videos is not None:
        if args.max_videos < 1:
            parser.error("max-videos must be positive")
        split_videos = [sorted(video for video, assigned in assignments.items() if assigned == split) for split in ("train", "val", "test")]
        interleaved = [videos[index] for index in range(max(map(len, split_videos))) for videos in split_videos if index < len(videos)]
        selected_videos = set(interleaved[:args.max_videos])
    tracks = [track for track in tracks if track["video_id"] in selected_videos]
    paths = {}
    for path in args.videos.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".mp4", ".mov", ".mkv"):
            name = path.name.casefold()
            if name in paths:
                raise ValueError(f"duplicate video basename: {path.name}")
            paths[name] = path.resolve()
    by_video = defaultdict(list)
    for track in tracks:
        by_video[track["video_id"]].append(track)

    def convert_video(item):
        video_id, video_tracks = item
        name = video_tracks[0]["video_name"].casefold()
        if name not in paths:
            raise FileNotFoundError(f"annotation video {name} was not found under {args.videos}")
        video_path = paths[name]
        info = probe_video(video_path, args.ffprobe, cache_path=output / "video_pts" / f"{video_id}.json")
        timestamps = info["timestamps"]
        records, selected_indices = [], set()
        for track in video_tracks:
            start = float(track["sequence"][0]["time"])
            # Final enabled box persists through declared clip duration. Actual
            # frame PTS bounds protect against rounded annotation endpoints.
            end = min(track["duration"], info["duration"], timestamps[-1])
            if start > end:
                raise ValueError(f"{video_id}/{track['instance_id']}: track starts beyond last video frame")
            cutoff = start + args.observation_ratio * (end - start)
            requested = [start + (cutoff - start) * i / max(1, args.num_frames - 1) for i in range(args.num_frames)]
            indices = observed_frame_indices(timestamps, requested)
            selected_indices.update(indices)
            actual = [timestamps[index] for index in indices]
            records.append({
                "video_id": video_id, "instance_id": track["instance_id"],
                "split": assignments[video_id],
                "frames": [(Path("frames") / video_id / f"{index:08d}.jpg").as_posix() for index in indices],
                "bboxes": [interpolate_box(track["sequence"], time, info["width"], info["height"]) for time in actual],
                "text": track["text"], "labels": track["labels"],
                "source": {
                    "task_id": track["task_id"], "annotation_id": track["annotation_id"],
                    "region_id": track["instance_id"], "video": str(video_path),
                    "source_labels": track["source_labels"], "native_frame_indices": indices,
                    "requested_timestamps": requested, "actual_timestamps": actual,
                    "observation_cutoff": cutoff, "observation_ratio": args.observation_ratio,
                    "track_start": start, "track_end": end,
                    "annotation_frames_count": track["annotation_frames_count"],
                    "annotation_frame_time_rates": track["annotation_frame_time_rates"],
                    "annotation_keyframe_timestamps": [float(point["time"]) for point in track["sequence"]],
                    "annotation_duration": track["duration"],
                    "video_width": info["width"], "video_height": info["height"],
                },
            })
        extract_video_frames(video_path, output / "frames" / video_id, selected_indices, args.ffmpeg)
        return records, len(selected_indices)

    records, frame_count = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for converted, count in pool.map(convert_video, sorted(by_video.items())):
            records.extend(converted)
            frame_count += count
            print(json.dumps({"converted_video": converted[0]["video_id"], "instances": len(converted), "frames": count}), flush=True)
    records.sort(key=lambda record: (record["video_id"], record["instance_id"]))
    (output / "manifest.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    empirical = sorted({tuple(record["labels"][level] for level in LEVELS) for record in records if record["split"] == "train"})
    (output / "empirical_train_triplets.json").write_text(json.dumps({
        "valid_triplets": empirical,
        "metadata": {"source": "Only observed training-record label triplets from the converted manifest",
                     "official": False, "complete_logical_graph": False,
                     "warning": "Empirical training support is not the official semantic compatibility graph. Unseen combinations need not be logically invalid.",
                     "train_records": sum(record["split"] == "train" for record in records)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    split_payload = {split: sorted(video for video, assigned in assignments.items() if assigned == split) for split in ("train", "val", "test")}
    split_payload["_metadata"] = split_metadata
    saved_split_path = output / "splits.json"
    if args.split_file is None or args.split_file.resolve() != saved_split_path:
        saved_split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.update({
        "split": split_metadata,
        "observation_protocol": {"official": False, "kind": "causal prefix", "ratio": args.observation_ratio,
                                 "num_frames": args.num_frames, "sampling": "uniform requested times; latest primary-stream frame PTS not exceeding each time",
                                 "bbox": "linear interpolation between enabled percentage boxes; last enabled box held through clip end",
                                 "note": "Paper does not specify observation/prediction horizon. No future video pixels beyond cutoff are selected."},
        "prepared_instances": len(records), "prepared_unique_frames": frame_count,
        "prepared_videos": len(by_video), "limit_videos_per_split": args.limit_videos_per_split,
        "max_videos": args.max_videos,
        "prepared_split_instances": dict(Counter(record["split"] for record in records)),
        "prepared_split_videos": {split: len({record["video_id"] for record in records if record["split"] == split}) for split in ("train", "val", "test")},
        "manifest": str(output / "manifest.jsonl"),
    })
    (output / "annotation_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("manifest", "prepared_instances", "prepared_unique_frames", "prepared_videos", "prepared_split_instances", "prepared_split_videos", "split", "observation_protocol")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
