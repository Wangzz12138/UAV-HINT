#!/usr/bin/env python3
"""Generate user-requested 7:2:1 video splits and a complete reviewable CSV."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re


def video_id(value):
    match = re.search(r"(?:^|[-/])(\d+)\.mp4$", str(value), flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify video: {value}")
    return f"{int(match.group(1)):03d}"


def allocate_counts(total, ratios):
    raw = [total * ratio / sum(ratios) for ratio in ratios]
    counts = [int(number) for number in raw]
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def ensure_class_coverage(splits, statistics, shuffled):
    """Deterministic video swaps keep sizes fixed and cover all observed labels.

    This is split stratification before fitting any model. It does not depend on
    predictions, features, validation scores, or a logical compatibility graph.
    """
    classes = {identifier: {(level, value) for labels in row["labels"]
                            for level, value in enumerate(labels)}
               for identifier, row in statistics.items()}
    universe = set().union(*classes.values())
    rank = {identifier: index for index, identifier in enumerate(shuffled)}

    def union(identifiers):
        return set().union(*(classes[identifier] for identifier in identifiers))

    swaps = []
    for target in splits:
        for missing in sorted(universe - union(splits[target])):
            if missing in union(splits[target]):
                continue
            found = None
            for source in splits:
                if source == target:
                    continue
                before_source, before_target = union(splits[source]), union(splits[target])
                for incoming in sorted(splits[source], key=rank.get):
                    if missing not in classes[incoming]:
                        continue
                    source_rest = union([i for i in splits[source] if i != incoming])
                    for outgoing in sorted(splits[target], key=rank.get):
                        after_source = source_rest | classes[outgoing]
                        after_target = union([i for i in splits[target] if i != outgoing]) | classes[incoming]
                        if before_source <= after_source and before_target <= after_target:
                            found = (source, incoming, outgoing)
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"Cannot cover label {missing} in {target} while preserving all split sizes")
            source, incoming, outgoing = found
            splits[source].remove(incoming)
            splits[source].append(outgoing)
            splits[target].remove(outgoing)
            splits[target].append(incoming)
            swaps.append({"target": target, "source": source,
                          "incoming_video": incoming, "outgoing_video": outgoing,
                          "reason": {"level_index": missing[0], "label": missing[1]}})
    for name in splits:
        splits[name].sort()
        if union(splits[name]) != universe:
            raise ValueError(f"Class coverage is incomplete in {name}")
    return swaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("datasets/videos"))
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=int, default=[7, 2, 1], metavar=("TRAIN", "VAL", "TEST"))
    args = parser.parse_args()
    if any(number <= 0 for number in args.ratios):
        parser.error("ratios must be positive")
    paths = sorted((path for path in args.videos.iterdir() if path.suffix.lower() == ".mp4"), key=lambda p: int(p.stem))
    files = {video_id(path.name): path.name for path in paths}
    if len(files) != len(paths) or not files:
        raise ValueError("Video IDs must be unique and nonempty")
    tasks = json.loads(args.annotations.read_text(encoding="utf-8"))
    statistics = {}
    fields = ("current_action", "short_term_intention", "high_level_intention")
    for task in tasks:
        identifier = video_id(task["data"]["video"])
        if identifier not in files or identifier in statistics:
            raise ValueError(f"Missing/duplicate video in annotations: {identifier}")
        annotations = [row for row in task.get("annotations", []) if not row.get("was_cancelled", False)]
        if len(annotations) != 1:
            raise ValueError(f"Expected one completed annotation for {identifier}")
        groups = defaultdict(dict)
        for result in annotations[0].get("result", []):
            groups[result["id"]][result["from_name"]] = result
        complete, skipped, labels = 0, 0, []
        for group in groups.values():
            boxes = group.get("box")
            if boxes is None:
                continue
            names = [group.get(field, {}).get("value", {}).get("choices", []) for field in fields]
            if boxes.get("value", {}).get("sequence") and all(len(value) == 1 for value in names):
                complete += 1
                labels.append([value[0] for value in names])
            else:
                skipped += 1
        statistics[identifier] = {"complete_labeled_tracks": complete, "skipped_tracks": skipped, "labels": labels}
    if set(statistics) != set(files):
        raise ValueError("Every video must have an annotation task, including empty tasks")
    shuffled = sorted(files)
    random.Random(args.seed).shuffle(shuffled)
    counts = allocate_counts(len(shuffled), args.ratios)
    splits, offset = {}, 0
    for name, count in zip(("train", "val", "test"), counts):
        splits[name] = sorted(shuffled[offset:offset + count])
        offset += count
    swaps = ensure_class_coverage(splits, statistics, shuffled)
    assignment = {identifier: name for name, identifiers in splits.items() for identifier in identifiers}
    assert len(assignment) == len(files) == sum(counts)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    with (output / "splits.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "file_name", "split", "complete_labeled_tracks", "skipped_tracks", "status"])
        writer.writeheader()
        for identifier in sorted(files):
            row = statistics[identifier]
            writer.writerow({"video_id": identifier, "file_name": files[identifier],
                             "split": assignment[identifier], "complete_labeled_tracks": row["complete_labeled_tracks"],
                             "skipped_tracks": row["skipped_tracks"],
                             "status": "usable" if row["complete_labeled_tracks"] else "no_complete_labels"})
    summary = {}
    for name, identifiers in splits.items():
        records = [values for identifier in identifiers for values in statistics[identifier]["labels"]]
        summary[name] = {
            "videos": len(identifiers),
            "videos_with_complete_labels": sum(statistics[i]["complete_labeled_tracks"] > 0 for i in identifiers),
            "complete_labeled_tracks": len(records),
            "class_counts": {field: dict(Counter(values[index] for values in records)) for index, field in enumerate(fields)},
        }
    metadata = {"source": "New split generated at user request; user specified train:val:test=7:2:1",
                "original_official_video_ids": False, "seed": args.seed,
                "ratio": args.ratios, "rounding": "largest remainder", "grouping": "video_id",
                "stratification": "Deterministic video swaps to cover all observed labels in every split",
                "coverage_swaps": swaps,
                "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                "all_videos": len(files), "excluded_video_ids_without_complete_labels": [i for i in sorted(files) if not statistics[i]["complete_labeled_tracks"]],
                "summary": summary}
    (output / "splits.metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    markdown = ["# UAV-HINT 视频划分表", "", f"按用户指定 train/val/test=7:2:1 新生成，seed={args.seed}；不是恢复原始官方视频 ID。", "", "固定视频数后，以确定的视频交换补齐各集合的类别覆盖；三层 7/8/4 类均有正例。同一视频的人物不会跨集合。", "", "| 划分 | 视频数 | 有完整标签的视频 | 可用人物轨迹 |", "| --- | ---: | ---: | ---: |"]
    for name, values in summary.items():
        markdown.append(f"| {name} | {values['videos']} | {values['videos_with_complete_labels']} | {values['complete_labeled_tracks']} |")
    markdown.extend(["", "完整逐视频表见 [splits.csv](splits.csv)，程序输入见 [splits.json](splits.json)。", "", "011 没有完整的已提交人物标注，仍保留视频划分归属，但不作为有监督训练/评估样本。草稿未合并。", ""])
    (output / "split_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "counts": counts, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
