#!/usr/bin/env python3
"""Compare two Habitat metrics directories by episode id."""

import argparse
import json
from pathlib import Path


def _episode_id(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _load_metrics(directory: Path) -> dict[int, dict]:
    files = sorted(directory.glob("episode_*.json"), key=_episode_id)
    metrics = {}
    for path in files:
        with path.open(encoding="utf-8") as file_obj:
            metrics[_episode_id(path)] = json.load(file_obj)
    return metrics


def _numeric(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _summarize(metrics: dict[int, dict]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in metrics.values():
        for key, value in row.items():
            numeric = _numeric(value)
            if numeric is None:
                continue
            sums[key] = sums.get(key, 0.0) + numeric
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sorted(sums)}


def compare(left_dir: Path, right_dir: Path) -> dict:
    left = _load_metrics(left_dir)
    right = _load_metrics(right_dir)
    common_ids = sorted(set(left) & set(right))
    all_keys = sorted(
        {
            key
            for episode_id in common_ids
            for key in set(left[episode_id]) | set(right[episode_id])
        }
    )

    paired = {}
    for key in all_keys:
        left_values = []
        right_values = []
        diffs = []
        changed = 0
        for episode_id in common_ids:
            left_value = _numeric(left[episode_id].get(key))
            right_value = _numeric(right[episode_id].get(key))
            if left_value is None or right_value is None:
                continue
            left_values.append(left_value)
            right_values.append(right_value)
            diff = left_value - right_value
            diffs.append(diff)
            if abs(diff) > 1e-9:
                changed += 1
        if left_values:
            paired[key] = {
                "left_mean": _mean(left_values),
                "right_mean": _mean(right_values),
                "mean_diff": _mean(diffs),
                "changed": changed,
                "paired_count": len(left_values),
            }

    success_flips = {
        "left_success_right_fail": 0,
        "left_fail_right_success": 0,
        "same_success": 0,
        "same_fail": 0,
    }
    for episode_id in common_ids:
        left_success = int(float(left[episode_id].get("success", 0.0)))
        right_success = int(float(right[episode_id].get("success", 0.0)))
        if left_success == 1 and right_success == 0:
            success_flips["left_success_right_fail"] += 1
        elif left_success == 0 and right_success == 1:
            success_flips["left_fail_right_success"] += 1
        elif left_success == 1 and right_success == 1:
            success_flips["same_success"] += 1
        else:
            success_flips["same_fail"] += 1

    return {
        "left_dir": str(left_dir),
        "right_dir": str(right_dir),
        "left_count": len(left),
        "right_count": len(right),
        "common_count": len(common_ids),
        "left_only_count": len(set(left) - set(right)),
        "right_only_count": len(set(right) - set(left)),
        "left_summary": _summarize(left),
        "right_summary": _summarize(right),
        "paired": paired,
        "success_flips": success_flips,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    result = compare(args.left.resolve(), args.right.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
