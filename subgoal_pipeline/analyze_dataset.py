import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import load_json, write_json


def analyze(args: argparse.Namespace) -> None:
    dataset_path = Path(args.dataset_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(dataset_path)
    episodes = data.get("episodes") or []
    if not isinstance(episodes, list):
        raise RuntimeError(f"{dataset_path} does not contain an episodes list.")

    by_trajectory: Dict[int, dict] = {}
    duplicate_mismatches: List[dict] = []
    for episode in episodes:
        trajectory_id = int(episode.get("trajectory_id", episode["episode_id"]))
        goals = episode.get("goals") or []
        positions = [[float(v) for v in goal["position"]] for goal in goals]
        if trajectory_id not in by_trajectory:
            by_trajectory[trajectory_id] = episode
            continue
        prev = [[float(v) for v in goal["position"]] for goal in by_trajectory[trajectory_id].get("goals", [])]
        if positions != prev:
            duplicate_mismatches.append(
                {
                    "trajectory_id": trajectory_id,
                    "episode_id": int(episode["episode_id"]),
                    "first_episode_id": int(by_trajectory[trajectory_id]["episode_id"]),
                }
            )

    distance_rows = []
    start_rows = []
    subgoal_counts = Counter()
    for trajectory_id, episode in sorted(by_trajectory.items()):
        goals = episode.get("goals") or []
        subgoal_counts[len(goals)] += 1
        if goals:
            start = [float(v) for v in episode["start_position"]]
            first = [float(v) for v in goals[0]["position"]]
            start_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "representative_episode_id_in_dataset": int(episode["episode_id"]),
                    "subgoal_count": len(goals),
                    "start_to_first_distance_3d": math.dist(start, first),
                    "start_to_first_distance_xz": math.dist([start[0], start[2]], [first[0], first[2]]),
                    "start_position": json.dumps(start),
                    "first_subgoal_position": json.dumps(first),
                }
            )
        for idx, (a, b) in enumerate(zip(goals[:-1], goals[1:])):
            pos_a = [float(v) for v in a["position"]]
            pos_b = [float(v) for v in b["position"]]
            distance_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "representative_episode_id_in_dataset": int(episode["episode_id"]),
                    "pair_index": idx,
                    "from_subgoal": idx,
                    "to_subgoal": idx + 1,
                    "distance_3d": math.dist(pos_a, pos_b),
                    "distance_xz": math.dist([pos_a[0], pos_a[2]], [pos_b[0], pos_b[2]]),
                    "from_position": json.dumps(pos_a),
                    "to_position": json.dumps(pos_b),
                }
            )

    write_csv(out_dir / "subgoal_distances.csv", distance_rows)
    write_csv(out_dir / "start_to_first_subgoal_distances.csv", start_rows)

    summary = {
        "input_path": str(dataset_path),
        "episode_count": len(episodes),
        "trajectory_count": len(by_trajectory),
        "duplicate_trajectory_goal_mismatch_count": len(duplicate_mismatches),
        "distance_pair_count": len(distance_rows),
        "subgoal_count_distribution_per_trajectory": dict(sorted(subgoal_counts.items())),
        "distance_3d": describe([row["distance_3d"] for row in distance_rows]),
        "distance_xz": describe([row["distance_xz"] for row in distance_rows]),
        "start_to_first_3d": describe([row["start_to_first_distance_3d"] for row in start_rows]),
        "start_to_first_xz": describe([row["start_to_first_distance_xz"] for row in start_rows]),
        "start_to_first_le1_count": sum(row["start_to_first_distance_3d"] <= 1.0 + 1e-9 for row in start_rows),
        "start_to_first_gt6_count": sum(row["start_to_first_distance_3d"] > 6.0 + 1e-9 for row in start_rows),
        "adjacent_le1_count": sum(row["distance_3d"] <= 1.0 + 1e-9 for row in distance_rows),
        "adjacent_gt6_count": sum(row["distance_3d"] > 6.0 + 1e-9 for row in distance_rows),
        "duplicate_trajectory_goal_mismatches": duplicate_mismatches[:50],
    }
    write_json(out_dir / "subgoal_distance_summary.json", summary, pretty=True)

    plot_histogram(
        [row["distance_3d"] for row in distance_rows],
        out_dir / "subgoal_distance_histogram.png",
        title="Adjacent Sub-goal Distance",
        xlabel="Distance (m, 3D)",
        bin_width=float(args.bin_width),
    )
    plot_histogram(
        [row["start_to_first_distance_3d"] for row in start_rows],
        out_dir / "start_to_first_subgoal_distance_histogram.png",
        title="Start to First Sub-goal Distance",
        xlabel="Distance (m, 3D)",
        bin_width=float(args.bin_width),
    )
    plot_count_histogram(subgoal_counts, out_dir / "subgoal_count_histogram.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved analysis to {out_dir}")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def describe(values: List[float]) -> dict:
    if not values:
        return {}
    values = sorted(float(v) for v in values)
    return {
        "min": values[0],
        "max": values[-1],
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
    }


def percentile(sorted_values: List[float], pct: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_values[lo]
    weight = rank - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def plot_histogram(values: List[float], path: Path, title: str, xlabel: str, bin_width: float) -> None:
    if not values:
        return
    min_v, max_v = min(values), max(values)
    width = max(float(bin_width), 1e-6)
    bins = max(1, int(math.ceil((max_v - min_v) / width)))
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Trajectory pair count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_count_histogram(counts: Counter, path: Path) -> None:
    if not counts:
        return
    xs = sorted(counts)
    ys = [counts[x] for x in xs]
    plt.figure(figsize=(7, 4))
    plt.bar([str(x) for x in xs], ys)
    plt.title("Sub-goal Count per Trajectory")
    plt.xlabel("Sub-goal count")
    plt.ylabel("Trajectory count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze R2R sub-goal distance distributions.")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="VLN-CE/datasets/rxr/train/train_guide_subgoals_reachable.json.gz",
    )
    parser.add_argument("--out_dir", type=str, default="results/r2r_subgoal_analysis")
    parser.add_argument("--bin_width", type=float, default=0.5)
    return parser


def main() -> None:
    analyze(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
