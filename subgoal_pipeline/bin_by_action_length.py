"""Bin a reachable dataset by GT action length.

Reads GT action lengths from a ``*_gt_reachable.json.gz`` mapping
(``{episode_id: {actions: [...]}}``) and splits a matching
``*_reachable.json.gz`` habitat episode file into per-bin shards of a
fixed step interval. Bins are half-open ``[lo, hi)``; an episode with
action length ``L`` falls into bin ``lo = (L // bin_size) * bin_size``.
"""

import argparse
from pathlib import Path

from .common import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-path",
        type=Path,
        help="Path to *_gt_reachable.json.gz (episode_id -> {actions: [...]}).",
        default="VLN-CE/datasets/rxr/train/train_guide_gt_reachable_english.json.gz",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to *_reachable.json.gz ({episodes: [...]}).",
        default="VLN-CE/datasets/rxr/train/train_guide_reachable_english.json.gz",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to write the binned *_reachable_<lo>-<hi>.json.gz files.",
        default="VLN-CE/datasets/rxr/train/train_split",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=50,
        help="Step interval per bin (default: 50).",
    )
    return parser.parse_args()


def bin_by_action_length(args: argparse.Namespace) -> None:
    if args.bin_size <= 0:
        raise ValueError(f"--bin-size must be positive, got {args.bin_size}.")

    gt = load_json(args.gt_path)
    length_map = {
        str(eid): len(entry["actions"])
        for eid, entry in gt.items()
        if isinstance(entry, dict) and "actions" in entry
    }
    if not length_map:
        raise RuntimeError(f"No GT actions found in {args.gt_path}.")

    data = load_json(args.dataset_path)
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError(f"{args.dataset_path} has no 'episodes' list.")

    stem = args.dataset_path.name
    for suffix in (".json.gz", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    bins: dict[int, list] = {}
    missing = 0
    for ep in episodes:
        eid = str(ep["episode_id"])
        length = length_map.get(eid)
        if length is None:
            missing += 1
            continue
        lo = (length // args.bin_size) * args.bin_size
        bins.setdefault(lo, []).append(ep)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for lo in sorted(bins):
        hi = lo + args.bin_size
        eps = bins[lo]
        total += len(eps)
        out_path = out_dir / f"{stem}_{lo}-{hi}.json.gz"
        write_json(out_path, {"episodes": eps}, pretty=False)
        print(f"{out_path.name}: [{lo},{hi}) {len(eps)} episodes")

    print("-" * 40)
    print(
        f"bins={len(bins)} binned={total} missing_gt={missing} "
        f"bin_size={args.bin_size}"
    )
    assert total + missing == len(episodes), "bin count invariant violated"


def main() -> None:
    bin_by_action_length(parse_args())


if __name__ == "__main__":
    main()
