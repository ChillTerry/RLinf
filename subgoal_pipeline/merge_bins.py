"""Merge selected binned ``*_reachable_<lo>-<hi>.json.gz`` shards.

Loads each input shard (``{episodes: [...]}``), concatenates their episode
lists in the given order, and writes a single ``{"episodes": [...]}`` file.
Useful for recombining a subset of the bins produced by
``bin_by_action_length`` (e.g. only the 50-100 and 100-150 step ranges).
"""

import argparse

from .common import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input shard paths (*_reachable_<lo>-<hi>.json.gz), in merge order.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .json.gz path for the merged dataset.",
    )
    return parser.parse_args()


def merge_bins(args: argparse.Namespace) -> None:
    from pathlib import Path

    inputs = [Path(p) for p in args.inputs]
    output = Path(args.output)

    merged: list = []
    for p in inputs:
        data = load_json(p)
        episodes = data.get("episodes")
        if not isinstance(episodes, list):
            raise RuntimeError(f"{p} has no 'episodes' list.")
        merged.extend(episodes)
        print(f"{p.name}: {len(episodes)} episodes")

    write_json(output, {"episodes": merged}, pretty=False)
    print("-" * 40)
    print(f"{output.name}: {len(merged)} episodes (merged {len(inputs)} files)")


def main() -> None:
    merge_bins(parse_args())


if __name__ == "__main__":
    main()
