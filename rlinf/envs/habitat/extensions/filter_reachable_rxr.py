# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
JSON_GZ_SUFFIX = ".json.gz"
REACHABLE_SUFFIX = "_reachable"
GT_SUFFIX = "_gt"


def load_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def dump_json_gz(path: Path, data: dict[str, Any]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file_obj:
        json.dump(data, file_obj)


def strip_json_gz_suffix(path: Path) -> str:
    if not path.name.endswith(JSON_GZ_SUFFIX):
        raise ValueError(f"Expected a {JSON_GZ_SUFFIX} file: {path}")
    return path.name[: -len(JSON_GZ_SUFFIX)]


def with_reachable_suffix(path: Path) -> Path:
    stem = strip_json_gz_suffix(path)
    if stem.endswith(REACHABLE_SUFFIX):
        return path
    return path.with_name(f"{stem}{REACHABLE_SUFFIX}{JSON_GZ_SUFFIX}")


def get_gt_path(main_path: Path) -> Path:
    stem = strip_json_gz_suffix(main_path)
    return main_path.with_name(f"{stem}{GT_SUFFIX}{JSON_GZ_SUFFIX}")


def is_main_dataset_file(path: Path) -> bool:
    if not path.name.endswith(JSON_GZ_SUFFIX):
        return False
    stem = strip_json_gz_suffix(path)
    return not stem.endswith(GT_SUFFIX) and not stem.endswith(REACHABLE_SUFFIX)


def discover_main_dataset_files(dataset_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in dataset_dir.glob(f"*{JSON_GZ_SUFFIX}")
        if is_main_dataset_file(path)
    )


def get_habitat_navmesh_path(scene_id: str, scenes_dir: Path) -> Path:
    direct_candidate = Path(os.path.splitext(scene_id)[0] + ".navmesh")
    if direct_candidate.exists():
        return direct_candidate

    scene_relative_path = Path(scene_id)
    if str(scene_relative_path).startswith(str(scenes_dir)):
        scene_relative_path = Path(os.path.relpath(scene_relative_path, scenes_dir))
    return scenes_dir / scene_relative_path.with_suffix(".navmesh")


def episode_is_reachable(pathfinder, episode: dict[str, Any]) -> bool:
    import habitat_sim

    start_position = episode["start_position"]
    for goal in episode["goals"]:
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = start_position
        shortest_path.requested_end = goal["position"]
        found_path = pathfinder.find_path(shortest_path)
        distance = float(shortest_path.geodesic_distance)
        if not found_path or not np.isfinite(distance):
            return False
    return True


def filter_reachable_episodes(
    episodes: list[dict[str, Any]],
    *,
    scenes_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    import habitat_sim

    episodes_by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episodes_by_scene[episode["scene_id"]].append(episode)

    kept_episodes: list[dict[str, Any]] = []
    dropped_episodes: list[dict[str, Any]] = []
    dropped_by_scene: Counter[str] = Counter()
    for scene_id, scene_episodes in sorted(episodes_by_scene.items()):
        navmesh_path = get_habitat_navmesh_path(scene_id, scenes_dir)
        pathfinder = habitat_sim.PathFinder()
        if not pathfinder.load_nav_mesh(str(navmesh_path)):
            dropped_episodes.extend(scene_episodes)
            dropped_by_scene[scene_id] += len(scene_episodes)
            logger.warning(
                "Dropped all %s episodes for scene %s because navmesh failed to load: %s",
                len(scene_episodes),
                scene_id,
                navmesh_path,
            )
            continue

        for episode in scene_episodes:
            if episode_is_reachable(pathfinder, episode):
                kept_episodes.append(episode)
            else:
                dropped_episodes.append(episode)
                dropped_by_scene[scene_id] += 1

    return kept_episodes, dropped_episodes, dropped_by_scene


def build_reachable_main_data(
    main_data: dict[str, Any],
    kept_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    reachable_data = dict(main_data)
    reachable_data["episodes"] = kept_episodes
    return reachable_data


def build_reachable_gt_data(
    gt_data: dict[str, Any],
    kept_episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    kept_episode_ids = {str(episode["episode_id"]) for episode in kept_episodes}
    missing_gt_ids = sorted(kept_episode_ids.difference(gt_data))
    if missing_gt_ids:
        preview = ", ".join(missing_gt_ids[:10])
        raise KeyError(f"Missing GT for kept episode_id(s): {preview}")

    reachable_gt_data = {
        episode_id: gt_data[episode_id]
        for episode_id in gt_data
        if episode_id in kept_episode_ids
    }
    if set(reachable_gt_data) != kept_episode_ids:
        raise ValueError("Reachable GT episode IDs do not match kept episode IDs.")
    return reachable_gt_data


def process_dataset_file(main_path: Path, *, scenes_dir: Path) -> None:
    main_data = load_json_gz(main_path)
    episodes = main_data["episodes"]
    kept_episodes, dropped_episodes, dropped_by_scene = filter_reachable_episodes(
        episodes,
        scenes_dir=scenes_dir,
    )

    output_main_path = with_reachable_suffix(main_path)
    dump_json_gz(output_main_path, build_reachable_main_data(main_data, kept_episodes))

    gt_path = get_gt_path(main_path)
    output_gt_path = None
    if gt_path.exists():
        output_gt_path = with_reachable_suffix(gt_path)
        gt_data = load_json_gz(gt_path)
        reachable_gt_data = build_reachable_gt_data(gt_data, kept_episodes)
        dump_json_gz(output_gt_path, reachable_gt_data)
    else:
        logger.warning("GT file not found for %s: %s", main_path, gt_path)

    logger.info(
        "Processed %s: input=%s kept=%s dropped=%s output=%s gt_output=%s",
        main_path,
        len(episodes),
        len(kept_episodes),
        len(dropped_episodes),
        output_main_path,
        output_gt_path,
    )
    for scene_id, count in sorted(dropped_by_scene.items()):
        logger.info("Dropped %s unreachable episodes from scene %s", count, scene_id)
    if dropped_episodes:
        logger.info(
            "First dropped episode ids for %s: %s",
            main_path.name,
            [episode["episode_id"] for episode in dropped_episodes[:50]],
        )


def process_dataset_dir(dataset_dir: Path, *, scenes_dir: Path) -> None:
    main_paths = discover_main_dataset_files(dataset_dir)
    if not main_paths:
        raise ValueError(f"No RxR main dataset files found in {dataset_dir}")

    for main_path in main_paths:
        process_dataset_file(main_path, scenes_dir=scenes_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create *_reachable RxR dataset files by dropping episodes whose "
            "start-to-goal geodesic path is unreachable on the Habitat navmesh."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Directory containing RxR *.json.gz files to process.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path("VLN-CE/scene_dataset"),
        help="Habitat scenes directory containing MP3D navmesh files.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    process_dataset_dir(args.dataset_dir, scenes_dir=args.scenes_dir)


if __name__ == "__main__":
    main()
