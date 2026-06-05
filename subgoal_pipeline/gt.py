import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .common import instruction_text, to_float_list


@dataclass
class GroundTruthTrajectory:
    episode_id: int
    actions: List[int]
    locations: List[List[float]]
    forward_steps: int


def load_ground_truth_trajectories(gt_json_path: Path) -> Dict[int, GroundTruthTrajectory]:
    if gt_json_path.suffix != ".gz":
        raise RuntimeError(f"{gt_json_path} must be a gzipped train_gt JSON file ending with .gz.")
    with gzip.open(gt_json_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"{gt_json_path} must be a dict keyed by episode_id.")

    trajectories: Dict[int, GroundTruthTrajectory] = {}
    for raw_episode_id, raw_traj in data.items():
        episode_id = int(raw_episode_id)
        if not isinstance(raw_traj, dict):
            raise RuntimeError(f"{gt_json_path}: episode {episode_id} GT entry is not an object.")

        actions = [int(a) for a in (raw_traj.get("actions") or [])]
        locations = [to_float_list(loc) for loc in (raw_traj.get("locations") or [])]
        forward_steps = int(raw_traj.get("forward_steps", sum(1 for a in actions if a == 1)))
        actual_forward_steps = sum(1 for a in actions if a == 1)
        if actual_forward_steps != forward_steps:
            raise RuntimeError(
                f"{gt_json_path}: episode {episode_id} forward_steps={forward_steps}, "
                f"but actions contain {actual_forward_steps} MOVE_FORWARD steps."
            )
        if locations and len(locations) != forward_steps + 1:
            raise RuntimeError(
                f"{gt_json_path}: episode {episode_id} has {len(locations)} locations, "
                f"expected forward_steps + 1 = {forward_steps + 1}."
            )

        trajectories[episode_id] = GroundTruthTrajectory(
            episode_id=episode_id,
            actions=actions,
            locations=locations,
            forward_steps=forward_steps,
        )
    return trajectories


def build_dataset_indices(train_data: dict) -> Tuple[Dict[int, dict], Dict[int, List[int]]]:
    episodes = train_data.get("episodes")
    if not isinstance(episodes, list):
        raise RuntimeError("train_json does not contain an episodes list.")

    by_id: Dict[int, dict] = {}
    by_trajectory: Dict[int, List[int]] = {}
    for episode in episodes:
        episode_id = int(episode["episode_id"])
        trajectory_id = int(episode.get("trajectory_id", episode_id))
        by_id[episode_id] = episode
        by_trajectory.setdefault(trajectory_id, []).append(episode_id)
    for episode_ids in by_trajectory.values():
        episode_ids.sort()
    return by_id, by_trajectory


def select_trajectory_groups(
    train_data: dict,
    gt_trajectories: Dict[int, GroundTruthTrajectory],
    max_gt_actions: int,
    target_episodes: int,
) -> Tuple[List[Tuple[int, List[int], int]], Dict[str, int]]:
    by_id, by_trajectory = build_dataset_indices(train_data)
    eligible: List[Tuple[int, List[int], int]] = []
    skipped_missing_gt = 0
    skipped_too_long = 0
    skipped_partial_group = 0

    for trajectory_id in sorted(by_trajectory):
        episode_ids = by_trajectory[trajectory_id]
        usable_ids: List[int] = []
        group_missing = False
        group_too_long = False
        for episode_id in episode_ids:
            gt = gt_trajectories.get(int(episode_id))
            if gt is None:
                group_missing = True
                continue
            if int(max_gt_actions) >= 0 and len(gt.actions) > int(max_gt_actions):
                group_too_long = True
                continue
            usable_ids.append(int(episode_id))

        if group_missing:
            skipped_missing_gt += 1
        if group_too_long:
            skipped_too_long += 1
        if len(usable_ids) != len(episode_ids):
            skipped_partial_group += 1
            continue
        if not usable_ids:
            continue

        representative_id = max(
            usable_ids,
            key=lambda eid: (len(instruction_text(by_id[eid])), -int(eid)),
        )
        eligible.append((int(trajectory_id), usable_ids, int(representative_id)))

    selected, selection_mode = _choose_groups_for_target(eligible, int(target_episodes))
    selected_episode_count = sum(len(episode_ids) for _tid, episode_ids, _rep in selected)
    stats = {
        "selection_mode": selection_mode,
        "eligible_trajectories": len(eligible),
        "eligible_episodes": sum(len(episode_ids) for _tid, episode_ids, _rep in eligible),
        "selected_trajectories": len(selected),
        "selected_episodes": selected_episode_count,
        "skipped_groups_with_missing_gt": skipped_missing_gt,
        "skipped_groups_with_too_long_episode": skipped_too_long,
        "skipped_partial_groups": skipped_partial_group,
    }
    return selected, stats


def _choose_groups_for_target(
    eligible: List[Tuple[int, List[int], int]],
    target_episodes: int,
) -> Tuple[List[Tuple[int, List[int], int]], str]:
    if int(target_episodes) <= 0:
        return list(eligible), "all_eligible_trajectories"

    target = int(target_episodes)
    reachable: Dict[int, List[int]] = {0: []}
    for idx, (_trajectory_id, episode_ids, _representative_id) in enumerate(eligible):
        size = len(episode_ids)
        for count, chosen in list(reachable.items()):
            new_count = count + size
            if new_count <= target and new_count not in reachable:
                reachable[new_count] = chosen + [idx]
        if target in reachable:
            break

    if target in reachable:
        return [eligible[idx] for idx in reachable[target]], "exact_target_episode_count"

    selected = []
    selected_episode_count = 0
    for group in eligible:
        selected.append(group)
        selected_episode_count += len(group[1])
        if selected_episode_count >= target:
            break
    return selected, "prefix_at_or_above_target_episode_count"
