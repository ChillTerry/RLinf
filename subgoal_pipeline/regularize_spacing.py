import argparse
import math
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .common import load_json, write_json


def backup_once(path: Path, suffix: str) -> Path:
    backup_path = path.with_suffix(path.suffix + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def original_goal_by_episode_id(original_train_json: Path) -> Dict[int, List[float]]:
    if original_train_json.suffix != ".gz":
        raise RuntimeError(f"{original_train_json} must be a gzipped train JSON file ending with .gz.")
    data = load_json(original_train_json)
    result: Dict[int, List[float]] = {}
    for episode in data.get("episodes", []) or []:
        goals = episode.get("goals") or []
        if goals and isinstance(goals[0], dict) and goals[0].get("position"):
            result[int(episode["episode_id"])] = [float(v) for v in goals[0]["position"]]
    return result


def regularize_artifact(
    payload_path: Path,
    original_goal_by_eid: Dict[int, List[float]],
    close_threshold: float,
    far_threshold: float,
    max_insert_iterations: int,
) -> Tuple[dict, dict]:
    payload = load_json(payload_path)
    source = load_json(payload_path.parent / "source_episode.json")
    steps_by_id = step_map(source)
    start_position = [float(v) for v in source["source_episode"]["start_position"]]

    force_original_final_goal(payload, original_goal_by_eid)
    original_subgoal_count = len(payload.get("subgoals") or [])
    cleaned, removed_start = remove_close_to_start(
        payload.get("subgoals") or [],
        start_position=start_position,
        close_threshold=close_threshold,
    )
    cleaned, removed_pairs = remove_close_pairs(cleaned, close_threshold=close_threshold)
    removed = removed_start + removed_pairs
    cleaned, inserted_start_count, skipped_start_insertions = insert_far_from_start(
        subgoals=cleaned,
        steps_by_id=steps_by_id,
        start_position=start_position,
        far_threshold=far_threshold,
        max_iterations=max_insert_iterations,
    )
    cleaned, inserted_pair_count, skipped_pair_insertions = insert_far_pairs(
        subgoals=cleaned,
        steps_by_id=steps_by_id,
        far_threshold=far_threshold,
        max_iterations=max_insert_iterations,
    )
    inserted_count = inserted_start_count + inserted_pair_count
    skipped_insertions = skipped_start_insertions + skipped_pair_insertions
    payload["subgoals"] = renumber(cleaned)
    force_original_final_goal(payload, original_goal_by_eid)

    notes = list(payload.get("postprocess_notes") or [])
    note = (
        f"regularized sub-goal spacing: removed adjacent <= {float(close_threshold)}m, "
        f"inserted midpoint sub-goals for adjacent > {float(far_threshold)}m"
    )
    if note not in notes:
        notes.append(note)
    payload["postprocess_notes"] = notes
    payload["_spacing_regularization"] = {
        "close_threshold": float(close_threshold),
        "far_threshold": float(far_threshold),
        "original_subgoal_count": int(original_subgoal_count),
        "removed_start_close_subgoals": len(removed_start),
        "removed_pair_close_subgoals": len(removed_pairs),
        "removed_close_subgoals": len(removed),
        "inserted_start_midpoint_subgoals": int(inserted_start_count),
        "inserted_pair_midpoint_subgoals": int(inserted_pair_count),
        "inserted_midpoint_subgoals": int(inserted_count),
        "skipped_far_insertions": int(skipped_insertions),
        "final_subgoal_count": len(payload["subgoals"]),
        "removed_subgoals": [
            {
                "old_subgoal_id": int(item.get("subgoal_id", -1)),
                "landmark": str(item.get("landmark") or ""),
                "best_step": int(item.get("best_step", -1)),
            }
            for item in removed
        ],
    }
    write_json(payload_path, payload, pretty=True)
    return payload, payload["_spacing_regularization"]


def regularize_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    dataset_path = Path(args.dataset_json)
    original_goal_by_eid = original_goal_by_episode_id(Path(args.original_train_json))

    if not args.no_backup:
        backup_once(dataset_path, ".before_spacing_regularization")

    payload_by_episode_id: Dict[int, dict] = {}
    aggregate = {
        "artifact_count": 0,
        "removed_close_subgoals": 0,
        "inserted_midpoint_subgoals": 0,
        "skipped_far_insertions": 0,
    }
    for payload_path in sorted(out_dir.glob("traj*/subgoals_openai.json")):
        if not args.no_backup:
            backup_once(payload_path, ".before_spacing_regularization")
        payload, stats = regularize_artifact(
            payload_path=payload_path,
            original_goal_by_eid=original_goal_by_eid,
            close_threshold=float(args.close_threshold),
            far_threshold=float(args.far_threshold),
            max_insert_iterations=int(args.max_insert_iterations),
        )
        aggregate["artifact_count"] += 1
        aggregate["removed_close_subgoals"] += int(stats["removed_close_subgoals"])
        aggregate["inserted_midpoint_subgoals"] += int(stats["inserted_midpoint_subgoals"])
        aggregate["skipped_far_insertions"] += int(stats["skipped_far_insertions"])
        for episode_id in payload.get("trajectory_episode_ids") or [payload["episode_id"]]:
            payload_by_episode_id[int(episode_id)] = payload

    dataset = load_json(dataset_path)
    changed_episodes = 0
    missing_payload = 0
    for episode in dataset.get("episodes", []) or []:
        episode_id = int(episode["episode_id"])
        payload = payload_by_episode_id.get(episode_id)
        if payload is None:
            missing_payload += 1
            continue
        episode["goals"] = goals_from_payload(payload, radius=float(args.subgoal_radius))
        changed_episodes += 1

    meta = deepcopy(dataset.get("_subgoal_generation") or {})
    meta["spacing_regularization"] = {
        "close_threshold": float(args.close_threshold),
        "far_threshold": float(args.far_threshold),
        "changed_episodes": int(changed_episodes),
        **aggregate,
    }
    dataset["_subgoal_generation"] = meta
    write_json(dataset_path, dataset, pretty=True)
    print(meta["spacing_regularization"])
    print(f"Dataset episodes updated: {changed_episodes}")
    print(f"Dataset episodes without payload: {missing_payload}")


def position(item: dict) -> List[float]:
    return [float(v) for v in item["subgoal_position"]]


def distance(a: dict, b: dict) -> float:
    return math.dist(position(a), position(b))


def distance_pos(pos: List[float], item: dict) -> float:
    return math.dist([float(v) for v in pos], position(item))


def step_map(source_payload: dict) -> Dict[int, dict]:
    steps = source_payload.get("steps") or []
    return {int(step["step"]): step for step in steps}


def force_original_final_goal(payload: dict, original_goal_by_eid: Dict[int, List[float]]) -> None:
    subgoals = payload.get("subgoals") or []
    if not subgoals:
        return
    original_goal = original_goal_by_eid.get(int(payload["episode_id"]))
    if original_goal is None:
        return
    final = subgoals[-1]
    final["subgoal_position"] = list(original_goal)
    final["subgoal_position_source"] = "original_train_goal_position"
    final["agent_position_at_keyframe"] = list(original_goal)
    payload["goal_position"] = list(original_goal)
    payload["goal_position_source"] = "original_train.goals[0].position"


def renumber(subgoals: List[dict]) -> List[dict]:
    for idx, item in enumerate(subgoals):
        item["subgoal_id"] = int(idx)
        item["is_final_goal"] = idx == len(subgoals) - 1
    return subgoals


def remove_close_to_start(
    subgoals: List[dict],
    start_position: List[float],
    close_threshold: float,
) -> Tuple[List[dict], List[dict]]:
    subgoals = [deepcopy(item) for item in sorted(subgoals, key=lambda x: int(x["subgoal_id"]))]
    removed: List[dict] = []
    while len(subgoals) > 1 and distance_pos(start_position, subgoals[0]) <= float(close_threshold) + 1e-9:
        removed.append(subgoals.pop(0))
    return renumber(subgoals), removed


def remove_close_pairs(subgoals: List[dict], close_threshold: float) -> Tuple[List[dict], List[dict]]:
    subgoals = [deepcopy(item) for item in sorted(subgoals, key=lambda x: int(x["subgoal_id"]))]
    removed: List[dict] = []
    changed = True
    while changed and len(subgoals) > 1:
        changed = False
        kept: List[dict] = []
        for candidate in subgoals:
            if not kept:
                kept.append(candidate)
                continue
            if distance(kept[-1], candidate) <= float(close_threshold) + 1e-9:
                changed = True
                if bool(candidate.get("is_final_goal")):
                    removed.append(kept.pop())
                    while kept and distance(kept[-1], candidate) <= float(close_threshold) + 1e-9:
                        removed.append(kept.pop())
                    kept.append(candidate)
                else:
                    removed.append(kept.pop())
                    kept.append(candidate)
                continue
            kept.append(candidate)
        subgoals = kept
    return renumber(subgoals), removed


def step_position(step_info: dict) -> List[float]:
    return [float(v) for v in step_info["agent_position"]]


def path_midpoint_step(steps_by_id: Dict[int, dict], start_step: int, end_step: int) -> Optional[int]:
    available = sorted(step for step in steps_by_id if int(start_step) <= step <= int(end_step))
    if len(available) < 3:
        return None

    distances: List[float] = [0.0]
    total = 0.0
    for prev, cur in zip(available[:-1], available[1:]):
        total += math.dist(step_position(steps_by_id[prev]), step_position(steps_by_id[cur]))
        distances.append(total)

    if total <= 1e-9:
        candidate = available[len(available) // 2]
    else:
        target = total / 2.0
        candidate = available[-2]
        for step, cumulative in zip(available, distances):
            if cumulative >= target:
                candidate = step
                break

    if candidate == start_step:
        candidate = available[1]
    if candidate == end_step:
        candidate = available[-2]
    if candidate == start_step or candidate == end_step:
        return None
    return int(candidate)


def inserted_subgoal(a: dict, b: dict, step_info: dict) -> dict:
    step = int(step_info["step"])
    return {
        "subgoal_id": -1,
        "landmark": f"trajectory midpoint between {a.get('landmark', 'sub-goal')} and {b.get('landmark', 'sub-goal')}",
        "landmark_source": "inserted_trajectory_midpoint",
        "is_final_goal": False,
        "best_step": step,
        "best_frame": f"memory_step_{step:04d}",
        "subgoal_position": step_position(step_info),
        "subgoal_position_source": "inserted_trajectory_midpoint",
        "agent_position_at_keyframe": step_position(step_info),
        "agent_rotation_at_keyframe": [float(v) for v in step_info["agent_rotation"]],
        "reason": "Inserted because adjacent sub-goals were more than the far-distance threshold apart.",
    }


def inserted_start_midpoint_subgoal(b: dict, step_info: dict) -> dict:
    step = int(step_info["step"])
    return {
        "subgoal_id": -1,
        "landmark": f"trajectory midpoint from start to {b.get('landmark', 'sub-goal')}",
        "landmark_source": "inserted_start_to_subgoal_midpoint",
        "is_final_goal": False,
        "best_step": step,
        "best_frame": f"memory_step_{step:04d}",
        "subgoal_position": step_position(step_info),
        "subgoal_position_source": "inserted_start_to_subgoal_midpoint",
        "agent_position_at_keyframe": step_position(step_info),
        "agent_rotation_at_keyframe": [float(v) for v in step_info["agent_rotation"]],
        "reason": "Inserted because the distance from episode start to the first sub-goal was more than the far-distance threshold.",
    }


def insert_far_pairs(
    subgoals: List[dict],
    steps_by_id: Dict[int, dict],
    far_threshold: float,
    max_iterations: int,
) -> Tuple[List[dict], int, int]:
    subgoals = [deepcopy(item) for item in sorted(subgoals, key=lambda x: int(x["subgoal_id"]))]
    inserted_count = 0
    skipped_count = 0

    for _ in range(max(1, int(max_iterations))):
        changed = False
        new_subgoals: List[dict] = []
        used_steps = {int(item["best_step"]) for item in subgoals}
        for a, b in zip(subgoals[:-1], subgoals[1:]):
            new_subgoals.append(a)
            if distance(a, b) <= float(far_threshold) + 1e-9:
                continue
            mid_step = path_midpoint_step(steps_by_id, int(a["best_step"]), int(b["best_step"]))
            if mid_step is None or mid_step in used_steps:
                skipped_count += 1
                continue
            new_subgoals.append(inserted_subgoal(a, b, steps_by_id[mid_step]))
            used_steps.add(mid_step)
            inserted_count += 1
            changed = True
        new_subgoals.append(subgoals[-1])
        subgoals = renumber(new_subgoals)
        if not changed:
            break
    return subgoals, inserted_count, skipped_count


def insert_far_from_start(
    subgoals: List[dict],
    steps_by_id: Dict[int, dict],
    start_position: List[float],
    far_threshold: float,
    max_iterations: int,
) -> Tuple[List[dict], int, int]:
    subgoals = [deepcopy(item) for item in sorted(subgoals, key=lambda x: int(x["subgoal_id"]))]
    inserted_count = 0
    skipped_count = 0
    for _ in range(max(1, int(max_iterations))):
        if not subgoals or distance_pos(start_position, subgoals[0]) <= float(far_threshold) + 1e-9:
            break
        mid_step = path_midpoint_step(steps_by_id, 0, int(subgoals[0]["best_step"]))
        if mid_step is None or mid_step in {int(item["best_step"]) for item in subgoals}:
            skipped_count += 1
            break
        subgoals.insert(0, inserted_start_midpoint_subgoal(subgoals[0], steps_by_id[mid_step]))
        inserted_count += 1
        subgoals = renumber(subgoals)
    return subgoals, inserted_count, skipped_count


def goals_from_payload(payload: dict, radius: float) -> List[dict]:
    return [
        {
            "position": [float(v) for v in item["subgoal_position"]],
            "radius": float(radius),
        }
        for item in sorted(payload.get("subgoals", []) or [], key=lambda x: int(x["subgoal_id"]))
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regularize generated R2R sub-goal spacing.")
    parser.add_argument("--out_dir", type=str, default="results/r2r_subgoal_online_train2000")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json",
    )
    parser.add_argument("--original_train_json", type=str, default="R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
    parser.add_argument("--close_threshold", type=float, default=1.0)
    parser.add_argument("--far_threshold", type=float, default=6.0)
    parser.add_argument("--max_insert_iterations", type=int, default=4)
    parser.add_argument("--subgoal_radius", type=float, default=3.0)
    parser.add_argument("--no_backup", action="store_true")
    return parser


def main() -> None:
    regularize_all(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
