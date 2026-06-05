import argparse
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from .common import load_json, write_json
from .regularize_spacing import goals_from_payload, original_goal_by_episode_id


def backup_once(path: Path, suffix: str) -> Path:
    backup_path = path.with_suffix(path.suffix + suffix)
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def shift_artifact(payload_path: Path, step_offset: int, original_goal_by_eid: Dict[int, List[float]]) -> dict:
    source_path = payload_path.parent / "source_episode.json"
    payload = load_json(payload_path)
    source_payload = load_json(source_path)
    steps_by_id = {int(step["step"]): step for step in source_payload.get("steps", []) or []}

    changed = 0
    clamped = 0
    for item in payload.get("subgoals", []) or []:
        if bool(item.get("is_final_goal")):
            original_goal = original_goal_by_eid.get(int(payload["episode_id"]))
            if original_goal is not None:
                item["subgoal_position"] = list(original_goal)
                item["subgoal_position_source"] = "original_train_goal_position"
                item["agent_position_at_keyframe"] = list(original_goal)
                payload["goal_position"] = list(original_goal)
                payload["goal_position_source"] = "original_train.goals[0].position"
            continue

        old_step = int(item["best_step"])
        requested_step = old_step + int(step_offset)
        available_steps = sorted(steps_by_id)
        if requested_step not in steps_by_id:
            later_or_equal = [step for step in available_steps if step >= requested_step]
            if later_or_equal:
                new_step = later_or_equal[0]
            else:
                final_step = int((payload.get("subgoals") or [])[-1]["best_step"])
                before_final = [step for step in available_steps if step < final_step]
                new_step = before_final[-1] if before_final else available_steps[-1]
            clamped += 1
        else:
            new_step = requested_step

        step_info = steps_by_id[int(new_step)]
        item["best_step"] = int(new_step)
        item["best_frame"] = f"memory_step_{int(new_step):04d}"
        item["subgoal_position"] = [float(v) for v in step_info["agent_position"]]
        item["subgoal_position_source"] = f"agent_position_at_keyframe_shifted_plus_{int(step_offset)}"
        item["agent_position_at_keyframe"] = [float(v) for v in step_info["agent_position"]]
        item["agent_rotation_at_keyframe"] = [float(v) for v in step_info["agent_rotation"]]
        changed += 1

    notes = list(payload.get("postprocess_notes") or [])
    note = f"shifted all non-final sub-goals by +{int(step_offset)} steps"
    if note not in notes:
        notes.append(note)
    payload["postprocess_notes"] = notes
    payload["_subgoal_shift"] = {
        "non_final_step_offset": int(step_offset),
        "changed_non_final_subgoals": int(changed),
        "clamped_non_final_subgoals": int(clamped),
    }
    write_json(payload_path, payload, pretty=True)
    return payload


def shift_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    dataset_path = Path(args.dataset_json)
    original_goal_by_eid = original_goal_by_episode_id(Path(args.original_train_json))

    if not args.no_backup:
        backup_once(dataset_path, ".before_shift_plus_steps")

    payload_by_episode_id: Dict[int, dict] = {}
    artifact_count = 0
    for payload_path in sorted(out_dir.glob("traj*/subgoals_openai.json")):
        if not args.no_backup:
            backup_once(payload_path, ".before_shift_plus_steps")
        payload = shift_artifact(
            payload_path=payload_path,
            step_offset=int(args.step_offset),
            original_goal_by_eid=original_goal_by_eid,
        )
        artifact_count += 1
        for episode_id in payload.get("trajectory_episode_ids") or [payload["episode_id"]]:
            payload_by_episode_id[int(episode_id)] = payload

    dataset = load_json(dataset_path)
    changed_episodes = 0
    missing_payload = 0
    for episode in dataset.get("episodes", []) or []:
        payload = payload_by_episode_id.get(int(episode["episode_id"]))
        if payload is None:
            missing_payload += 1
            continue
        episode["goals"] = goals_from_payload(payload, radius=float(args.subgoal_radius))
        changed_episodes += 1

    meta = deepcopy(dataset.get("_subgoal_generation") or {})
    meta["non_final_subgoal_shift"] = {
        "step_offset": int(args.step_offset),
        "artifact_count": int(artifact_count),
        "changed_episodes": int(changed_episodes),
        "missing_payload_episodes": int(missing_payload),
    }
    dataset["_subgoal_generation"] = meta
    write_json(dataset_path, dataset, pretty=True)
    print(meta["non_final_subgoal_shift"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shift non-final sub-goals forward along the replayed GT trajectory.")
    parser.add_argument("--out_dir", type=str, default="results/r2r_subgoal_online_train2000")
    parser.add_argument(
        "--dataset_json",
        type=str,
        default="R2R_VLNCE_v1-3_preprocessed/train/r2r_train_with_subgoals.json",
    )
    parser.add_argument("--original_train_json", type=str, default="R2R_VLNCE_v1-3_preprocessed/train/train.json.gz")
    parser.add_argument("--step_offset", type=int, default=2)
    parser.add_argument("--subgoal_radius", type=float, default=3.0)
    parser.add_argument("--no_backup", action="store_true")
    return parser


def main() -> None:
    shift_all(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
