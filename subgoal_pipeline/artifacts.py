from pathlib import Path
from typing import List, Optional

from .common import extract_goal_payload, extract_info_payload, instruction_text, to_float_list, write_json
from .gt import GroundTruthTrajectory
from .replay import MemoryStep


def meta_step_from_memory(step: MemoryStep, rgb_path: str = "") -> dict:
    return {
        "step": int(step.step),
        "rgb_path": rgb_path,
        "depth_path": "",
        "agent_position": list(step.agent_position),
        "agent_rotation": list(step.agent_rotation),
    }


def build_episode_meta(
    episode: object,
    source_episode: dict,
    gt_trajectory: GroundTruthTrajectory,
    steps: List[MemoryStep],
    rollout_info: dict,
    habitat_data_path: str,
    input_data_path: str,
) -> dict:
    reference_path = [list(p) for p in getattr(episode, "reference_path", [])]
    goal_position, goals_payload = extract_goal_payload(episode)
    if goal_position is None and reference_path:
        goal_position = to_float_list(reference_path[-1])
        goals_payload = [{"position": goal_position}]

    return {
        "scene_id": episode.scene_id,
        "episode_id": int(episode.episode_id),
        "trajectory_id": int(episode.trajectory_id),
        "instruction": instruction_text(source_episode),
        "input_data_path": input_data_path,
        "habitat_data_path": habitat_data_path,
        "reference_path_len": len(reference_path),
        "rollout_reference_path_len": len(gt_trajectory.locations),
        "camera_intrinsics": {},
        "start_position": to_float_list(getattr(episode, "start_position", None)),
        "start_rotation": to_float_list(getattr(episode, "start_rotation", None)),
        "goal_position": goal_position,
        "goals": goals_payload,
        "reference_path": reference_path,
        "info": extract_info_payload(episode),
        "gt_action_count": len(gt_trajectory.actions),
        "gt_forward_steps": int(gt_trajectory.forward_steps),
        **rollout_info,
        "steps": [meta_step_from_memory(step) for step in steps],
    }


def enrich_subgoals_from_memory(subgoals: List[dict], steps: List[MemoryStep]) -> List[dict]:
    step_map = {int(step.step): step for step in steps}
    enriched: List[dict] = []
    for item in subgoals:
        step = int(item["best_step"])
        if step not in step_map:
            raise ValueError(f"best_step={step} not found in memory steps.")
        step_info = step_map[step]
        agent_position = list(step_info.agent_position)
        agent_rotation = list(step_info.agent_rotation)
        enriched.append(
            {
                "subgoal_id": int(item["subgoal_id"]),
                "landmark": str(item.get("landmark") or ""),
                "landmark_source": str(item.get("landmark_source") or ""),
                "is_final_goal": bool(item.get("is_final_goal", False)),
                "best_step": int(step),
                "best_frame": f"memory_step_{step:04d}",
                "subgoal_position": list(agent_position),
                "subgoal_position_source": "agent_position_at_keyframe",
                "agent_position_at_keyframe": list(agent_position),
                "agent_rotation_at_keyframe": list(agent_rotation),
                "reason": str(item.get("reason") or ""),
            }
        )
    return enriched


def validate_subgoal_payload(payload: dict) -> None:
    subgoals = payload.get("subgoals")
    if not isinstance(subgoals, list) or not subgoals:
        raise ValueError("subgoals_openai.json must contain a non-empty subgoals list.")

    prev_step: Optional[int] = None
    for idx, item in enumerate(subgoals):
        if int(item["subgoal_id"]) != idx:
            raise ValueError(f"invalid subgoal_id at index {idx}")
        step = int(item["best_step"])
        if prev_step is not None and step <= prev_step:
            raise ValueError("subgoal best_step values must be strictly increasing.")
        prev_step = step
    if not bool(subgoals[-1].get("is_final_goal")):
        raise ValueError("the last subgoal must be marked as final goal.")


def subgoals_to_goals(subgoals: List[dict], original_episode: dict, radius: float) -> List[dict]:
    goals = [
        {
            "position": [float(v) for v in item["subgoal_position"]],
            "radius": float(radius),
        }
        for item in sorted(subgoals, key=lambda x: int(x["subgoal_id"]))
    ]
    original_goals = original_episode.get("goals") or []
    if not goals:
        return goals
    if not original_goals or not isinstance(original_goals[0], dict) or not original_goals[0].get("position"):
        raise RuntimeError(f"episode_id={original_episode.get('episode_id')} has no original goal position.")
    goals[-1]["position"] = [float(v) for v in original_goals[0]["position"]]
    return goals


def write_source_episode_artifact(
    out_dir: Path,
    source_episode: dict,
    meta: dict,
    selected_episode_ids: List[int],
    representative_id: int,
) -> None:
    payload = {
        "representative_episode_id": int(representative_id),
        "trajectory_episode_ids": [int(v) for v in selected_episode_ids],
        "source_episode": source_episode,
        "rollout_meta": {
            key: value
            for key, value in meta.items()
            if key != "steps"
        },
        "steps": [
            {
                "step": int(step["step"]),
                "agent_position": list(step["agent_position"]),
                "agent_rotation": list(step["agent_rotation"]),
            }
            for step in meta.get("steps", [])
        ],
    }
    write_json(out_dir / "source_episode.json", payload, pretty=True)
