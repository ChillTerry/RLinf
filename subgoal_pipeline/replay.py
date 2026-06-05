from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from .common import ACTION_ID_TO_NAME
from .gt import GroundTruthTrajectory


@dataclass
class MemoryStep:
    step: int
    rgb: np.ndarray
    agent_position: List[float]
    agent_rotation: List[float]


def replay_gt_actions_in_memory(
    env: Any,
    episode: Any,
    gt_trajectory: GroundTruthTrajectory,
    max_steps: int,
) -> Tuple[List[MemoryStep], dict]:
    env.current_episode = episode
    obs = env.reset()
    gt_actions = [int(a) for a in gt_trajectory.actions]

    steps: List[MemoryStep] = []
    step_idx = 0
    action_cursor = 0
    stop_action_seen = False

    while (not env.episode_over) and step_idx < int(max_steps):
        agent_state = env.sim.get_agent_state()
        steps.append(
            MemoryStep(
                step=int(step_idx),
                rgb=np.asarray(obs["rgb"]).copy(),
                agent_position=agent_state.position.tolist(),
                agent_rotation=[
                    agent_state.rotation.x,
                    agent_state.rotation.y,
                    agent_state.rotation.z,
                    agent_state.rotation.w,
                ],
            )
        )

        if action_cursor >= len(gt_actions):
            break
        action = int(gt_actions[action_cursor])
        action_cursor += 1
        if action == 0:
            stop_action_seen = True
            break

        obs = env.step(action)
        step_idx += 1

    return steps, {
        "rollout_source": "gt_actions",
        "rollout_action_id_to_name": ACTION_ID_TO_NAME,
        "rollout_action_count": len(gt_actions),
        "rollout_replayed_action_count": int(action_cursor),
        "rollout_executed_non_stop_actions": int(step_idx),
        "rollout_stop_action_seen": bool(stop_action_seen),
        "rollout_truncated_by_max_steps": (
            action_cursor < len(gt_actions)
            and not stop_action_seen
            and step_idx >= int(max_steps)
        ),
        "gt_locations_len": len(gt_trajectory.locations),
        "gt_forward_steps": int(gt_trajectory.forward_steps),
    }


def sample_memory_steps_with_final(
    steps: List[MemoryStep],
    frame_stride: int,
    max_frames: int,
) -> List[MemoryStep]:
    if not steps:
        return []
    stride = max(1, int(frame_stride))
    sampled = list(steps[::stride])
    if sampled[-1].step != steps[-1].step:
        sampled.append(steps[-1])

    if int(max_frames) >= 0 and len(sampled) > int(max_frames):
        keep = max(1, int(max_frames))
        if keep == 1:
            sampled = [sampled[-1]]
        else:
            idxs = np.linspace(0, len(sampled) - 1, num=keep, dtype=int).tolist()
            idxs[-1] = len(sampled) - 1
            unique_idxs: List[int] = []
            for idx in idxs:
                if idx not in unique_idxs:
                    unique_idxs.append(idx)
            sampled = [sampled[idx] for idx in unique_idxs]
            if sampled[-1].step != steps[-1].step:
                sampled[-1] = steps[-1]
    return sampled

