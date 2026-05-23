# Copyright 2025 The RLinf Authors.
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

import math

import torch

from rlinf.algorithms.utils import (
    calculate_scores,
    preprocess_embodied_advantages_inputs,
)
from rlinf.models.embodiment.uninavid.nav_rollout import (
    NO_OP_ACTION_ID,
    STOP_ACTION_ID,
)
from rlinf.scheduler import Worker


def compute_keep_group_ratio(keep_group_mask: torch.Tensor) -> float:
    if keep_group_mask.numel() == 0:
        return 0.0
    return keep_group_mask.float().mean().item()


def all_reduce_keep_group_ratio(keep_group_mask: torch.Tensor) -> float:
    if keep_group_mask.numel() == 0:
        return 0.0
    device = Worker.torch_platform.current_device()
    stats = torch.tensor(
        [float(keep_group_mask.float().sum().item()), float(keep_group_mask.numel())],
        dtype=torch.float32,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    total_groups = stats[1].item()
    if total_groups == 0.0:
        return 0.0
    return float((stats[0] / stats[1]).item())


def compute_action_ratios(actions: torch.Tensor) -> dict[str, float]:
    if actions.numel() == 0:
        return {
            "action/stop_ratio": 0.0,
            "action/no_op_ratio": 0.0,
        }
    actions = actions.detach()
    return {
        "action/stop_ratio": (actions == STOP_ACTION_ID).float().mean().item(),
        "action/no_op_ratio": (actions == NO_OP_ACTION_ID).float().mean().item(),
    }


def all_reduce_action_ratios(actions: torch.Tensor) -> dict[str, float]:
    device = Worker.torch_platform.current_device()
    actions = actions.detach()
    stats = torch.tensor(
        [
            float((actions == STOP_ACTION_ID).float().sum().item()),
            float((actions == NO_OP_ACTION_ID).float().sum().item()),
            float(actions.numel()),
        ],
        dtype=torch.float32,
        device=device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    total_actions = stats[2].item()
    if total_actions == 0.0:
        return {
            "action/stop_ratio": 0.0,
            "action/no_op_ratio": 0.0,
        }
    return {
        "action/stop_ratio": float((stats[0] / stats[2]).item()),
        "action/no_op_ratio": float((stats[1] / stats[2]).item()),
    }


def _zero_metrics() -> dict[str, float]:
    return {
        "grpo/zero_sr_group_ratio": 0.0,
        "grpo/group_sr_mean": 0.0,
        "grpo/group_sr_max": 0.0,
        "grpo/group_sr_min": 0.0,
        "grpo/zero_score_std_group_ratio": 0.0,
        "grpo/trajectory_score_std": 0.0,
        "grpo/trajectory_score_mean": 0.0,
        "grpo/group_score_std": 0.0,
        "grpo/group_score_mean": 0.0,
    }


def _zero_stats(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "kept_group_count": torch.tensor(0.0, device=device),
        "zero_sr_group_count": torch.tensor(0.0, device=device),
        "group_sr_sum": torch.tensor(0.0, device=device),
        "group_sr_max": torch.tensor(float("-inf"), device=device),
        "group_sr_min": torch.tensor(float("inf"), device=device),
        "zero_score_std_group_count": torch.tensor(0.0, device=device),
        "trajectory_score_count": torch.tensor(0.0, device=device),
        "trajectory_score_sum": torch.tensor(0.0, device=device),
        "trajectory_score_sq_sum": torch.tensor(0.0, device=device),
        "group_score_std_sum": torch.tensor(0.0, device=device),
        "group_score_mean_sum": torch.tensor(0.0, device=device),
    }


def reduce_success_to_trajectory_success(
    success: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if success.numel() == batch_size:
        return success.reshape(batch_size)
    if success.dim() < 2 or success.shape[-1] != batch_size:
        raise ValueError(
            f"success must be a trajectory vector or step-by-trajectory tensor, got {tuple(success.shape)} for {batch_size=}"
        )
    return success.reshape(-1, batch_size).float().max(dim=0).values


def finalize_grpo_diagnostic_stats(
    stats: dict[str, torch.Tensor | float],
) -> dict[str, float]:
    kept_group_count = float(stats["kept_group_count"])
    trajectory_score_count = float(stats["trajectory_score_count"])
    if kept_group_count == 0.0 or trajectory_score_count == 0.0:
        return _zero_metrics()

    trajectory_score_mean = (
        float(stats["trajectory_score_sum"]) / trajectory_score_count
    )
    trajectory_score_var = (
        float(stats["trajectory_score_sq_sum"]) / trajectory_score_count
        - trajectory_score_mean**2
    )

    return {
        "grpo/zero_sr_group_ratio": float(stats["zero_sr_group_count"])
        / kept_group_count,
        "grpo/group_sr_mean": float(stats["group_sr_sum"]) / kept_group_count,
        "grpo/group_sr_max": float(stats["group_sr_max"]),
        "grpo/group_sr_min": float(stats["group_sr_min"]),
        "grpo/zero_score_std_group_ratio": float(stats["zero_score_std_group_count"])
        / kept_group_count,
        "grpo/trajectory_score_std": math.sqrt(max(trajectory_score_var, 0.0)),
        "grpo/trajectory_score_mean": trajectory_score_mean,
        "grpo/group_score_std": float(stats["group_score_std_sum"]) / kept_group_count,
        "grpo/group_score_mean": float(stats["group_score_mean_sum"])
        / kept_group_count,
    }


def all_reduce_grpo_diagnostic_stats(
    stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = Worker.torch_platform.current_device()
    stats = {key: value.to(device=device) for key, value in stats.items()}
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return stats

    sum_keys = [
        "kept_group_count",
        "zero_sr_group_count",
        "group_sr_sum",
        "zero_score_std_group_count",
        "trajectory_score_count",
        "trajectory_score_sum",
        "trajectory_score_sq_sum",
        "group_score_std_sum",
        "group_score_mean_sum",
    ]
    for key in sum_keys:
        torch.distributed.all_reduce(stats[key], op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(
        stats["group_sr_max"], op=torch.distributed.ReduceOp.MAX
    )
    torch.distributed.all_reduce(
        stats["group_sr_min"], op=torch.distributed.ReduceOp.MIN
    )
    return stats


def construct_grpo_trajectory_scores(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    group_size: int,
    reward_type: str,
    loss_mask: torch.Tensor | None = None,
    loss_mask_sum: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = preprocess_embodied_advantages_inputs(
        task_type="embodied",
        adv_type="grpo",
        rewards=rewards,
        dones=dones,
        group_size=group_size,
        reward_type=reward_type,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
    )
    kwargs = calculate_scores(**kwargs)
    return kwargs["rewards"]


def compute_grpo_diagnostic_stats(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    trajectory_success: torch.Tensor,
    keep_group_mask: torch.Tensor,
    group_size: int,
    reward_type: str,
    loss_mask: torch.Tensor | None = None,
    loss_mask_sum: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = rewards.device
    if keep_group_mask.numel() == 0 or not keep_group_mask.any():
        return _zero_stats(device)

    expected_trajectories = keep_group_mask.numel() * group_size
    if trajectory_success.numel() != expected_trajectories:
        raise ValueError(
            f"trajectory_success must contain {expected_trajectories} trajectory values, got {trajectory_success.numel()}"
        )
    if trajectory_success.dim() != 1:
        raise ValueError("trajectory_success must be a 1D trajectory-level vector.")
    trajectory_success = trajectory_success.to(device=device)
    keep_group_mask = keep_group_mask.to(device=device, dtype=torch.bool)
    group_success = trajectory_success.reshape(-1, group_size)
    kept_group_success = group_success[keep_group_mask]
    group_sr = kept_group_success.float().mean(dim=1)

    group_score = construct_grpo_trajectory_scores(
        rewards=rewards,
        dones=dones,
        group_size=group_size,
        reward_type=reward_type,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
    )
    kept_group_score = group_score[keep_group_mask]
    kept_trajectory_score = kept_group_score.reshape(-1)
    group_score_std_per_group = kept_group_score.std(dim=1, unbiased=False)
    group_score_mean_per_group = kept_group_score.mean(dim=1)

    return {
        "kept_group_count": torch.tensor(float(group_sr.numel()), device=device),
        "zero_sr_group_count": (group_sr == 0).float().sum(),
        "group_sr_sum": group_sr.sum(),
        "group_sr_max": group_sr.max(),
        "group_sr_min": group_sr.min(),
        "zero_score_std_group_count": (group_score_std_per_group <= 1e-6).float().sum(),
        "trajectory_score_count": torch.tensor(
            float(kept_trajectory_score.numel()), device=device
        ),
        "trajectory_score_sum": kept_trajectory_score.sum(),
        "trajectory_score_sq_sum": (kept_trajectory_score**2).sum(),
        "group_score_std_sum": group_score_std_per_group.sum(),
        "group_score_mean_sum": group_score_mean_per_group.sum(),
    }


def compute_grpo_diagnostics(**kwargs) -> dict[str, float]:
    return finalize_grpo_diagnostic_stats(compute_grpo_diagnostic_stats(**kwargs))
