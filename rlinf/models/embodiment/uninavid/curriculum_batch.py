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

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from rlinf.data.embodied_io_struct import RolloutEpochSpec


@dataclass(kw_only=True)
class ProcessedEpochBatch:
    spec: RolloutEpochSpec
    trajectory_ids: tuple[str, ...]
    fields: dict[str, Any]
    valid_chunk_mask: torch.Tensor

    def __post_init__(self):
        expected_shape = (self.spec.n_chunk_steps, len(self.trajectory_ids))
        if tuple(self.valid_chunk_mask.shape) != expected_shape:
            raise ValueError(
                f"valid_chunk_mask must have shape {expected_shape}, got "
                f"{tuple(self.valid_chunk_mask.shape)}."
            )


def compute_valid_action_slots(dones: torch.Tensor) -> torch.Tensor:
    """Keep the terminal action itself and mask every later action slot."""
    if dones.dim() < 2:
        raise ValueError("dones must include batch and action-slot dimensions.")
    prior_done = torch.zeros_like(dones, dtype=torch.bool)
    prior_done[..., 1:] = dones.to(torch.bool)[..., :-1].cummax(dim=-1).values
    return ~prior_done


def compute_truncation_aware_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminations: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE for one epoch with timeout bootstrap and no cross-epoch state."""
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must contain one bootstrap row per epoch.")
    if terminations.shape != values.shape or dones.shape != values.shape:
        raise ValueError("Boundary flags must match values, including bootstrap row.")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        next_is_terminal = terminations[step + 1].to(torch.bool)
        next_is_done = dones[step + 1].to(torch.bool)
        delta = (
            rewards[step]
            + float(gamma) * values[step + 1] * (~next_is_terminal)
            - values[step]
        )
        gae = delta + float(gamma * gae_lambda) * (~next_is_done) * gae
        advantages[step] = gae
    return advantages, advantages + values[:-1]


def compute_grpo_epoch_advantages(
    *,
    rewards: torch.Tensor,
    valid_chunk_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Compute trajectory scores before broadcasting group-relative advantages."""
    if rewards.shape[:2] != valid_chunk_mask.shape:
        raise ValueError("rewards and valid_chunk_mask must share [T, B].")
    batch_size = rewards.shape[1]
    if group_size <= 1 or batch_size % group_size != 0:
        raise ValueError("GRPO epoch batch must contain complete prompt groups.")
    scores = torch.where(
        valid_chunk_mask,
        rewards.reshape(rewards.shape[0], batch_size, -1).sum(dim=-1),
        0.0,
    ).sum(dim=0)
    grouped_scores = scores.reshape(-1, group_size)
    group_mean = grouped_scores.mean(dim=1, keepdim=True)
    group_std = grouped_scores.std(dim=1, keepdim=True)
    trajectory_advantages = (
        (grouped_scores - group_mean) / (group_std + 1e-6)
    ).reshape(batch_size)
    return trajectory_advantages.view(1, batch_size, 1).expand(rewards.shape[0], -1, -1)


def _compact_nested(value: Any, mask: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        if tuple(value.shape[:2]) != tuple(mask.shape):
            raise ValueError(
                "Every compacted tensor must start with the epoch [T, B] dimensions."
            )
        return value.flatten(0, 1)[mask.flatten()].contiguous()
    if isinstance(value, Mapping):
        return {key: _compact_nested(item, mask) for key, item in value.items()}
    raise TypeError(f"Unsupported compact field type: {type(value)!r}.")


def compact_curriculum_epochs(
    epochs: Sequence[ProcessedEpochBatch],
) -> dict[str, Any]:
    """Compact variable-length epochs and assign fixed trajectory-wise weights."""
    if not epochs:
        raise ValueError("At least one processed epoch is required.")
    bucket_trajectory_counts: Counter[str] = Counter()
    valid_lengths: dict[str, int] = {}
    seen_trajectory_ids = set()
    for epoch in epochs:
        for trajectory_index, trajectory_id in enumerate(epoch.trajectory_ids):
            if trajectory_id in seen_trajectory_ids:
                raise ValueError(f"Duplicate trajectory_id: {trajectory_id}")
            seen_trajectory_ids.add(trajectory_id)
            valid_length = int(epoch.valid_chunk_mask[:, trajectory_index].sum())
            if valid_length <= 0:
                raise ValueError(
                    f"Trajectory {trajectory_id} has no valid action chunk."
                )
            valid_lengths[trajectory_id] = valid_length
            bucket_trajectory_counts[epoch.spec.bucket_id] += 1

    compacted_epochs = []
    bucket_ids = []
    trajectory_ids = []
    chunk_weights = []
    for epoch in epochs:
        compacted_epochs.append(_compact_nested(epoch.fields, epoch.valid_chunk_mask))
        valid_flat_indices = epoch.valid_chunk_mask.flatten().nonzero().flatten()
        for flat_index in valid_flat_indices.tolist():
            trajectory_index = flat_index % len(epoch.trajectory_ids)
            trajectory_id = epoch.trajectory_ids[trajectory_index]
            valid_length = valid_lengths[trajectory_id]
            bucket_ids.append(epoch.spec.bucket_id)
            trajectory_ids.append(trajectory_id)
            chunk_weights.append(
                epoch.spec.curriculum_weight
                / bucket_trajectory_counts[epoch.spec.bucket_id]
                / valid_length
            )

    def concatenate(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(values, dim=0)
        return {key: concatenate([value[key] for value in values]) for key in first}

    compacted = concatenate(compacted_epochs)
    compacted["bucket_ids"] = tuple(bucket_ids)
    compacted["trajectory_ids"] = tuple(trajectory_ids)
    compacted["chunk_weights"] = torch.tensor(chunk_weights, dtype=torch.float64)
    _validate_weight_invariants(
        compacted["chunk_weights"],
        compacted["bucket_ids"],
        compacted["trajectory_ids"],
        epochs,
        bucket_trajectory_counts,
    )
    return compacted


def _validate_weight_invariants(
    weights: torch.Tensor,
    bucket_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    epochs: Sequence[ProcessedEpochBatch],
    bucket_trajectory_counts: Mapping[str, int],
) -> None:
    expected_alpha = {}
    for epoch in epochs:
        previous = expected_alpha.setdefault(
            epoch.spec.bucket_id, epoch.spec.curriculum_weight
        )
        if abs(previous - epoch.spec.curriculum_weight) > 1e-12:
            raise ValueError("A bucket must use one curriculum weight per iteration.")
    if abs(sum(expected_alpha.values()) - 1.0) > 1e-8:
        raise ValueError("Active bucket curriculum weights must sum to 1.")
    for bucket_id, alpha in expected_alpha.items():
        indices = [
            index for index, value in enumerate(bucket_ids) if value == bucket_id
        ]
        if not torch.isclose(
            weights[indices].sum(), torch.tensor(alpha, dtype=weights.dtype)
        ):
            raise RuntimeError(f"Chunk weights do not sum to alpha for {bucket_id}.")
        per_trajectory = alpha / bucket_trajectory_counts[bucket_id]
        for trajectory_id in dict.fromkeys(trajectory_ids[index] for index in indices):
            trajectory_indices = [
                index for index in indices if trajectory_ids[index] == trajectory_id
            ]
            if not torch.isclose(
                weights[trajectory_indices].sum(),
                torch.tensor(per_trajectory, dtype=weights.dtype),
            ):
                raise RuntimeError(
                    f"Chunk weights do not preserve trajectory mass for {trajectory_id}."
                )
    if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype)):
        raise RuntimeError("All real chunk weights must sum to 1.")
