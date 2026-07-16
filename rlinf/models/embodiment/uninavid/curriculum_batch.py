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

import math
import random
from collections import Counter, deque
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


@dataclass(frozen=True, kw_only=True)
class CurriculumGlobalBatchPlan:
    sample_slots: tuple[int | None, ...]
    rank_slots: tuple[tuple[int | None, ...], ...]
    bucket_mass: dict[str, float]
    bucket_mass_error: dict[str, float]


@dataclass(frozen=True, kw_only=True)
class CurriculumTrainingPlan:
    global_batches: tuple[CurriculumGlobalBatchPlan, ...]
    global_batch_size: int
    world_size: int
    real_chunk_count: int
    alignment_padding_count: int

    @property
    def num_updates(self) -> int:
        return len(self.global_batches)


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


def _build_bucket_queues(
    *,
    bucket_ids: Sequence[str],
    trajectory_ids: Sequence[str],
    seed: int,
) -> dict[str, deque[int]]:
    bucket_trajectories: dict[str, dict[str, deque[int]]] = {}
    for index, (bucket_id, trajectory_id) in enumerate(zip(bucket_ids, trajectory_ids)):
        bucket_trajectories.setdefault(bucket_id, {}).setdefault(
            trajectory_id, deque()
        ).append(index)

    rng = random.Random(seed)
    queues = {}
    for bucket_id, trajectories in bucket_trajectories.items():
        trajectory_order = list(trajectories)
        rng.shuffle(trajectory_order)
        queue = deque()
        while trajectory_order:
            next_order = []
            for trajectory_id in trajectory_order:
                trajectory_queue = trajectories[trajectory_id]
                queue.append(trajectory_queue.popleft())
                if trajectory_queue:
                    next_order.append(trajectory_id)
            trajectory_order = next_order
        queues[bucket_id] = queue
    return queues


def plan_curriculum_global_batches(
    compacted: Mapping[str, Any],
    *,
    global_batch_size: int,
    world_size: int,
    micro_batch_size: int,
    seed: int,
    update_epoch: int = 0,
) -> CurriculumTrainingPlan:
    """Plan fixed-mass global batches and equal rank slots without resampling."""
    if global_batch_size <= 0 or world_size <= 0 or micro_batch_size <= 0:
        raise ValueError("Batch sizes and world_size must be positive.")
    if global_batch_size % world_size != 0:
        raise ValueError("global_batch_size must be divisible by world_size.")
    slots_per_rank = global_batch_size // world_size
    if slots_per_rank % micro_batch_size != 0:
        raise ValueError(
            "global_batch_size / world_size must be divisible by micro_batch_size."
        )

    bucket_ids = tuple(compacted["bucket_ids"])
    trajectory_ids = tuple(compacted["trajectory_ids"])
    weights = compacted["chunk_weights"].to(dtype=torch.float64)
    chunk_count = len(bucket_ids)
    if chunk_count == 0 or len(trajectory_ids) != chunk_count:
        raise ValueError("Compacted curriculum metadata is empty or misaligned.")
    if weights.shape != (chunk_count,):
        raise ValueError("chunk_weights must contain one value per real chunk.")

    num_updates = int(math.ceil(chunk_count / global_batch_size))
    active_buckets = tuple(dict.fromkeys(bucket_ids))
    valid_counts = Counter(bucket_ids)
    for bucket_id in active_buckets:
        if valid_counts[bucket_id] < num_updates:
            raise ValueError(
                f"Bucket {bucket_id} has {valid_counts[bucket_id]} valid chunks, "
                f"but {num_updates} global batches require at least "
                f"{num_updates}."
            )

    alpha = {
        bucket_id: float(
            weights[
                torch.tensor(
                    [value == bucket_id for value in bucket_ids], dtype=torch.bool
                )
            ].sum()
        )
        for bucket_id in active_buckets
    }
    if abs(sum(alpha.values()) - 1.0) > 1e-8:
        raise ValueError("Compacted curriculum chunk weights must sum to 1.")
    queues = _build_bucket_queues(
        bucket_ids=bucket_ids,
        trajectory_ids=trajectory_ids,
        seed=int(seed) + int(update_epoch),
    )

    plans = []
    remainder = chunk_count % num_updates
    for batch_index in range(num_updates):
        real_quota = chunk_count // num_updates + (batch_index < remainder)
        if real_quota < len(active_buckets):
            raise RuntimeError(
                "Global batch real-sample quota cannot include every active bucket."
            )
        selected = []
        mass = dict.fromkeys(active_buckets, 0.0)
        for bucket_id in active_buckets:
            sample_index = queues[bucket_id].popleft()
            selected.append(sample_index)
            mass[bucket_id] += float(weights[sample_index])

        future_batches = num_updates - batch_index - 1
        while len(selected) < real_quota:
            eligible = [
                bucket_id
                for bucket_id in active_buckets
                if len(queues[bucket_id]) > future_batches
            ]
            if not eligible:
                raise RuntimeError(
                    "Curriculum planner exhausted unreserved bucket samples."
                )
            bucket_id = max(
                eligible,
                key=lambda value: (alpha[value] / num_updates - mass[value], value),
            )
            sample_index = queues[bucket_id].popleft()
            selected.append(sample_index)
            mass[bucket_id] += float(weights[sample_index])

        slots: tuple[int | None, ...] = tuple(
            selected + [None] * (global_batch_size - len(selected))
        )
        rank_slots = tuple(
            tuple(slots[start : start + slots_per_rank])
            for start in range(0, global_batch_size, slots_per_rank)
        )
        mass_error = {
            bucket_id: abs(num_updates * mass[bucket_id] - alpha[bucket_id])
            for bucket_id in active_buckets
        }
        plans.append(
            CurriculumGlobalBatchPlan(
                sample_slots=slots,
                rank_slots=rank_slots,
                bucket_mass=mass,
                bucket_mass_error=mass_error,
            )
        )

    if any(queue for queue in queues.values()):
        raise RuntimeError("Curriculum planner did not consume every real chunk.")
    consumed = [
        index for plan in plans for index in plan.sample_slots if index is not None
    ]
    if sorted(consumed) != list(range(chunk_count)):
        raise RuntimeError("Every real chunk must be consumed exactly once.")
    padding_count = num_updates * global_batch_size - chunk_count
    return CurriculumTrainingPlan(
        global_batches=tuple(plans),
        global_batch_size=global_batch_size,
        world_size=world_size,
        real_chunk_count=chunk_count,
        alignment_padding_count=padding_count,
    )


def materialize_curriculum_rank_batch(
    compacted: Mapping[str, Any],
    *,
    training_plan: CurriculumTrainingPlan,
    update_index: int,
    rank: int,
) -> dict[str, Any]:
    if not 0 <= update_index < training_plan.num_updates:
        raise IndexError("Invalid curriculum update_index.")
    if not 0 <= rank < training_plan.world_size:
        raise IndexError("Invalid actor rank.")
    slots = training_plan.global_batches[update_index].rank_slots[rank]
    real_indices = [index if index is not None else 0 for index in slots]
    padding_mask = torch.tensor([index is None for index in slots], dtype=torch.bool)

    def materialize(value):
        if isinstance(value, torch.Tensor):
            result = value[real_indices].clone()
            result[padding_mask] = 0
            return result
        if isinstance(value, Mapping):
            return {key: materialize(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(
                "__padding__" if index is None else value[index] for index in slots
            )
        raise TypeError(f"Unsupported planned batch field type: {type(value)!r}.")

    batch = {key: materialize(value) for key, value in compacted.items()}
    batch["alignment_padding_mask"] = padding_mask
    batch["train_chunk_weights"] = batch["chunk_weights"] * training_plan.num_updates
    return batch


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
