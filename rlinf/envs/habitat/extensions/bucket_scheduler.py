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

import hashlib
import json
from numbers import Integral
from typing import Mapping, Sequence


def _require_integer(value, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field} must be an integer.")
    return int(value)


def _ordered_items(config, *, field: str) -> list[tuple[str, object]]:
    if config is None or not hasattr(config, "items"):
        raise ValueError(f"{field} must be a map.")
    items = [(str(key), value) for key, value in config.items()]
    if not items:
        raise ValueError(f"{field} must contain at least one entry.")
    names = [name for name, _ in items]
    if any(not name for name in names):
        raise ValueError(f"{field} names must be non-empty.")
    if len(set(names)) != len(names):
        raise ValueError(f"{field} names must be unique.")
    return items


def _validate_non_overlapping_ranges(
    bucket_ranges: Sequence[tuple[int, int]],
    bucket_ids: Sequence[str],
) -> None:
    # GT action lengths are integers. Every bucket except the last is [lo, hi),
    # while the last declared bucket is [lo, hi].
    integer_bounds = [
        (lower, upper if index == len(bucket_ranges) - 1 else upper - 1)
        for index, (lower, upper) in enumerate(bucket_ranges)
    ]
    for left in range(len(integer_bounds)):
        for right in range(left + 1, len(integer_bounds)):
            overlap_lower = max(integer_bounds[left][0], integer_bounds[right][0])
            overlap_upper = min(integer_bounds[left][1], integer_bounds[right][1])
            if overlap_lower <= overlap_upper:
                raise ValueError(
                    "bucket_step_range_map contains overlapping buckets "
                    f"{bucket_ids[left]!r} and {bucket_ids[right]!r}."
                )


def normalize_bucket_curriculum_plan(
    *,
    bucket_step_range_map,
    bucket_max_steps,
    curriculum_stages_map,
    rollout_epoch: int,
    curriculum_interval: int,
    effective_num_action_chunks: int,
    bucket_schedule_seed: int,
) -> dict:
    """Validate and serialize the declaration-ordered curriculum contract."""
    rollout_epoch = _require_integer(rollout_epoch, field="rollout_epoch")
    if rollout_epoch <= 0:
        raise ValueError("rollout_epoch must be positive.")
    curriculum_interval = _require_integer(
        curriculum_interval, field="curriculum_interval"
    )
    if curriculum_interval <= 0:
        raise ValueError("curriculum_interval must be positive.")
    effective_num_action_chunks = _require_integer(
        effective_num_action_chunks, field="effective_num_action_chunks"
    )
    if effective_num_action_chunks <= 0:
        raise ValueError("effective_num_action_chunks must be positive.")
    bucket_schedule_seed = _require_integer(
        bucket_schedule_seed, field="bucket_schedule_seed"
    )

    range_items = _ordered_items(
        bucket_step_range_map, field="bucket_step_range_map"
    )
    bucket_ids = [bucket_id for bucket_id, _ in range_items]
    bucket_ranges = []
    for bucket_id, configured_range in range_items:
        if not isinstance(configured_range, Sequence) or isinstance(
            configured_range, (str, bytes)
        ):
            raise ValueError(f"Bucket {bucket_id!r} range must contain two integers.")
        if len(configured_range) != 2:
            raise ValueError(f"Bucket {bucket_id!r} range must contain two integers.")
        lower = _require_integer(
            configured_range[0], field=f"Bucket {bucket_id!r} lower bound"
        )
        upper = _require_integer(
            configured_range[1], field=f"Bucket {bucket_id!r} upper bound"
        )
        if lower >= upper:
            raise ValueError(f"Bucket {bucket_id!r} must satisfy lower < upper.")
        bucket_ranges.append((lower, upper))
    _validate_non_overlapping_ranges(bucket_ranges, bucket_ids)

    if not isinstance(bucket_max_steps, Sequence) or isinstance(
        bucket_max_steps, (str, bytes)
    ):
        raise ValueError("bucket_max_steps must be a sequence.")
    if len(bucket_max_steps) != len(bucket_ids):
        raise ValueError("bucket_max_steps length must match the bucket count.")
    normalized_max_steps = []
    for bucket_id, configured_steps in zip(bucket_ids, bucket_max_steps):
        max_steps = _require_integer(
            configured_steps, field=f"bucket_max_steps for {bucket_id!r}"
        )
        if max_steps <= 0:
            raise ValueError(f"bucket_max_steps for {bucket_id!r} must be positive.")
        if max_steps % effective_num_action_chunks != 0:
            raise ValueError(
                f"Bucket {bucket_id!r} max steps {max_steps} is not divisible by "
                f"the effective action chunk count {effective_num_action_chunks}."
            )
        normalized_max_steps.append(max_steps)

    stage_items = _ordered_items(
        curriculum_stages_map, field="curriculum_stages_map"
    )
    stages = []
    for stage_name, configured_quotas in stage_items:
        if not isinstance(configured_quotas, Sequence) or isinstance(
            configured_quotas, (str, bytes)
        ):
            raise ValueError(f"Stage {stage_name!r} quotas must be a sequence.")
        if len(configured_quotas) != len(bucket_ids):
            raise ValueError(
                f"Stage {stage_name!r} quota length must match the bucket count."
            )
        quotas = tuple(
            _require_integer(value, field=f"Stage {stage_name!r} quota")
            for value in configured_quotas
        )
        if any(quota < 0 for quota in quotas):
            raise ValueError(f"Stage {stage_name!r} quotas must be non-negative.")
        if sum(quotas) != rollout_epoch:
            raise ValueError(
                f"Stage {stage_name!r} quotas must sum to rollout_epoch="
                f"{rollout_epoch}."
            )
        stages.append(
            {
                "stage_name": stage_name,
                "quotas": list(quotas),
                "target_weights": [quota / rollout_epoch for quota in quotas],
            }
        )

    bucket_plans = {}
    for index, (bucket_id, (lower, upper), max_steps) in enumerate(
        zip(bucket_ids, bucket_ranges, normalized_max_steps)
    ):
        bucket_plans[bucket_id] = {
            "bucket_id": bucket_id,
            "bucket_index": index,
            "lower_bound": lower,
            "upper_bound": upper,
            "upper_inclusive": index == len(bucket_ids) - 1,
            "horizon_steps": max_steps,
            "n_chunk_steps": max_steps // effective_num_action_chunks,
        }

    return {
        "bucket_ids": bucket_ids,
        "bucket_plans": bucket_plans,
        "stages": stages,
        "rollout_epoch": rollout_epoch,
        "curriculum_interval": curriculum_interval,
        "effective_num_action_chunks": effective_num_action_chunks,
        "bucket_schedule_seed": bucket_schedule_seed,
    }


def build_round_robin_bucket_order(
    bucket_ids: Sequence[str], quotas: Sequence[int]
) -> list[str]:
    if len(bucket_ids) != len(quotas):
        raise ValueError("Bucket IDs and quotas must have matching lengths.")
    remaining = [int(quota) for quota in quotas]
    if any(quota < 0 for quota in remaining):
        raise ValueError("Bucket quotas must be non-negative.")
    order = []
    while any(remaining):
        for index, bucket_id in enumerate(bucket_ids):
            if remaining[index] > 0:
                order.append(str(bucket_id))
                remaining[index] -= 1
    return order


def curriculum_plan_signature(plan: Mapping) -> str:
    signature_payload = {
        "bucket_ids": list(plan["bucket_ids"]),
        "bucket_schedule_seed": int(plan["bucket_schedule_seed"]),
        "buckets": [
            {
                "bucket_id": bucket_id,
                "lower_bound": int(plan["bucket_plans"][bucket_id]["lower_bound"]),
                "upper_bound": int(plan["bucket_plans"][bucket_id]["upper_bound"]),
                "upper_inclusive": bool(
                    plan["bucket_plans"][bucket_id]["upper_inclusive"]
                ),
                "episode_ids": list(plan["bucket_plans"][bucket_id]["episode_ids"]),
            }
            for bucket_id in plan["bucket_ids"]
        ],
    }
    encoded = json.dumps(
        signature_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class BucketCurriculumScheduler:
    """Global-step stage selection and deterministic global episode cursors."""

    def __init__(self, *, plan: Mapping, total_group_streams: int):
        self.plan = plan
        self.bucket_ids = tuple(str(value) for value in plan["bucket_ids"])
        self.stages = tuple(plan["stages"])
        self.rollout_epoch = int(plan["rollout_epoch"])
        self.curriculum_interval = int(plan["curriculum_interval"])
        self.total_group_streams = _require_integer(
            total_group_streams, field="total_group_streams"
        )
        if self.total_group_streams <= 0:
            raise ValueError("total_group_streams must be positive.")
        self._bucket_cursors = dict.fromkeys(self.bucket_ids, 0)
        self._plan_signature = curriculum_plan_signature(plan)

    def stage_index(self, global_step: int) -> int:
        global_step = _require_integer(global_step, field="global_step")
        if global_step < 0:
            raise ValueError("global_step must be non-negative.")
        return min(global_step // self.curriculum_interval, len(self.stages) - 1)

    def stage(self, global_step: int) -> Mapping:
        return self.stages[self.stage_index(global_step)]

    def active_bucket_ids(self, global_step: int) -> tuple[str, ...]:
        quotas = self.stage(global_step)["quotas"]
        return tuple(
            bucket_id
            for bucket_id, quota in zip(self.bucket_ids, quotas)
            if int(quota) > 0
        )

    def build_epoch_bucket_order(self, global_step: int) -> list[str]:
        order = build_round_robin_bucket_order(
            self.bucket_ids, self.stage(global_step)["quotas"]
        )
        if len(order) != self.rollout_epoch:
            raise RuntimeError("Curriculum stage produced an invalid rollout total.")
        return order

    def target_weights(self, global_step: int) -> dict[str, float]:
        weights = self.stage(global_step)["target_weights"]
        return {
            bucket_id: float(weight)
            for bucket_id, weight in zip(self.bucket_ids, weights)
        }

    def allocate_episode_ids(self, bucket_id: str) -> tuple[str, ...]:
        if bucket_id not in self._bucket_cursors:
            raise KeyError(f"Unknown curriculum bucket: {bucket_id}")
        episode_ids = tuple(self.plan["bucket_plans"][bucket_id]["episode_ids"])
        if len(episode_ids) < self.total_group_streams:
            raise ValueError(
                f"Bucket {bucket_id!r} has {len(episode_ids)} episodes, but "
                f"{self.total_group_streams} global group streams are required."
            )
        cursor = self._bucket_cursors[bucket_id]
        assigned = tuple(
            episode_ids[(cursor + index) % len(episode_ids)]
            for index in range(self.total_group_streams)
        )
        self._bucket_cursors[bucket_id] = (
            cursor + self.total_group_streams
        ) % len(episode_ids)
        return assigned

    @property
    def bucket_cursors(self) -> dict[str, int]:
        return dict(self._bucket_cursors)

    def state_dict(self) -> dict:
        return {
            "bucket_cursors": self.bucket_cursors,
            "plan_signature": self._plan_signature,
        }

    def load_state_dict(self, state: Mapping) -> None:
        if state.get("plan_signature") != self._plan_signature:
            raise ValueError(
                "Curriculum checkpoint does not match the configured bucket plan "
                "or deterministic episode order."
            )
        cursors = state.get("bucket_cursors")
        if not isinstance(cursors, Mapping) or tuple(cursors) != self.bucket_ids:
            raise ValueError(
                "Curriculum checkpoint bucket IDs or declaration order do not match."
            )
        restored = {}
        for bucket_id in self.bucket_ids:
            cursor = _require_integer(
                cursors[bucket_id], field=f"Cursor for bucket {bucket_id!r}"
            )
            episode_count = len(self.plan["bucket_plans"][bucket_id]["episode_ids"])
            if not 0 <= cursor < episode_count:
                raise ValueError(
                    f"Cursor {cursor} for bucket {bucket_id!r} is outside the "
                    f"episode list of length {episode_count}."
                )
            restored[bucket_id] = cursor
        self._bucket_cursors = restored
