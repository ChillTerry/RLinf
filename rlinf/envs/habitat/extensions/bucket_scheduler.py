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

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CurriculumStage:
    active_bucket_count: int
    weights: tuple[float, ...]

    @classmethod
    def from_config(cls, config) -> "CurriculumStage":
        return cls(
            active_bucket_count=int(config.active_bucket_count),
            weights=tuple(float(weight) for weight in config.weights),
        )


def validate_curriculum_stages(
    stages: Sequence[CurriculumStage], *, rollout_epoch: int
) -> None:
    if not stages:
        raise ValueError("curriculum_stages must contain at least one stage.")
    previous_bucket_count = 0
    for stage_index, stage in enumerate(stages):
        if stage.active_bucket_count <= previous_bucket_count:
            raise ValueError(
                "curriculum_stages active_bucket_count must increase strictly."
            )
        if stage.active_bucket_count > rollout_epoch:
            raise ValueError(
                f"Curriculum stage {stage_index} activates "
                f"{stage.active_bucket_count} buckets, exceeding rollout_epoch="
                f"{rollout_epoch}."
            )
        if len(stage.weights) != stage.active_bucket_count:
            raise ValueError(
                f"Curriculum stage {stage_index} weights must match "
                "active_bucket_count."
            )
        if any(weight <= 0.0 for weight in stage.weights):
            raise ValueError("Curriculum weights must be strictly positive.")
        if abs(sum(stage.weights) - 1.0) > 1e-8:
            raise ValueError("Curriculum weights must sum to 1.")
        previous_bucket_count = stage.active_bucket_count


class BucketCurriculumScheduler:
    """Stateful, deterministic rollout-epoch quota scheduler."""

    def __init__(
        self,
        *,
        bucket_ids: Sequence[str],
        stages: Sequence[CurriculumStage],
        rollout_epoch: int,
        curriculum_interval: int,
        success_threshold: float | None = None,
        seed: int = 0,
    ):
        validate_curriculum_stages(stages, rollout_epoch=rollout_epoch)
        if len(bucket_ids) < stages[-1].active_bucket_count:
            raise ValueError("Not enough buckets for the configured curriculum stages.")
        if curriculum_interval <= 0:
            raise ValueError("curriculum_interval must be positive.")
        self.bucket_ids = tuple(str(bucket_id) for bucket_id in bucket_ids)
        self.stages = tuple(stages)
        self.rollout_epoch = int(rollout_epoch)
        self.curriculum_interval = int(curriculum_interval)
        self.success_threshold = (
            None if success_threshold is None else float(success_threshold)
        )
        self.seed = int(seed)
        self.stage_index = 0
        self.iteration = 0
        self._deficit = [0.0] * self.stages[0].active_bucket_count
        self._success = {}
        self._timeout = {}

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def active_bucket_ids(self) -> tuple[str, ...]:
        return self.bucket_ids[: self.stage.active_bucket_count]

    def _resize_deficit(self) -> None:
        active_count = self.stage.active_bucket_count
        if len(self._deficit) < active_count:
            self._deficit.extend([0.0] * (active_count - len(self._deficit)))
        else:
            self._deficit = self._deficit[:active_count]

    def allocate_epoch_quotas(self) -> dict[str, int]:
        """Allocate every active bucket once, then distribute remaining slots."""
        stage = self.stage
        self._resize_deficit()
        quotas = [1] * stage.active_bucket_count
        for index, weight in enumerate(stage.weights):
            self._deficit[index] += self.rollout_epoch * weight - 1.0

        for _ in range(self.rollout_epoch - stage.active_bucket_count):
            selected = max(
                range(stage.active_bucket_count),
                key=lambda index: (self._deficit[index], -index),
            )
            quotas[selected] += 1
            self._deficit[selected] -= 1.0

        return dict(zip(self.active_bucket_ids, quotas))

    def build_epoch_bucket_order(self) -> list[str]:
        quotas = self.allocate_epoch_quotas()
        order = []
        while quotas:
            for bucket_id in self.active_bucket_ids:
                remaining = quotas.get(bucket_id, 0)
                if remaining > 0:
                    order.append(bucket_id)
                    if remaining == 1:
                        del quotas[bucket_id]
                    else:
                        quotas[bucket_id] = remaining - 1
        if len(order) != self.rollout_epoch:
            raise RuntimeError("Curriculum quota allocation produced an invalid total.")
        return order

    def record_bucket_metrics(
        self,
        *,
        success_rates: Mapping[str, float] | None = None,
        timeout_rates: Mapping[str, float] | None = None,
    ) -> None:
        if success_rates is not None:
            self._success.update(
                {str(key): float(value) for key, value in success_rates.items()}
            )
        if timeout_rates is not None:
            self._timeout.update(
                {str(key): float(value) for key, value in timeout_rates.items()}
            )

    def advance(self) -> bool:
        self.iteration += 1
        if self.stage_index == len(self.stages) - 1:
            return False
        interval_ready = self.iteration % self.curriculum_interval == 0
        success_ready = False
        if self.success_threshold is not None:
            hardest = self.active_bucket_ids[-1]
            success_ready = (
                self._success.get(hardest, float("-inf")) >= self.success_threshold
            )
        if not interval_ready and not success_ready:
            return False
        self.stage_index += 1
        self._resize_deficit()
        return True

    def state_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "iteration": self.iteration,
            "deficit": list(self._deficit),
            "success": dict(self._success),
            "timeout": dict(self._timeout),
            "seed": self.seed,
        }

    def load_state_dict(self, state: Mapping) -> None:
        stage_index = int(state["stage_index"])
        if not 0 <= stage_index < len(self.stages):
            raise ValueError("Invalid curriculum stage_index in checkpoint.")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("Curriculum checkpoint seed does not match config.")
        self.stage_index = stage_index
        self.iteration = int(state["iteration"])
        self._deficit = [float(value) for value in state["deficit"]]
        self._resize_deficit()
        self._success = {
            str(key): float(value) for key, value in state.get("success", {}).items()
        }
        self._timeout = {
            str(key): float(value) for key, value in state.get("timeout", {}).items()
        }
