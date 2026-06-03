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

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubgoalRewardConfig:
    progress_reward_coef: float
    subgoal_success_reward_coef: float
    subgoal_switch_distance: float
    subgoal_success_distance: float
    stop_success_reward_coef: float
    final_success_distance: float
    premature_stop_coeff: float
    stall_patience: int
    stall_penalty_coeff: float
    distance_epsilon: float = 1.0e-6


class SubgoalRewardTracker:
    def __init__(self, num_envs: int, config: SubgoalRewardConfig):
        self.num_envs = int(num_envs)
        self.config = config
        self.active_subgoal_index = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_subgoal_count = np.zeros(self.num_envs, dtype=np.int32)
        self.num_subgoals = np.ones(self.num_envs, dtype=np.int32)
        self.previous_distance_to_active_subgoal = np.zeros(
            self.num_envs, dtype=np.float32
        )
        self.initial_distance_to_active_subgoal = np.ones(
            self.num_envs, dtype=np.float32
        )
        self.non_positive_progress_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.cumulative_progress = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_subgoal_success = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_penalty = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_stop = np.zeros(self.num_envs, dtype=np.float32)
        self.subgoal_success_given = [[False] for _ in range(self.num_envs)]
        self.all_subgoals_finished = np.zeros(self.num_envs, dtype=bool)

    def reset(self, env_indices, distances_to_subgoals):
        for env_idx, distances in zip(env_indices, distances_to_subgoals):
            env_idx = int(env_idx)
            distances = self._validate_distances(distances)
            self.active_subgoal_index[env_idx] = 0
            self.completed_subgoal_count[env_idx] = 0
            self.num_subgoals[env_idx] = len(distances)
            self.previous_distance_to_active_subgoal[env_idx] = distances[0]
            self.initial_distance_to_active_subgoal[env_idx] = max(
                distances[0], self.config.distance_epsilon
            )
            self.non_positive_progress_steps[env_idx] = 0
            self.cumulative_progress[env_idx] = 0.0
            self.cumulative_subgoal_success[env_idx] = 0.0
            self.cumulative_penalty[env_idx] = 0.0
            self.cumulative_stop[env_idx] = 0.0
            self.subgoal_success_given[env_idx] = [False] * len(distances)
            self.all_subgoals_finished[env_idx] = False

    def compute_step(self, distances_to_subgoals, is_stop, valid_mask):
        is_stop = np.asarray(is_stop, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        components = self._empty_components()

        for env_idx, distances in enumerate(distances_to_subgoals):
            if not valid_mask[env_idx]:
                continue

            distances = self._validate_distances(distances)
            active_idx = int(self.active_subgoal_index[env_idx])
            final_distance = float(distances[-1])

            if self.all_subgoals_finished[env_idx]:
                stop_reward = self._stop_reward(env_idx, final_distance, is_stop[env_idx])
                self.cumulative_stop[env_idx] += stop_reward
                reward[env_idx] = stop_reward
                components["r_stop"][env_idx] = stop_reward
                self._write_state_components(components, env_idx, 0.0, final_distance)
                continue

            active_distance = float(distances[active_idx])
            progress_delta = (
                float(self.previous_distance_to_active_subgoal[env_idx])
                - active_distance
            )
            denominator = max(
                float(self.initial_distance_to_active_subgoal[env_idx]),
                self.config.distance_epsilon,
            )
            normalized_progress = float(
                np.clip(progress_delta / denominator, -1.0, 1.0)
            )
            progress_reward = float(self.config.progress_reward_coef) * max(
                normalized_progress, 0.0
            )

            if normalized_progress <= 0.0:
                self.non_positive_progress_steps[env_idx] += 1
            else:
                self.non_positive_progress_steps[env_idx] = 0

            penalty = 0.0
            if self.non_positive_progress_steps[env_idx] >= int(
                self.config.stall_patience
            ):
                penalty = -float(self.config.stall_penalty_coeff)

            subgoal_success = 0.0
            if (
                active_distance <= float(self.config.subgoal_success_distance)
                and not self.subgoal_success_given[env_idx][active_idx]
            ):
                subgoal_success = float(
                    self.config.subgoal_success_reward_coef
                ) / float(self.num_subgoals[env_idx])
                self.subgoal_success_given[env_idx][active_idx] = True

            should_switch = active_distance <= float(self.config.subgoal_switch_distance)
            if should_switch:
                self.completed_subgoal_count[env_idx] += 1
                next_idx = active_idx + 1
                if next_idx >= int(self.num_subgoals[env_idx]):
                    self.all_subgoals_finished[env_idx] = True
                    self.active_subgoal_index[env_idx] = active_idx
                else:
                    self.active_subgoal_index[env_idx] = next_idx
                    next_distance = float(distances[next_idx])
                    self.previous_distance_to_active_subgoal[env_idx] = next_distance
                    self.initial_distance_to_active_subgoal[env_idx] = max(
                        next_distance, self.config.distance_epsilon
                    )
                    self.non_positive_progress_steps[env_idx] = 0
            else:
                self.previous_distance_to_active_subgoal[env_idx] = active_distance

            stop_reward = self._stop_reward(env_idx, final_distance, is_stop[env_idx])
            reward[env_idx] = (
                progress_reward + subgoal_success + penalty + stop_reward
            )
            self.cumulative_progress[env_idx] += progress_reward
            self.cumulative_subgoal_success[env_idx] += subgoal_success
            self.cumulative_penalty[env_idx] += penalty
            self.cumulative_stop[env_idx] += stop_reward

            components["r_progress"][env_idx] = progress_reward
            components["r_subgoal_success"][env_idx] = subgoal_success
            components["r_penalty"][env_idx] = penalty
            components["r_stop"][env_idx] = stop_reward
            self._write_state_components(
                components,
                env_idx,
                normalized_progress,
                active_distance,
            )

        return reward, components

    def _stop_reward(self, env_idx: int, final_distance: float, is_stop: bool) -> float:
        if not is_stop:
            return 0.0
        if not self.all_subgoals_finished[env_idx]:
            return -float(self.config.premature_stop_coeff)
        success_scale = 1.0 - min(
            final_distance / float(self.config.final_success_distance), 1.0
        )
        return float(self.config.stop_success_reward_coef) * success_scale

    def _empty_components(self):
        return {
            "r_progress": np.zeros(self.num_envs, dtype=np.float32),
            "r_subgoal_success": np.zeros(self.num_envs, dtype=np.float32),
            "r_penalty": np.zeros(self.num_envs, dtype=np.float32),
            "r_stop": np.zeros(self.num_envs, dtype=np.float32),
            "active_subgoal_index": self.active_subgoal_index.astype(np.float32).copy(),
            "completed_subgoal_count": self.completed_subgoal_count.astype(
                np.float32
            ).copy(),
            "distance_to_active_subgoal": np.zeros(self.num_envs, dtype=np.float32),
            "normalized_progress": np.zeros(self.num_envs, dtype=np.float32),
            "cumulative_progress": self.cumulative_progress.copy(),
            "cumulative_subgoal_success": self.cumulative_subgoal_success.copy(),
            "cumulative_penalty": self.cumulative_penalty.copy(),
            "cumulative_stop": self.cumulative_stop.copy(),
        }

    def _write_state_components(
        self,
        components,
        env_idx: int,
        normalized_progress: float,
        distance_to_active_subgoal: float,
    ):
        components["active_subgoal_index"][env_idx] = float(
            self.active_subgoal_index[env_idx]
        )
        components["completed_subgoal_count"][env_idx] = float(
            self.completed_subgoal_count[env_idx]
        )
        components["distance_to_active_subgoal"][env_idx] = float(
            distance_to_active_subgoal
        )
        components["normalized_progress"][env_idx] = float(normalized_progress)
        components["cumulative_progress"][env_idx] = float(
            self.cumulative_progress[env_idx]
        )
        components["cumulative_subgoal_success"][env_idx] = float(
            self.cumulative_subgoal_success[env_idx]
        )
        components["cumulative_penalty"][env_idx] = float(
            self.cumulative_penalty[env_idx]
        )
        components["cumulative_stop"][env_idx] = float(self.cumulative_stop[env_idx])

    @staticmethod
    def _validate_distances(distances):
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim != 1 or len(distances) == 0:
            raise ValueError("Subgoal distances must be a non-empty 1D sequence.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("Subgoal distances must be finite.")
        return distances
