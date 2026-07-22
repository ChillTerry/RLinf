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
    failure_stop_coeff: float
    stall_patience: int
    stall_penalty_coeff: float
    stall_recovery_patience: int
    stall_observation_patience: int
    step_cost_coeff: float = 0.0
    reward_mode: str = "subgoal_progress"


class SubgoalRewardTracker:
    def __init__(self, num_envs: int, config: SubgoalRewardConfig):
        self.num_envs = int(num_envs)
        self.config = config
        if self.config.reward_mode not in {"subgoal_progress", "simple_subgoal"}:
            raise ValueError(
                "Subgoal reward mode must be 'subgoal_progress' or "
                f"'simple_subgoal', got {self.config.reward_mode!r}."
            )
        self.active_subgoal_index = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_subgoal_count = np.zeros(self.num_envs, dtype=np.int32)
        self.num_goals = np.ones(self.num_envs, dtype=np.int32)
        self.num_subgoals = np.zeros(self.num_envs, dtype=np.int32)
        self.previous_distance_to_active_subgoal = np.zeros(
            self.num_envs, dtype=np.float32
        )
        self.initial_distance_to_active_subgoal = np.ones(
            self.num_envs, dtype=np.float32
        )
        self.previous_distance_to_final_goal = np.zeros(self.num_envs, dtype=np.float32)
        self.initial_distance_to_final_goal = np.ones(self.num_envs, dtype=np.float32)
        self.non_positive_progress_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.stalled = np.zeros(self.num_envs, dtype=bool)
        self.stall_recovery_progress_steps = np.zeros(
            self.num_envs, dtype=np.int32
        )
        self.stall_observation_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.valid_reward_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.stall_penalty_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.cumulative_normalized_progress = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_progress = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_subgoal_success = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_penalty = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_stop = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_step_cost = np.zeros(self.num_envs, dtype=np.float32)
        self.subgoal_success_given = [[False] for _ in range(self.num_envs)]
        self.stall_penalty_given = [[False] for _ in range(self.num_envs)]
        self.all_subgoals_finished = np.zeros(self.num_envs, dtype=bool)

    def reset(
        self,
        env_indices,
        distances_to_subgoals,
        distances_to_final_goal,
    ):
        self._validate_batch_lengths(
            env_indices,
            distances_to_subgoals,
            distances_to_final_goal,
        )
        for env_idx, subgoal_distances, final_distance in zip(
            env_indices,
            distances_to_subgoals,
            distances_to_final_goal,
        ):
            env_idx = int(env_idx)
            distances = self._ordered_target_distances(
                subgoal_distances,
                final_distance,
            )
            num_goals = len(distances)
            num_subgoals = max(num_goals - 1, 0)
            self.active_subgoal_index[env_idx] = 0
            self.completed_subgoal_count[env_idx] = 0
            self.num_goals[env_idx] = num_goals
            self.num_subgoals[env_idx] = num_subgoals
            self.previous_distance_to_active_subgoal[env_idx] = distances[0]
            if self.config.reward_mode == "simple_subgoal":
                self.initial_distance_to_active_subgoal[env_idx] = distances[0]
            else:
                self.initial_distance_to_active_subgoal[env_idx] = (
                    self._require_positive_reference_distance(
                        distances[0],
                    )
                )
            self.previous_distance_to_final_goal[env_idx] = distances[-1]
            self.initial_distance_to_final_goal[env_idx] = (
                self._require_positive_reference_distance(distances[-1])
                if self.config.reward_mode == "simple_subgoal"
                else distances[-1]
            )
            self.non_positive_progress_steps[env_idx] = 0
            self.stalled[env_idx] = False
            self.stall_recovery_progress_steps[env_idx] = 0
            self.stall_observation_steps[env_idx] = 0
            self.valid_reward_steps[env_idx] = 0
            self.stall_penalty_steps[env_idx] = 0
            self.cumulative_normalized_progress[env_idx] = 0.0
            self.cumulative_progress[env_idx] = 0.0
            self.cumulative_subgoal_success[env_idx] = 0.0
            self.cumulative_penalty[env_idx] = 0.0
            self.cumulative_stop[env_idx] = 0.0
            self.cumulative_step_cost[env_idx] = 0.0
            self.subgoal_success_given[env_idx] = [False] * num_subgoals
            self.stall_penalty_given[env_idx] = [False] * num_goals
            self.all_subgoals_finished[env_idx] = num_subgoals == 0

    def compute_step(
        self,
        distances_to_subgoals,
        distances_to_final_goal,
        is_stop,
        valid_mask,
        is_truncated=None,
    ):
        self._validate_batch_lengths(
            range(self.num_envs),
            distances_to_subgoals,
            distances_to_final_goal,
        )
        is_stop = np.asarray(is_stop, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if is_truncated is None:
            is_truncated = np.zeros(self.num_envs, dtype=bool)
        else:
            is_truncated = np.asarray(is_truncated, dtype=bool)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        components = self._empty_components()

        for env_idx, (subgoal_distances, final_distance) in enumerate(
            zip(distances_to_subgoals, distances_to_final_goal)
        ):
            if not valid_mask[env_idx]:
                continue

            distances = self._ordered_target_distances(
                subgoal_distances,
                final_distance,
            )
            finalgoal_idx = len(distances) - 1
            if self.all_subgoals_finished[env_idx]:
                active_idx = finalgoal_idx
            else:
                active_idx = int(self.active_subgoal_index[env_idx])
            final_distance = float(distances[finalgoal_idx])
            self.valid_reward_steps[env_idx] += 1

            active_distance = float(distances[active_idx])
            if self.config.reward_mode == "simple_subgoal":
                progress_delta = (
                    float(self.previous_distance_to_final_goal[env_idx])
                    - final_distance
                )
                denominator = float(self.initial_distance_to_final_goal[env_idx])
                progress_target_idx = finalgoal_idx
                progress_scale = 1.0
            else:
                progress_delta = (
                    float(self.previous_distance_to_active_subgoal[env_idx])
                    - active_distance
                )
                denominator = float(self.initial_distance_to_active_subgoal[env_idx])
                progress_target_idx = active_idx
                progress_scale = 1.0 / float(self.num_goals[env_idx])
            normalized_progress = float(
                np.clip(progress_delta / denominator, -1.0, 1.0)
            )
            progress_reward = (
                float(self.config.progress_reward_coef)
                * normalized_progress
                * progress_scale
            )

            stall_penalty_reward = 0.0
            if self.stalled[env_idx]:
                if normalized_progress > 0.0:
                    self.stall_recovery_progress_steps[env_idx] += 1
                    self.stall_observation_steps[env_idx] = 0
                    if self.stall_recovery_progress_steps[env_idx] >= int(
                        self.config.stall_recovery_patience
                    ):
                        self.stalled[env_idx] = False
                        self.non_positive_progress_steps[env_idx] = 0
                        self.stall_recovery_progress_steps[env_idx] = 0
                        self.stall_observation_steps[env_idx] = 0
                elif normalized_progress == 0.0:
                    self.stall_recovery_progress_steps[env_idx] = 0
                    self.stall_observation_steps[env_idx] += 1
                    if self.stall_observation_steps[env_idx] > int(
                        self.config.stall_observation_patience
                    ):
                        stall_penalty_reward = -float(self.config.stall_penalty_coeff)
                else:
                    self.stall_recovery_progress_steps[env_idx] = 0
                    self.stall_observation_steps[env_idx] = 0
                    stall_penalty_reward = -float(self.config.stall_penalty_coeff)
            elif normalized_progress <= 0.0:
                self.non_positive_progress_steps[env_idx] += 1
                if self.non_positive_progress_steps[env_idx] >= int(
                    self.config.stall_patience
                ):
                    self.stalled[env_idx] = True
                    self.stall_recovery_progress_steps[env_idx] = 0
                    self.stall_observation_steps[env_idx] = 0
                    stall_penalty_reward = -float(self.config.stall_penalty_coeff)
            else:
                self.non_positive_progress_steps[env_idx] = 0

            if stall_penalty_reward < 0.0:
                self.stall_penalty_steps[env_idx] += 1
                self.stall_penalty_given[env_idx][progress_target_idx] = True

            subgoal_success_reward = 0.0
            is_tracking_subgoal = active_idx < finalgoal_idx
            if self.config.reward_mode == "simple_subgoal":
                subgoal_success_reward, active_distance = self._update_simple_subgoals(
                    env_idx,
                    distances,
                    finalgoal_idx,
                )
            elif (
                is_tracking_subgoal
                and active_distance <= float(self.config.subgoal_success_distance)
                and not self.subgoal_success_given[env_idx][active_idx]
            ):
                num_rewarded_subgoals = max(int(self.num_subgoals[env_idx]), 1)
                subgoal_success_reward = float(
                    self.config.subgoal_success_reward_coef
                ) / float(num_rewarded_subgoals)
                self.subgoal_success_given[env_idx][active_idx] = True

            should_switch = (
                self.config.reward_mode != "simple_subgoal"
                and is_tracking_subgoal
                and active_distance <= float(self.config.subgoal_switch_distance)
            )
            if should_switch:
                self.completed_subgoal_count[env_idx] += 1
                next_idx = active_idx + 1
                if next_idx >= int(self.num_subgoals[env_idx]):
                    self.all_subgoals_finished[env_idx] = True
                    self.active_subgoal_index[env_idx] = finalgoal_idx
                    self.previous_distance_to_active_subgoal[env_idx] = final_distance
                    self.initial_distance_to_active_subgoal[env_idx] = (
                        self._require_positive_reference_distance(
                            final_distance,
                        )
                    )
                    self.non_positive_progress_steps[env_idx] = 0
                    self.stalled[env_idx] = False
                    self.stall_recovery_progress_steps[env_idx] = 0
                    self.stall_observation_steps[env_idx] = 0
                else:
                    self.active_subgoal_index[env_idx] = next_idx
                    next_distance = float(distances[next_idx])
                    self.previous_distance_to_active_subgoal[env_idx] = next_distance
                    self.initial_distance_to_active_subgoal[env_idx] = (
                        self._require_positive_reference_distance(
                            next_distance,
                        )
                    )
                    self.non_positive_progress_steps[env_idx] = 0
                    self.stalled[env_idx] = False
                    self.stall_recovery_progress_steps[env_idx] = 0
                    self.stall_observation_steps[env_idx] = 0
            elif self.config.reward_mode != "simple_subgoal":
                self.previous_distance_to_active_subgoal[env_idx] = active_distance

            self.previous_distance_to_final_goal[env_idx] = final_distance

            stop_reward = self._stop_reward(
                env_idx,
                final_distance,
                is_stop[env_idx],
                is_truncated[env_idx],
            )
            step_cost_reward = -float(self.config.step_cost_coeff)
            reward[env_idx] = (
                progress_reward
                + subgoal_success_reward
                + stall_penalty_reward
                + stop_reward
                + step_cost_reward
            )
            self.cumulative_progress[env_idx] += progress_reward
            self.cumulative_subgoal_success[env_idx] += subgoal_success_reward
            self.cumulative_penalty[env_idx] += stall_penalty_reward
            self.cumulative_stop[env_idx] += stop_reward
            self.cumulative_step_cost[env_idx] += step_cost_reward
            self.cumulative_normalized_progress[env_idx] += normalized_progress

            components["r_progress"][env_idx] = progress_reward
            components["r_subgoal_success"][env_idx] = subgoal_success_reward
            components["r_penalty"][env_idx] = stall_penalty_reward
            components["r_stop"][env_idx] = stop_reward
            components["r_step_cost"][env_idx] = step_cost_reward
            self._write_state_components(
                components,
                env_idx,
                normalized_progress,
                active_distance,
                final_distance,
                is_stop[env_idx],
            )

        return reward, components

    def _update_simple_subgoals(self, env_idx, distances, finalgoal_idx):
        num_subgoals = int(self.num_subgoals[env_idx])
        newly_reached = [
            subgoal_idx
            for subgoal_idx in range(num_subgoals)
            if not self.subgoal_success_given[env_idx][subgoal_idx]
            and float(distances[subgoal_idx])
            <= float(self.config.subgoal_success_distance)
        ]
        for subgoal_idx in newly_reached:
            self.subgoal_success_given[env_idx][subgoal_idx] = True

        completed_count = sum(self.subgoal_success_given[env_idx])
        self.completed_subgoal_count[env_idx] = completed_count
        self.all_subgoals_finished[env_idx] = completed_count == num_subgoals

        unreached = [
            subgoal_idx
            for subgoal_idx in range(num_subgoals)
            if not self.subgoal_success_given[env_idx][subgoal_idx]
        ]
        if unreached:
            active_idx = min(unreached, key=lambda idx: float(distances[idx]))
        else:
            active_idx = finalgoal_idx
        active_distance = float(distances[active_idx])
        self.active_subgoal_index[env_idx] = active_idx
        self.previous_distance_to_active_subgoal[env_idx] = active_distance

        reward_per_subgoal = float(self.config.subgoal_success_reward_coef) / float(
            max(num_subgoals, 1)
        )
        return reward_per_subgoal * len(newly_reached), active_distance

    def _stop_reward(
        self,
        env_idx: int,
        final_distance: float,
        is_stop: bool,
        is_truncated: bool,
    ) -> float:
        if not is_stop:
            if is_truncated:
                return -float(self.config.failure_stop_coeff)
            return 0.0
        # if not self.all_subgoals_finished[env_idx]:
        #     return -float(self.config.premature_stop_coeff)
        if final_distance > float(self.config.final_success_distance):
            return -float(self.config.failure_stop_coeff)
        # success_scale = 1.0 - min(
        #     final_distance / float(self.config.final_success_distance), 1.0
        # )
        return float(self.config.stop_success_reward_coef)

    def _empty_components(self):
        return {
            "r_progress": np.zeros(self.num_envs, dtype=np.float32),
            "r_subgoal_success": np.zeros(self.num_envs, dtype=np.float32),
            "r_penalty": np.zeros(self.num_envs, dtype=np.float32),
            "r_stop": np.zeros(self.num_envs, dtype=np.float32),
            "r_step_cost": np.zeros(self.num_envs, dtype=np.float32),
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
            "cumulative_step_cost": self.cumulative_step_cost.copy(),
            "num_goals": self.num_goals.astype(np.float32).copy(),
            "num_subgoals": self.num_subgoals.astype(np.float32).copy(),
            "num_intermediate_subgoals": self.num_subgoals.astype(np.float32).copy(),
            "subgoal_completion_ratio": np.zeros(self.num_envs, dtype=np.float32),
            "precision_subgoal_success_ratio": np.zeros(
                self.num_envs, dtype=np.float32
            ),
            "stall_penalty_rate": np.zeros(self.num_envs, dtype=np.float32),
            "any_stall_penalty_subgoal": np.zeros(self.num_envs, dtype=np.float32),
            "stall_then_goal_success": np.zeros(self.num_envs, dtype=np.float32),
            "first_stall_penalty_rate": np.zeros(self.num_envs, dtype=np.float32),
            "premature_stop_ratio": np.zeros(self.num_envs, dtype=np.float32),
            "final_goal_success_ratio": np.zeros(self.num_envs, dtype=np.float32),
            "distance_to_final_goal": np.zeros(self.num_envs, dtype=np.float32),
            "stop_action_ratio": np.zeros(self.num_envs, dtype=np.float32),
            "mean_normalized_progress": np.zeros(self.num_envs, dtype=np.float32),
        }

    def _write_state_components(
        self,
        components,
        env_idx: int,
        normalized_progress: float,
        distance_to_active_subgoal: float,
        distance_to_final_goal: float,
        is_stop: bool,
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
        components["cumulative_step_cost"][env_idx] = float(
            self.cumulative_step_cost[env_idx]
        )
        components["num_goals"][env_idx] = float(self.num_goals[env_idx])
        components["num_subgoals"][env_idx] = float(self.num_subgoals[env_idx])
        components["num_intermediate_subgoals"][env_idx] = float(
            self.num_subgoals[env_idx]
        )
        components["subgoal_completion_ratio"][env_idx] = self._safe_ratio(
            float(self.completed_subgoal_count[env_idx]),
            float(max(int(self.num_subgoals[env_idx]), 1)),
        )
        components["precision_subgoal_success_ratio"][env_idx] = self._safe_ratio(
            float(sum(self.subgoal_success_given[env_idx])),
            float(max(int(self.num_subgoals[env_idx]), 1)),
        )
        components["stall_penalty_rate"][env_idx] = self._safe_ratio(
            float(self.stall_penalty_steps[env_idx]),
            float(self.valid_reward_steps[env_idx]),
        )
        has_stall = any(self.stall_penalty_given[env_idx])
        final_goal_success = bool(
            is_stop
            and distance_to_final_goal <= float(self.config.final_success_distance)
        )
        premature_stop = bool(is_stop and not self.all_subgoals_finished[env_idx])
        components["any_stall_penalty_subgoal"][env_idx] = float(has_stall)
        components["stall_then_goal_success"][env_idx] = float(
            has_stall and final_goal_success
        )
        components["first_stall_penalty_rate"][env_idx] = self._safe_ratio(
            float(sum(self.stall_penalty_given[env_idx])),
            float(self.num_goals[env_idx]),
        )
        components["premature_stop_ratio"][env_idx] = float(premature_stop)
        components["final_goal_success_ratio"][env_idx] = float(final_goal_success)
        components["distance_to_final_goal"][env_idx] = float(distance_to_final_goal)
        components["stop_action_ratio"][env_idx] = float(is_stop)
        components["mean_normalized_progress"][env_idx] = self._safe_ratio(
            float(self.cumulative_normalized_progress[env_idx]),
            float(self.valid_reward_steps[env_idx]),
        )

    @staticmethod
    def _safe_ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0.0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _require_positive_reference_distance(distance):
        distance = float(distance)
        if distance <= 0.0:
            raise ValueError(
                "Active target reference distance must be positive; "
            )
        return distance

    @staticmethod
    def _validate_distances(distances):
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim != 1 or len(distances) == 0:
            raise ValueError("Target distances must be a non-empty 1D sequence.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("Target distances must be finite.")
        if np.any(distances < 0.0):
            raise ValueError("Target distances must be non-negative.")
        return distances

    @classmethod
    def _ordered_target_distances(cls, subgoal_distances, final_distance):
        subgoal_distances = np.asarray(subgoal_distances, dtype=np.float32)
        if subgoal_distances.ndim != 1:
            raise ValueError("Subgoal distances must be a 1D sequence.")
        return cls._validate_distances(
            [*subgoal_distances.tolist(), float(final_distance)]
        )

    @staticmethod
    def _validate_batch_lengths(
        env_indices,
        distances_to_subgoals,
        distances_to_final_goal,
    ):
        expected = len(env_indices)
        if len(distances_to_subgoals) != expected:
            raise ValueError(
                "Subgoal distance batch must match the environment batch size."
            )
        if len(distances_to_final_goal) != expected:
            raise ValueError(
                "Final-goal distance batch must match the environment batch size."
            )
