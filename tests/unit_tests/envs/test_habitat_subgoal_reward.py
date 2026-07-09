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

import math

import numpy as np
import pytest

from rlinf.envs.habitat.extensions.subgoal import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)


def _tracker(num_envs=1, **kwargs):
    config = SubgoalRewardConfig(
        progress_reward_coef=kwargs.get("progress_reward_coef", 1.0),
        subgoal_success_reward_coef=kwargs.get("subgoal_success_reward_coef", 6.0),
        subgoal_switch_distance=kwargs.get("subgoal_switch_distance", 1.0),
        subgoal_success_distance=kwargs.get("subgoal_success_distance", 0.5),
        stop_success_reward_coef=kwargs.get("stop_success_reward_coef", 10.0),
        final_success_distance=kwargs.get("final_success_distance", 3.0),
        premature_stop_coeff=kwargs.get("premature_stop_coeff", 4.0),
        failure_stop_coeff=kwargs.get("failure_stop_coeff", 5.0),
        step_cost_coeff=kwargs.get("step_cost_coeff", 0.0),
        stall_patience=kwargs.get("stall_patience", 3),
        stall_penalty_coeff=kwargs.get("stall_penalty_coeff", 1.0),
        stall_recovery_patience=kwargs.get("stall_recovery_patience", 2),
        stall_observation_patience=kwargs.get("stall_observation_patience", 2),
    )
    return SubgoalRewardTracker(num_envs=num_envs, config=config)


def test_progress_is_normalized_by_active_subgoal_initial_distance():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 8.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[3.0, 7.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(reward[0], 0.125, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.125, rel_tol=1e-6)
    assert components["normalized_progress"][0] == 0.25
    assert tracker.active_subgoal_index.tolist() == [0]


def test_progress_reward_is_normalized_by_episode_subgoal_count():
    tracker = _tracker(progress_reward_coef=1.0)
    tracker.reset([0], [[4.0, 8.0, 12.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[3.0, 7.0, 11.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert components["normalized_progress"][0] == 0.25
    assert math.isclose(components["r_progress"][0], 0.25 / 3.0, rel_tol=1e-6)
    assert math.isclose(reward[0], 0.25 / 3.0, rel_tol=1e-6)


def test_reset_rejects_zero_initial_active_target_distance():
    tracker = _tracker()

    with pytest.raises(ValueError, match="reference distance.*positive"):
        tracker.reset([0], [[0.0, 4.0]])


def test_switch_to_final_goal_rejects_zero_reference_distance():
    tracker = _tracker()
    tracker.reset([0], [[3.0, 5.0]])

    with pytest.raises(ValueError, match="reference distance.*positive"):
        tracker.compute_step(
            distances_to_goals=[[0.5, 0.0]],
            is_stop=np.array([False]),
            valid_mask=np.array([True]),
        )


def test_switch_to_next_subgoal_rejects_zero_reference_distance():
    tracker = _tracker()
    tracker.reset([0], [[3.0, 5.0, 7.0]])

    with pytest.raises(ValueError, match="reference distance.*positive"):
        tracker.compute_step(
            distances_to_goals=[[0.5, 0.0, 4.0]],
            is_stop=np.array([False]),
            valid_mask=np.array([True]),
        )


def test_progress_is_clipped_to_unit_scale():
    tracker = _tracker()
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[0.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert reward[0] == 1.0
    assert components["r_progress"][0] == 1.0
    assert components["r_subgoal_success"][0] == 0.0
    assert components["normalized_progress"][0] == 1.0


def test_negative_progress_produces_negative_progress_reward():
    tracker = _tracker(progress_reward_coef=1.0, stall_patience=3)
    tracker.reset([0], [[4.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[5.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(components["normalized_progress"][0], -0.25, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], -0.25, rel_tol=1e-6)
    assert components["r_penalty"][0] == 0.0
    assert math.isclose(reward[0], -0.25, rel_tol=1e-6)


def test_subgoal_success_bonus_is_normalized_by_subgoal_count():
    tracker = _tracker(subgoal_success_reward_coef=6.0)
    tracker.reset([0], [[3.0, 4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[0.4, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(components["r_subgoal_success"][0], 3.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 3.0, rel_tol=1e-6)
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert math.isclose(tracker.initial_distance_to_active_subgoal[0], 2.0, rel_tol=1e-6)
    assert math.isclose(components["normalized_progress"][0], 0.8666666667, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.2888888889, rel_tol=1e-6)
    assert math.isclose(reward[0], 3.2888888889, rel_tol=1e-6)


def test_switch_without_precision_bonus_permanently_loses_that_bonus():
    tracker = _tracker(subgoal_success_reward_coef=6.0)
    tracker.reset([0], [[3.0, 4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[0.75, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert tracker.cumulative_subgoal_success[0] == 0.0
    assert math.isclose(components["normalized_progress"][0], 0.75, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.25, rel_tol=1e-6)
    assert math.isclose(reward[0], 0.25, rel_tol=1e-6)

    _, second_components = tracker.compute_step(
        distances_to_goals=[[0.2, 0.4, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(second_components["r_subgoal_success"][0], 3.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 3.0, rel_tol=1e-6)


def test_stall_penalty_starts_after_patience_threshold():
    tracker = _tracker(stall_patience=2, stall_penalty_coeff=1.5)
    tracker.reset([0], [[4.0]])

    first_reward, first_components = tracker.compute_step(
        distances_to_goals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_goals=[[4.5]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_penalty"][0] == 0.0
    assert second_components["r_penalty"][0] == -1.5
    assert first_reward[0] == 0.0
    assert math.isclose(second_reward[0], -1.625, rel_tol=1e-6)


def test_stalled_state_requires_consecutive_positive_progress_to_recover():
    tracker = _tracker(
        progress_reward_coef=1.0,
        stall_patience=2,
        stall_penalty_coeff=1.5,
        stall_recovery_patience=2,
        stall_observation_patience=1,
    )
    tracker.reset([0], [[4.0]])

    _, first_components = tracker.compute_step(
        distances_to_goals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, second_components = tracker.compute_step(
        distances_to_goals=[[4.5]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, recovery_components = tracker.compute_step(
        distances_to_goals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, observation_components = tracker.compute_step(
        distances_to_goals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, stalled_components = tracker.compute_step(
        distances_to_goals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_penalty"][0] == 0.0
    assert second_components["r_penalty"][0] == -1.5
    assert recovery_components["r_penalty"][0] == 0.0
    assert observation_components["r_penalty"][0] == 0.0
    assert stalled_components["r_penalty"][0] == -1.5
    assert math.isclose(
        stalled_components["stall_penalty_rate"][0],
        2.0 / 5.0,
        rel_tol=1e-6,
    )


def test_subgoal_reward_tensorboard_diagnostics_track_episode_state():
    tracker = _tracker(stall_patience=2, stall_penalty_coeff=1.5)
    tracker.reset([0], [[4.0, 2.0, 3.0]])

    _, first_components = tracker.compute_step(
        distances_to_goals=[[4.0, 2.0, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, second_components = tracker.compute_step(
        distances_to_goals=[[4.2, 2.0, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, third_components = tracker.compute_step(
        distances_to_goals=[[0.4, 1.5, 2.5]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert third_components["num_goals"][0] == 3.0
    assert third_components["num_subgoals"][0] == 2.0
    assert third_components["num_intermediate_subgoals"][0] == 2.0
    assert math.isclose(
        third_components["subgoal_completion_ratio"][0],
        1.0 / 2.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        third_components["precision_subgoal_success_ratio"][0],
        0.5,
        rel_tol=1e-6,
    )
    assert math.isclose(
        third_components["stall_penalty_rate"][0],
        1.0 / 3.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        third_components["first_stall_penalty_rate"][0],
        1.0 / 3.0,
        rel_tol=1e-6,
    )
    assert math.isclose(
        third_components["mean_normalized_progress"][0],
        0.3,
        rel_tol=1e-6,
    )
    assert third_components["any_stall_penalty_subgoal"][0] == 1.0
    assert third_components["stall_then_goal_success"][0] == 0.0
    assert third_components["final_goal_success_ratio"][0] == 0.0
    assert third_components["premature_stop_ratio"][0] == 0.0
    assert third_components["stop_action_ratio"][0] == 0.0
    assert third_components["distance_to_final_goal"][0] == 2.5
    assert third_components["cumulative_progress"][0] == tracker.cumulative_progress[0]
    assert (
        third_components["cumulative_subgoal_success"][0]
        == tracker.cumulative_subgoal_success[0]
    )
    assert third_components["cumulative_penalty"][0] == tracker.cumulative_penalty[0]
    assert third_components["cumulative_stop"][0] == tracker.cumulative_stop[0]
    assert first_components["any_stall_penalty_subgoal"][0] == 0.0
    assert second_components["any_stall_penalty_subgoal"][0] == 1.0


def test_stall_then_goal_success_diagnostic_uses_stall_denominator():
    tracker = _tracker(stall_patience=1, stop_success_reward_coef=10.0)
    tracker.reset([0], [[2.0]])

    _, first_components = tracker.compute_step(
        distances_to_goals=[[2.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, second_components = tracker.compute_step(
        distances_to_goals=[[0.4]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    _, final_components = tracker.compute_step(
        distances_to_goals=[[0.75]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert first_components["any_stall_penalty_subgoal"][0] == 1.0
    assert first_components["stall_then_goal_success"][0] == 0.0
    assert second_components["any_stall_penalty_subgoal"][0] == 1.0
    assert second_components["stall_then_goal_success"][0] == 0.0
    assert final_components["any_stall_penalty_subgoal"][0] == 1.0
    assert final_components["stall_then_goal_success"][0] == 1.0
    assert final_components["final_goal_success_ratio"][0] == 1.0
    assert final_components["stop_action_ratio"][0] == 1.0
    assert final_components["premature_stop_ratio"][0] == 0.0


def test_premature_stop_is_penalized_before_all_subgoals_finish():
    tracker = _tracker(premature_stop_coeff=4.0)
    tracker.reset([0], [[3.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[2.5, 4.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert components["r_stop"][0] == -4.0
    assert math.isclose(components["normalized_progress"][0], 1.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 1.0 / 12.0, rel_tol=1e-6)
    assert math.isclose(reward[0], -3.9166666667, rel_tol=1e-6)
    assert tracker.active_subgoal_index.tolist() == [0]


def test_single_goal_stop_success_does_not_require_subgoal_switch():
    tracker = _tracker(stop_success_reward_coef=10.0, final_success_distance=3.0)
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[1.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert tracker.all_subgoals_finished.tolist() == [True]
    assert components["r_subgoal_success"][0] == 0.0
    assert math.isclose(components["r_progress"][0], 0.25, rel_tol=1e-6)
    assert math.isclose(components["r_stop"][0], 10.0, rel_tol=1e-6)
    assert components["premature_stop_ratio"][0] == 0.0
    assert math.isclose(reward[0], 10.25, rel_tol=1e-6)


def test_finalgoal_progress_continues_after_all_subgoals_finish():
    tracker = _tracker(progress_reward_coef=1.0)
    tracker.reset([0], [[3.0, 5.0]])

    _, first_components = tracker.compute_step(
        distances_to_goals=[[0.75, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_goals=[[0.5, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert tracker.all_subgoals_finished.tolist() == [True]
    assert first_components["r_subgoal_success"][0] == 0.0
    assert math.isclose(second_components["normalized_progress"][0], 0.25, rel_tol=1e-6)
    assert math.isclose(second_components["r_progress"][0], 0.125, rel_tol=1e-6)
    assert second_components["r_stop"][0] == 0.0
    assert math.isclose(second_reward[0], 0.125, rel_tol=1e-6)


def test_step_cost_coeff_breaks_ties_without_dominating_progress():
    tracker = _tracker(progress_reward_coef=1.0, step_cost_coeff=0.01)
    tracker.reset([0], [[4.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(components["r_step_cost"][0], -0.01, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.25, rel_tol=1e-6)
    assert math.isclose(reward[0], 0.24, rel_tol=1e-6)


def test_final_stop_failure_is_penalized_when_finalgoal_is_too_far():
    tracker = _tracker(
        failure_stop_coeff=5.0,
        final_success_distance=3.0,
        stop_success_reward_coef=10.0,
    )
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[3.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert tracker.all_subgoals_finished.tolist() == [True]
    assert components["r_stop"][0] == -5.0
    assert components["final_goal_success_ratio"][0] == 0.0
    assert components["premature_stop_ratio"][0] == 0.0
    assert math.isclose(reward[0], -5.75, rel_tol=1e-6)


def test_truncation_failure_is_penalized_without_stop():
    tracker = _tracker(
        failure_stop_coeff=5.0,
        final_success_distance=3.0,
        stop_success_reward_coef=10.0,
    )
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[3.5]],
        is_stop=np.array([False]),
        is_truncated=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert components["r_stop"][0] == -5.0
    assert components["final_goal_success_ratio"][0] == 0.0
    assert components["stop_action_ratio"][0] == 0.0
    assert math.isclose(reward[0], -5.75, rel_tol=1e-6)


def test_stop_on_same_step_as_last_subgoal_switch_can_succeed_at_finalgoal():
    tracker = _tracker(
        premature_stop_coeff=4.0,
        stop_success_reward_coef=10.0,
        final_success_distance=3.0,
    )
    tracker.reset([0], [[2.0, 1.5]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[0.4, 1.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert tracker.all_subgoals_finished.tolist() == [True]
    assert math.isclose(components["r_subgoal_success"][0], 6.0, rel_tol=1e-6)
    assert math.isclose(components["r_stop"][0], 10.0, rel_tol=1e-6)
    assert components["premature_stop_ratio"][0] == 0.0
    assert math.isclose(reward[0], 16.4, rel_tol=1e-6)


def test_invalid_mask_keeps_reward_and_state_unchanged():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_goals=[[0.2, 1.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([False]),
    )

    assert reward[0] == 0.0
    assert components["r_progress"][0] == 0.0
    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [0]
    assert tracker.completed_subgoal_count.tolist() == [0]
