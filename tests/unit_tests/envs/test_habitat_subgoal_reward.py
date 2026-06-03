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

from rlinf.envs.habitat.subgoal_reward import (
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
        stall_patience=kwargs.get("stall_patience", 3),
        stall_penalty_coeff=kwargs.get("stall_penalty_coeff", 1.0),
    )
    return SubgoalRewardTracker(num_envs=num_envs, config=config)


def test_progress_is_normalized_by_active_subgoal_initial_distance():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 8.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[3.0, 7.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(reward[0], 0.25, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.25, rel_tol=1e-6)
    assert components["normalized_progress"][0] == 0.25
    assert tracker.active_subgoal_index.tolist() == [0]


def test_progress_is_clipped_to_unit_scale():
    tracker = _tracker()
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert reward[0] == 7.0
    assert components["r_progress"][0] == 1.0
    assert components["r_subgoal_success"][0] == 6.0
    assert components["normalized_progress"][0] == 1.0


def test_negative_progress_produces_negative_progress_reward():
    tracker = _tracker(progress_reward_coef=1.0, stall_patience=3)
    tracker.reset([0], [[4.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[5.0]],
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
        distances_to_subgoals=[[0.4, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(components["r_subgoal_success"][0], 2.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 2.0, rel_tol=1e-6)
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert math.isclose(tracker.initial_distance_to_active_subgoal[0], 2.0, rel_tol=1e-6)
    assert math.isclose(reward[0], 2.8666666667, rel_tol=1e-6)


def test_switch_without_precision_bonus_permanently_loses_that_bonus():
    tracker = _tracker(subgoal_success_reward_coef=6.0)
    tracker.reset([0], [[3.0, 4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.75, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert tracker.cumulative_subgoal_success[0] == 0.0
    assert math.isclose(reward[0], 0.75, rel_tol=1e-6)

    _, second_components = tracker.compute_step(
        distances_to_subgoals=[[0.2, 0.4, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(second_components["r_subgoal_success"][0], 2.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 2.0, rel_tol=1e-6)


def test_stall_penalty_starts_after_patience_threshold():
    tracker = _tracker(stall_patience=2, stall_penalty_coeff=1.5)
    tracker.reset([0], [[4.0]])

    first_reward, first_components = tracker.compute_step(
        distances_to_subgoals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_subgoals=[[4.5]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_penalty"][0] == 0.0
    assert second_components["r_penalty"][0] == -1.5
    assert first_reward[0] == 0.0
    assert math.isclose(second_reward[0], -1.625, rel_tol=1e-6)


def test_premature_stop_is_penalized_before_all_subgoals_finish():
    tracker = _tracker(premature_stop_coeff=4.0)
    tracker.reset([0], [[3.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[2.5, 4.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert components["r_stop"][0] == -4.0
    assert math.isclose(reward[0], -3.8333333333, rel_tol=1e-6)
    assert tracker.active_subgoal_index.tolist() == [0]


def test_final_stop_success_is_scaled_by_final_distance_after_all_subgoals_finish():
    tracker = _tracker(stop_success_reward_coef=10.0, final_success_distance=3.0)
    tracker.reset([0], [[2.0]])

    first_reward, first_components = tracker.compute_step(
        distances_to_subgoals=[[0.4]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_subgoals=[[0.75]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_subgoal_success"][0] == 6.0
    assert tracker.all_subgoals_finished.tolist() == [True]
    assert math.isclose(second_components["r_stop"][0], 7.5, rel_tol=1e-6)
    assert second_reward[0] == 7.5


def test_invalid_mask_keeps_reward_and_state_unchanged():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.2, 1.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([False]),
    )

    assert reward[0] == 0.0
    assert components["r_progress"][0] == 0.0
    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [0]
    assert tracker.completed_subgoal_count.tolist() == [0]
