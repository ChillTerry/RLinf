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

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
    convert_trajectories_to_batch,
)
from rlinf.models.embodiment.uninavid.grpo_diagnostics import (
    all_reduce_grpo_diagnostic_stats,
    all_reduce_keep_group_ratio,
    compute_action_ratios,
    compute_grpo_diagnostics,
    compute_keep_group_ratio,
    finalize_grpo_diagnostic_stats,
    reduce_success_to_trajectory_success,
)
from rlinf.models.embodiment.uninavid.nav_rollout import (
    FORWARD_ACTION_ID,
    LEFT_ACTION_ID,
    NO_OP_ACTION_ID,
    RIGHT_ACTION_ID,
    STOP_ACTION_ID,
)
from rlinf.utils.metric_utils import compute_evaluate_metrics, count_trajectories
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker


def _rollout_tensors():
    rewards = torch.tensor(
        [
            [[1.0], [2.0], [0.0], [0.0], [3.0], [1.0]],
            [[0.0], [2.0], [0.0], [0.0], [1.0], [3.0]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros((3, 6, 1), dtype=torch.bool)
    dones[-1] = True
    success = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
    return rewards, dones, success


def test_grpo_diagnostics_use_kept_groups_and_advantage_score_construction():
    rewards, dones, success = _rollout_tensors()
    keep_group_mask = torch.tensor([True, False, True])

    metrics = compute_grpo_diagnostics(
        rewards=rewards,
        dones=dones,
        trajectory_success=success,
        keep_group_mask=keep_group_mask,
        group_size=2,
        reward_type="step_level",
    )

    expected_score = torch.tensor([[1.0, 4.0], [0.0, 0.0], [4.0, 4.0]])
    kept_score = expected_score[keep_group_mask]
    score_std = kept_score.std(dim=1, unbiased=False)
    score_mean = kept_score.mean(dim=1)
    kept_trajectory_score = kept_score.reshape(-1)

    assert metrics["grpo/zero_sr_group_ratio"] == 0.0
    assert metrics["grpo/group_sr_mean"] == 0.75
    assert metrics["grpo/group_sr_max"] == 1.0
    assert metrics["grpo/group_sr_min"] == 0.5
    assert metrics["grpo/zero_score_std_group_ratio"] == 0.5
    assert math.isclose(
        metrics["grpo/trajectory_score_std"],
        kept_trajectory_score.std(unbiased=False).item(),
        rel_tol=1e-6,
    )
    assert metrics["grpo/trajectory_score_mean"] == kept_trajectory_score.mean().item()
    assert metrics["grpo/group_score_std"] == score_std.mean().item()
    assert metrics["grpo/group_score_mean"] == score_mean.mean().item()


def test_grpo_diagnostics_empty_kept_group_set_returns_zero_without_nan():
    rewards, dones, success = _rollout_tensors()
    keep_group_mask = torch.tensor([False, False, False])

    metrics = compute_grpo_diagnostics(
        rewards=rewards,
        dones=dones,
        trajectory_success=success,
        keep_group_mask=keep_group_mask,
        group_size=2,
        reward_type="step_level",
    )

    for value in metrics.values():
        assert value == 0.0
        assert not math.isnan(value)


def test_grpo_diagnostics_rejects_non_trajectory_success_shape():
    rewards, dones, _success = _rollout_tensors()

    try:
        compute_grpo_diagnostics(
            rewards=rewards,
            dones=dones,
            trajectory_success=torch.ones((2, 6), dtype=torch.float32),
            keep_group_mask=torch.tensor([True, True, True]),
            group_size=2,
            reward_type="step_level",
        )
    except ValueError as exc:
        assert "trajectory_success" in str(exc)
    else:
        raise AssertionError("Expected non-trajectory success shape to fail.")


def test_grpo_diagnostic_stats_finalize_uses_weighted_counts():
    metrics = finalize_grpo_diagnostic_stats(
        {
            "kept_group_count": 3.0,
            "zero_sr_group_count": 1.0,
            "group_sr_sum": 2.0,
            "group_sr_max": 1.0,
            "group_sr_min": 0.0,
            "zero_score_std_group_count": 1.0,
            "trajectory_score_count": 6.0,
            "trajectory_score_sum": 15.0,
            "trajectory_score_sq_sum": 55.0,
            "group_score_std_sum": 4.5,
            "group_score_mean_sum": 7.5,
        }
    )

    assert metrics["grpo/zero_sr_group_ratio"] == 1.0 / 3.0
    assert metrics["grpo/group_sr_mean"] == 2.0 / 3.0
    assert metrics["grpo/group_score_std"] == 1.5
    assert metrics["grpo/group_score_mean"] == 2.5
    assert metrics["grpo/trajectory_score_mean"] == 2.5


def test_reduce_success_to_trajectory_success_uses_max_over_rollout_steps():
    success = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    trajectory_success = reduce_success_to_trajectory_success(success, batch_size=4)

    torch.testing.assert_close(
        trajectory_success,
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
    )


def test_keep_group_ratio_uses_all_groups_denominator():
    keep_group_mask = torch.tensor([True, False, False, False])

    assert compute_keep_group_ratio(keep_group_mask) == 0.25


def test_action_ratios_count_stop_and_no_op_actions():
    actions = torch.tensor(
        [
            [[STOP_ACTION_ID], [NO_OP_ACTION_ID], [1]],
            [[NO_OP_ACTION_ID], [STOP_ACTION_ID], [2]],
        ],
        dtype=torch.long,
    )

    metrics = compute_action_ratios(actions)

    assert math.isclose(metrics["action/stop_ratio"], 2.0 / 6.0, rel_tol=1e-6)
    assert math.isclose(metrics["action/no_op_ratio"], 2.0 / 6.0, rel_tol=1e-6)


def test_action_ratios_empty_input_returns_zero_without_nan():
    metrics = compute_action_ratios(torch.empty((0,), dtype=torch.long))

    for value in metrics.values():
        assert value == 0.0
        assert not math.isnan(value)


def test_all_reduce_keep_group_ratio_uses_accelerator_device_without_dist(monkeypatch):
    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.grpo_diagnostics.Worker.torch_platform.current_device",
        lambda: torch.device("cpu"),
    )

    assert math.isclose(
        all_reduce_keep_group_ratio(torch.tensor([True, False, True])),
        2.0 / 3.0,
        rel_tol=1e-6,
    )


def test_all_reduce_grpo_stats_moves_tensors_to_accelerator_device_without_dist(
    monkeypatch,
):
    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.grpo_diagnostics.Worker.torch_platform.current_device",
        lambda: torch.device("cpu"),
    )
    stats = {
        "kept_group_count": torch.tensor(1.0),
        "zero_sr_group_count": torch.tensor(0.0),
        "group_sr_sum": torch.tensor(1.0),
        "group_sr_max": torch.tensor(1.0),
        "group_sr_min": torch.tensor(1.0),
        "zero_score_std_group_count": torch.tensor(0.0),
        "trajectory_score_count": torch.tensor(2.0),
        "trajectory_score_sum": torch.tensor(3.0),
        "trajectory_score_sq_sum": torch.tensor(5.0),
        "group_score_std_sum": torch.tensor(0.5),
        "group_score_mean_sum": torch.tensor(1.5),
    }

    reduced = all_reduce_grpo_diagnostic_stats(stats)

    assert set(reduced) == set(stats)
    assert all(value.device.type == "cpu" for value in reduced.values())


def test_uninavid_habitat_grpo_success_env_info_reaches_actor_batch():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"train": {"env_type": "habitat", "auto_reset": False}},
            "algorithm": {"adv_type": "grpo", "filter_rewards": True},
        }
    )
    rollout_result = EmbodiedRolloutResult(max_episode_length=2)
    env_info = {"success": torch.tensor([1.0, 0.0])}

    worker._append_uninavid_habitat_grpo_env_info(
        rollout_result,
        env_info,
        torch.zeros((2, 1), dtype=torch.bool),
    )
    trajectory = rollout_result.to_trajectory()
    batch = convert_trajectories_to_batch([trajectory])

    assert set(batch["env_info"]) == {"success"}
    torch.testing.assert_close(
        batch["env_info"]["success"],
        torch.tensor([[1.0, 0.0]]),
    )


def test_uninavid_action_forward_input_reaches_actor_actions_batch():
    rollout_result = EmbodiedRolloutResult(max_episode_length=2)
    rollout_result.append_step_result(
        ChunkStepResult(
            actions=torch.tensor([[STOP_ACTION_ID, NO_OP_ACTION_ID]], dtype=torch.long),
            forward_inputs={
                "action": torch.tensor(
                    [[STOP_ACTION_ID, NO_OP_ACTION_ID]], dtype=torch.long
                )
            },
        )
    )

    trajectory = rollout_result.to_trajectory()
    batch = convert_trajectories_to_batch([trajectory])

    torch.testing.assert_close(
        batch["actions"],
        torch.tensor([[[STOP_ACTION_ID, NO_OP_ACTION_ID]]], dtype=torch.long),
    )


def test_env_info_batch_conversion_scans_all_trajectories():
    first = EmbodiedRolloutResult(max_episode_length=1).to_trajectory()
    second_rollout = EmbodiedRolloutResult(max_episode_length=1)
    second_rollout.append_env_info({"success": torch.tensor([1.0])})
    second = second_rollout.to_trajectory()

    batch = convert_trajectories_to_batch([first, second])

    assert set(batch["env_info"]) == {"success"}
    torch.testing.assert_close(batch["env_info"]["success"], torch.tensor([[1.0]]))


def test_uninavid_habitat_grpo_env_info_is_not_sent_without_reward_filter():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"train": {"env_type": "habitat"}},
            "algorithm": {"adv_type": "grpo", "filter_rewards": False},
        }
    )
    rollout_result = EmbodiedRolloutResult(max_episode_length=2)

    worker._append_uninavid_habitat_grpo_env_info(
        rollout_result,
        {},
        torch.zeros((2, 1), dtype=torch.bool),
    )

    assert rollout_result.env_info == []


def test_uninavid_habitat_grpo_auto_reset_success_is_scattered_to_full_batch():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"train": {"env_type": "habitat", "auto_reset": True}},
            "algorithm": {"adv_type": "grpo", "filter_rewards": True},
        }
    )
    rollout_result = EmbodiedRolloutResult(max_episode_length=2)

    worker._append_uninavid_habitat_grpo_env_info(
        rollout_result,
        {"success": torch.tensor([1.0, 0.0])},
        torch.tensor([[False], [True], [False], [True]]),
    )

    trajectory = rollout_result.to_trajectory()
    batch = convert_trajectories_to_batch([trajectory])

    torch.testing.assert_close(
        batch["env_info"]["success"],
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    )


def test_actor_reward_filter_keep_ratio_is_computed_at_mask_creation_site():
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {
                "train": {
                    "env_type": "habitat",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {
                "adv_type": "grpo",
                "rollout_epoch": 1,
                "reward_type": "step_level",
                "filter_rewards": True,
                "group_size": 2,
                "rewards_lower_bound": 1.5,
                "rewards_upper_bound": 10.0,
            },
        }
    )
    actor._uninavid_habitat_grpo_keep_group_mask = None
    actor._uninavid_habitat_grpo_keep_group_ratio = None
    rollout_batch = {
        "rewards": torch.tensor(
            [
                [[1.0], [0.0], [2.0], [2.0], [0.0], [0.0]],
                [[1.0], [0.0], [2.0], [2.0], [0.0], [0.0]],
            ],
            dtype=torch.float32,
        ),
        "dones": torch.zeros((3, 6, 1), dtype=torch.bool),
    }

    processed = actor._process_received_rollout_batch(rollout_batch)

    assert actor._uninavid_habitat_grpo_keep_group_mask.tolist() == [
        False,
        True,
        False,
    ]
    assert math.isclose(
        actor._uninavid_habitat_grpo_keep_group_ratio,
        1.0 / 3.0,
        rel_tol=1e-6,
    )
    assert processed["loss_mask"][:, :, 0].tolist() == [
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
    ]


def test_actor_grpo_diagnostics_are_skipped_when_reward_filter_is_disabled(
    monkeypatch,
):
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {
                "train": {
                    "env_type": "habitat",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {
                "adv_type": "grpo",
                "rollout_epoch": 1,
                "reward_type": "step_level",
                "filter_rewards": False,
                "group_size": 2,
            },
        }
    )
    actor.rollout_batch = {
        "rewards": torch.zeros((1, 2, 1), dtype=torch.float32),
        "dones": torch.ones((2, 2, 1), dtype=torch.bool),
        "loss_mask": torch.ones((1, 2, 1), dtype=torch.bool),
        "env_info": {"success": torch.ones((1, 2), dtype=torch.float32)},
    }
    actor._uninavid_habitat_grpo_keep_group_mask = None
    actor._uninavid_habitat_grpo_keep_group_ratio = None

    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.calculate_adv_and_returns",
        lambda **kwargs: {"advantages": torch.zeros((1, 2, 1), dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.compute_rollout_metrics",
        lambda rollout_batch: {"rewards": 0.0},
    )

    metrics = actor.compute_advantages_and_returns()

    assert metrics == {"rewards": 0.0}
    assert "env_info" in actor.rollout_batch


def test_actor_grpo_diagnostics_enabled_path_logs_metrics_and_removes_env_info(
    monkeypatch,
):
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {
                "train": {
                    "env_type": "habitat",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {
                "adv_type": "grpo",
                "rollout_epoch": 1,
                "reward_type": "step_level",
                "filter_rewards": True,
                "group_size": 2,
            },
        }
    )
    actor.rollout_batch = {
        "rewards": torch.tensor([[[1.0], [2.0]], [[0.0], [2.0]]]),
        "dones": torch.ones((3, 2, 1), dtype=torch.bool),
        "loss_mask": torch.ones((2, 2, 1), dtype=torch.bool),
        "env_info": {"success": torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
    }
    actor._uninavid_habitat_grpo_keep_group_mask = torch.tensor([True])
    actor._uninavid_habitat_grpo_keep_group_ratio = 1.0

    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.calculate_adv_and_returns",
        lambda **kwargs: {"advantages": torch.zeros((2, 2, 1), dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.compute_rollout_metrics",
        lambda rollout_batch: {"rewards": 0.0},
    )

    metrics = actor.compute_advantages_and_returns()

    assert metrics["reward_filter/keep_group_ratio"] == 1.0
    assert metrics["grpo/group_sr_mean"] == 1.0
    assert "grpo/trajectory_score_mean" in metrics
    assert "env_info" not in actor.rollout_batch


def test_actor_action_ratios_use_reward_filter_kept_groups_only(monkeypatch):
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {
                "train": {
                    "env_type": "habitat",
                    "auto_reset": False,
                    "ignore_terminations": False,
                }
            },
            "algorithm": {
                "adv_type": "grpo",
                "rollout_epoch": 1,
                "reward_type": "step_level",
                "filter_rewards": True,
                "group_size": 2,
            },
        }
    )
    actor.rollout_batch = {
        "actions": torch.tensor(
            [
                [
                    [STOP_ACTION_ID],
                    [STOP_ACTION_ID],
                    [NO_OP_ACTION_ID],
                    [1],
                    [2],
                    [2],
                ],
                [
                    [STOP_ACTION_ID],
                    [NO_OP_ACTION_ID],
                    [NO_OP_ACTION_ID],
                    [STOP_ACTION_ID],
                    [3],
                    [3],
                ],
            ],
            dtype=torch.long,
        ),
        "rewards": torch.zeros((2, 6, 1), dtype=torch.float32),
        "dones": torch.ones((3, 6, 1), dtype=torch.bool),
        "loss_mask": torch.ones((2, 6, 1), dtype=torch.bool),
        "env_info": {"success": torch.ones((1, 6), dtype=torch.float32)},
    }
    actor._uninavid_habitat_grpo_keep_group_mask = torch.tensor([False, True, False])
    actor._uninavid_habitat_grpo_keep_group_ratio = 1.0 / 3.0

    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.calculate_adv_and_returns",
        lambda **kwargs: {"advantages": torch.zeros((2, 6, 1), dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.compute_rollout_metrics",
        lambda rollout_batch: {},
    )
    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.grpo_diagnostics.Worker.torch_platform.current_device",
        lambda: torch.device("cpu"),
    )

    metrics = actor.compute_advantages_and_returns()

    assert metrics["action/stop_ratio"] == 1.0 / 4.0
    assert metrics["action/no_op_ratio"] == 2.0 / 4.0


def test_eval_action_metrics_use_all_eval_actions_for_uninavid_habitat():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"eval": {"env_type": "habitat"}},
        }
    )
    raw_chunk_actions = torch.tensor(
        [
            [[STOP_ACTION_ID], [NO_OP_ACTION_ID]],
            [[1], [NO_OP_ACTION_ID]],
        ],
        dtype=torch.long,
    )

    metrics = worker._compute_eval_action_metrics(raw_chunk_actions)

    torch.testing.assert_close(
        metrics["action/stop_ratio"],
        torch.tensor([0.25]),
    )
    torch.testing.assert_close(
        metrics["action/no_op_ratio"],
        torch.tensor([0.5]),
    )


def test_eval_action_metrics_are_gated_to_uninavid_habitat():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "openvla"}},
            "env": {"eval": {"env_type": "habitat"}},
        }
    )

    assert (
        worker._compute_eval_action_metrics(
            torch.tensor([[STOP_ACTION_ID]], dtype=torch.long)
        )
        == {}
    )


def test_eval_action_metrics_empty_input_returns_empty_metric_tensors():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"eval": {"env_type": "habitat"}},
        }
    )

    metrics = worker._compute_eval_action_metrics(torch.empty((0,), dtype=torch.long))

    assert metrics["action/stop_ratio"].numel() == 0
    assert metrics["action/no_op_ratio"].numel() == 0


def test_eval_episode_action_metrics_track_first_stop_and_valid_moves():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"eval": {"env_type": "habitat", "max_episode_steps": 5}},
        }
    )
    worker.eval_num_envs_per_stage = 3

    worker._reset_eval_episode_action_metrics(stage_id=0)
    worker._update_eval_episode_action_metrics(
        torch.tensor(
            [
                [[FORWARD_ACTION_ID], [LEFT_ACTION_ID]],
                [[FORWARD_ACTION_ID], [STOP_ACTION_ID]],
                [[NO_OP_ACTION_ID], [RIGHT_ACTION_ID]],
            ],
            dtype=torch.long,
        ),
        stage_id=0,
    )
    worker._update_eval_episode_action_metrics(
        torch.tensor(
            [
                [[RIGHT_ACTION_ID], [STOP_ACTION_ID]],
                [[FORWARD_ACTION_ID], [LEFT_ACTION_ID]],
                [[FORWARD_ACTION_ID], [LEFT_ACTION_ID]],
            ],
            dtype=torch.long,
        ),
        stage_id=0,
    )

    metrics = worker._collect_eval_episode_action_metrics(
        stage_id=0,
        done_mask=torch.tensor([True, True, False]),
    )

    torch.testing.assert_close(
        metrics["action/first_stop_step"],
        torch.tensor([4.0, 2.0]),
    )
    torch.testing.assert_close(
        metrics["action/valid_move_ratio"],
        torch.tensor([0.75, 0.5]),
    )


def test_eval_episode_action_metrics_use_timeout_sentinel_and_reset_done_envs():
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"eval": {"env_type": "habitat", "max_episode_steps": 3}},
        }
    )
    worker.eval_num_envs_per_stage = 1

    worker._reset_eval_episode_action_metrics(stage_id=0)
    worker._update_eval_episode_action_metrics(
        torch.tensor([[[FORWARD_ACTION_ID], [NO_OP_ACTION_ID]]], dtype=torch.long),
        stage_id=0,
    )
    worker._update_eval_episode_action_metrics(
        torch.tensor([[[LEFT_ACTION_ID], [RIGHT_ACTION_ID]]], dtype=torch.long),
        stage_id=0,
    )

    metrics = worker._collect_eval_episode_action_metrics(
        stage_id=0,
        done_mask=torch.tensor([True]),
    )

    torch.testing.assert_close(
        metrics["action/first_stop_step"],
        torch.tensor([4.0]),
    )
    torch.testing.assert_close(
        metrics["action/valid_move_ratio"],
        torch.tensor([2.0 / 3.0]),
    )

    worker._update_eval_episode_action_metrics(
        torch.tensor([[[STOP_ACTION_ID], [NO_OP_ACTION_ID]]], dtype=torch.long),
        stage_id=0,
    )
    metrics = worker._collect_eval_episode_action_metrics(
        stage_id=0,
        done_mask=torch.tensor([True]),
    )

    torch.testing.assert_close(
        metrics["action/first_stop_step"],
        torch.tensor([1.0]),
    )
    torch.testing.assert_close(
        metrics["action/valid_move_ratio"],
        torch.tensor([0.0]),
    )


def test_eval_action_metrics_do_not_drive_trajectory_count():
    metrics = {
        "action/stop_ratio": torch.tensor([0.25, 0.50]),
        "success": torch.tensor([1.0, 0.0, 1.0]),
    }

    assert count_trajectories(metrics) == 3


def test_evaluate_metrics_keep_action_mean_without_counting_as_trajectories():
    reduced = compute_evaluate_metrics(
        [
            {
                "action/stop_ratio": torch.tensor([0.25, 0.50]),
                "success": torch.tensor([1.0, 0.0, 1.0]),
            }
        ]
    )

    assert reduced["num_trajectories"] == 3
    assert math.isclose(float(reduced["action/stop_ratio"]), 0.375, rel_tol=1e-6)
