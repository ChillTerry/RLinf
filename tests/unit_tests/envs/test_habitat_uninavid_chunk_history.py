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

from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import rlinf.envs.habitat.habitat_env as habitat_env_module
from rlinf.envs.habitat.habitat_env import HabitatEnv


def test_uninavid_habitat_extension_config_composes_with_local_schema():
    from habitat_baselines.config.default import get_config

    cfg = get_config(
        "rlinf/envs/habitat/extensions/config/vlnce_r2r_uninavid.yaml",
        overrides=[
            "habitat.dataset.split=val_unseen",
            "habitat.dataset.data_path=/data/vln-dependency/VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz",
            "habitat.dataset.scenes_dir=/data/vln-dependency/VLN-CE/scene_dataset/",
        ],
    )

    assert cfg.habitat.task.type == "VLN-v0"
    assert cfg.habitat.simulator.forward_step_size == 0.25
    assert cfg.habitat.simulator.turn_angle == 30
    assert cfg.habitat.task.actions.no_op.type == "NoOpAction"
    assert cfg.habitat.task.measurements.trajectory_Length.type == "TrajectoryLength"
    assert (
        cfg.habitat.task.measurements.oracle_navigation_error.type
        == "OracleNavigationError"
    )
    assert cfg.habitat.task.measurements.ndtw.type == "NDTW"
    assert cfg.habitat.task.measurements.ndtw.SPLIT == "val_unseen"
    assert (
        cfg.habitat.task.measurements.ndtw.GT_PATH
        == "VLN-CE/datasets/r2r/{split}/{split}_gt.json.gz"
    )
    assert cfg.habitat.task.measurements.ndtw.SUCCESS_DISTANCE == 3.0
    assert cfg.habitat.task.measurements.ndtw.FDTW is False


def test_habitat_r2r_env_default_exposes_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/env/habitat_r2r.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in cfg
    assert raw_cfg["reward_mode"] == "weighted_success_ndtw"
    assert cfg.success_reward_coef == 10.0
    assert cfg.ndtw_reward_coef == 5.0
    assert cfg.ndtw_gt_path is None
    assert cfg.use_rel_reward is False


def test_habitat_grpo_uninavid_uses_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg["algorithm"]
    assert raw_cfg["algorithm"]["success_reward_coef"] == 10.0
    assert raw_cfg["algorithm"]["ndtw_reward_coef"] == 5.0
    assert raw_cfg["env"]["train"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["train"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert raw_cfg["env"]["train"]["use_rel_reward"] is False
    assert (
        raw_cfg["env"]["train"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert raw_cfg["env"]["eval"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["eval"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert raw_cfg["env"]["eval"]["use_rel_reward"] is False
    assert (
        raw_cfg["env"]["eval"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_gt.json.gz"
    )


def test_habitat_eval_uninavid_uses_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_eval_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg["algorithm"]
    assert raw_cfg["algorithm"]["success_reward_coef"] == 10.0
    assert raw_cfg["algorithm"]["ndtw_reward_coef"] == 5.0
    assert raw_cfg["env"]["train"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["train"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert raw_cfg["env"]["train"]["split"] == "train"
    assert (
        raw_cfg["env"]["train"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert raw_cfg["env"]["eval"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["eval"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert raw_cfg["env"]["eval"]["use_rel_reward"] is False
    assert (
        raw_cfg["env"]["eval"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_gt.json.gz"
    )


def test_habitat_env_fn_params_override_ndtw_config(monkeypatch):
    dummy_dataset = SimpleNamespace(
        episodes=[
            SimpleNamespace(episode_id="7"),
            SimpleNamespace(episode_id="19"),
        ]
    )
    dummy_config = SimpleNamespace(
        habitat=SimpleNamespace(dataset=SimpleNamespace(type="DummyDataset"))
    )

    monkeypatch.setattr(habitat_env_module, "get_config", lambda *args, **kwargs: dummy_config)
    monkeypatch.setattr(
        habitat_env_module.habitat.datasets,
        "make_dataset",
        lambda *args, **kwargs: dummy_dataset,
    )
    monkeypatch.setattr(
        habitat_env_module,
        "vram_balance_episode_ids",
        lambda *args, **kwargs: [["7", "19"]],
    )

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        init_params=SimpleNamespace(config_path="dummy-config.yaml"),
        split="train",
        data_path="/tmp/r2r/train/train.json.gz",
        ndtw_gt_path="/tmp/r2r/train/train_gt.json.gz",
        scenes_dir="/tmp/scenes",
        seed=42,
        total_num_envs=4,
        max_steps_per_rollout_epoch=12800,
    )
    env.auto_reset = True
    env.total_num_processes = 1
    env.num_group = 1
    env.seed_offset = 0
    env.seed = 42
    env.max_episode_steps = 512
    env.num_envs = 1
    env.group_size = 1
    env._sample_habitat_dataset_scenes = lambda *args, **kwargs: None

    env_fn_params = env._get_env_fn_params()

    assert "habitat.task.measurements.ndtw.SPLIT=train" in env_fn_params[0]["overrides"]
    assert (
        "habitat.task.measurements.ndtw.GT_PATH=/tmp/r2r/train/train_gt.json.gz"
        in env_fn_params[0]["overrides"]
    )


def test_habitat_weighted_reward_validation_requires_required_fields():
    env = object.__new__(HabitatEnv)
    env.reward_mode = "weighted_success_ndtw"
    env.cfg = SimpleNamespace(success_reward_coef=10.0, ndtw_reward_coef=5.0)

    with pytest.raises(ValueError, match="ndtw_gt_path"):
        env._validate_reward_config()


def test_habitat_weighted_reward_validation_rejects_unsupported_mode():
    env = object.__new__(HabitatEnv)
    env.reward_mode = None
    env.cfg = SimpleNamespace()

    with pytest.raises(ValueError, match="reward_mode.*weighted_success_ndtw"):
        env._validate_reward_config()


def _make_reward_test_env(num_envs):
    env = object.__new__(HabitatEnv)
    env.num_envs = num_envs
    env.cfg = SimpleNamespace(success_reward_coef=10.0, ndtw_reward_coef=5.0)
    env.env_config = SimpleNamespace(
        task=SimpleNamespace(
            measurements=SimpleNamespace(success=SimpleNamespace(success_distance=3.0))
        )
    )
    return env


def test_habitat_reward_uses_weighted_success_ndtw_only_for_first_normal_terminal():
    env = _make_reward_test_env(num_envs=4)
    env.dones_once = np.array([False, False, False, True])
    episode = {
        "success": torch.tensor([1.0, 0.0, 1.0, 1.0]),
        "distance_to_goal": torch.tensor([1.5, 4.0, 0.0, 0.0]),
        "ndtw": torch.tensor([0.2, 0.8, 1.0, 0.9]),
    }

    reward = env._calc_step_reward(
        episode,
        terminations=np.array([True, True, False, True]),
        truncations=np.array([False, False, True, False]),
    )

    assert reward.tolist() == [6.0, 4.0, 0.0, 0.0]


def test_habitat_reward_is_zero_for_non_terminal_steps():
    env = _make_reward_test_env(num_envs=2)
    env.dones_once = np.array([False, False])
    episode = {
        "success": torch.tensor([1.0, 0.0]),
        "distance_to_goal": torch.tensor([0.0, 1.0]),
        "ndtw": torch.tensor([1.0, 0.5]),
    }

    reward = env._calc_step_reward(
        episode,
        terminations=np.array([False, False]),
        truncations=np.array([False, False]),
    )

    assert reward.tolist() == [0.0, 0.0]


def test_habitat_record_metrics_includes_ndtw_and_seeds_initial_distance_to_goal():
    env = object.__new__(HabitatEnv)
    env.env_config = SimpleNamespace(
        task=SimpleNamespace(
            measurements=SimpleNamespace(success=SimpleNamespace(success_distance=4.0))
        )
    )
    env._elapsed_steps = np.array([1, 2], dtype=np.int32)
    env.initial_distance_to_goal = np.array([np.nan, 5.0], dtype=np.float32)

    infos = {
        "distance_to_goal": [3.5, 3.0],
        "ndtw": [0.25, 0.75],
        "trajectory_Length": [2.0, 2.0],
        "oracle_success": [0.0, 1.0],
        "oracle_navigation_error": [3.5, 0.5],
    }

    recorded_infos = env._record_metrics(infos, np.array([False, True]))

    assert env.initial_distance_to_goal.tolist() == [3.5, 5.0]
    assert recorded_infos["episode"]["success"].tolist() == [0.0, 1.0]
    assert recorded_infos["episode"]["spl"].tolist() == [0.0, 1.0]
    assert recorded_infos["episode"]["ndtw"].tolist() == [0.25, 0.75]
