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


def test_habitat_r2r_env_default_uses_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/env/habitat_r2r.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg
    assert "reward_mode" not in raw_cfg
    assert cfg.success_reward_coef == 10.0
    assert cfg.ndtw_reward_coef == 5.0
    assert cfg.ndtw_gt_path is None


def test_habitat_grpo_uninavid_uses_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg["algorithm"]
    assert raw_cfg["algorithm"]["success_reward_coef"] == 10.0
    assert raw_cfg["algorithm"]["ndtw_reward_coef"] == 5.0
    assert "reward_mode" not in raw_cfg["env"]["train"]
    assert raw_cfg["env"]["train"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["train"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert (
        raw_cfg["env"]["train"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert "reward_mode" not in raw_cfg["env"]["eval"]
    assert raw_cfg["env"]["eval"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["eval"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
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
    assert "reward_mode" not in raw_cfg["env"]["train"]
    assert raw_cfg["env"]["train"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["train"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert raw_cfg["env"]["train"]["split"] == "train"
    assert (
        raw_cfg["env"]["train"]["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert "reward_mode" not in raw_cfg["env"]["eval"]
    assert raw_cfg["env"]["eval"]["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_cfg["env"]["eval"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
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
        "vram_balance_episode_sequences",
        lambda *args, **kwargs: [[["7", "19"]]],
    )

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        init_params=SimpleNamespace(config_path="dummy-config.yaml"),
        split="train",
        data_path="/tmp/r2r/train/train.json.gz",
        ndtw_gt_path="/tmp/r2r/train/train_gt.json.gz",
        scenes_dir="/tmp/scenes",
        seed=42,
        auto_reset=True,
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


def _make_chunk_history_test_env(model_type="uninavid"):
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type=model_type)
    env.num_envs = 2
    return env


def test_uninavid_reset_attaches_single_rgb_frame_history():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        video_cfg=SimpleNamespace(save_video=False),
    )
    env.num_envs = 2
    env._elapsed_steps = np.ones(2, dtype=np.int32)
    env.first_done_cached_mask = np.ones(2, dtype=bool)
    env.initial_distance_to_goal = np.full(2, np.nan, dtype=np.float32)
    env.episode_info = None
    env.current_raw_obs = None

    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    raw_obs = [
        {"instruction": {"text": "go left", "tokens": [1]}, "rgb": rgb},
        {"instruction": {"text": "go right", "tokens": [2]}, "rgb": rgb + 1},
    ]
    env.env = SimpleNamespace(
        reset=lambda env_idx: [raw_obs[int(idx)] for idx in env_idx],
        get_current_metrics=lambda env_idx: {},
        get_current_episode_metadata=lambda: {"episode_id": ["10", "11"]},
    )

    obs, _ = env.reset()

    assert obs["wrist_images"].shape == (2, 2, 2, 3)
    assert obs["rgb_frame_history"].shape == (2, 1, 2, 2, 3)
    assert obs["rgb_frame_history_lengths"].tolist() == [1, 1]
    assert torch.equal(obs["rgb_frame_history"][:, 0], obs["wrist_images"])


def test_attach_rgb_frame_history_keeps_wrist_images_single_frame():
    env = _make_chunk_history_test_env()
    obs_list = []
    for step_id in range(3):
        obs_list.append(
            {
                "wrist_images": torch.full(
                    (2, 2, 2, 3),
                    fill_value=step_id,
                    dtype=torch.uint8,
                )
            }
        )

    env._attach_rgb_frame_history(obs_list)

    final_obs = obs_list[-1]
    assert final_obs["wrist_images"].shape == (2, 2, 2, 3)
    assert final_obs["rgb_frame_history"].shape == (2, 3, 2, 2, 3)
    assert final_obs["rgb_frame_history_lengths"].tolist() == [3, 3]
    assert torch.equal(final_obs["rgb_frame_history"][:, 0], obs_list[0]["wrist_images"])
    assert torch.equal(final_obs["rgb_frame_history"][:, 1], obs_list[1]["wrist_images"])
    assert torch.equal(final_obs["rgb_frame_history"][:, 2], obs_list[2]["wrist_images"])


def test_attach_rgb_frame_history_is_uninavid_only():
    env = _make_chunk_history_test_env(model_type="cma")
    obs_list = [
        {"wrist_images": torch.zeros((2, 2, 2, 3), dtype=torch.uint8)},
        {"wrist_images": torch.ones((2, 2, 2, 3), dtype=torch.uint8)},
    ]

    env._attach_rgb_frame_history(obs_list)

    assert "rgb_frame_history" not in obs_list[-1]
    assert "rgb_frame_history_lengths" not in obs_list[-1]
    assert obs_list[-1]["wrist_images"].shape == (2, 2, 2, 3)


def test_attach_rgb_frame_history_skips_missing_model_type():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace()
    env.num_envs = 2
    obs_list = [
        {"wrist_images": torch.zeros((2, 2, 2, 3), dtype=torch.uint8)},
        {"wrist_images": torch.ones((2, 2, 2, 3), dtype=torch.uint8)},
    ]

    env._attach_rgb_frame_history(obs_list)

    assert "rgb_frame_history" not in obs_list[-1]
    assert "rgb_frame_history_lengths" not in obs_list[-1]
    assert obs_list[-1]["wrist_images"].shape == (2, 2, 2, 3)


def test_update_rgb_frame_history_after_auto_reset_masks_done_envs_only():
    env = _make_chunk_history_test_env()
    original_history = torch.tensor(
        [
            [[[[1]]], [[[2]]], [[[3]]]],
            [[[[4]]], [[[5]]], [[[6]]]],
        ],
        dtype=torch.uint8,
    )
    final_obs = {
        "wrist_images": torch.tensor(
            [[[[30]]], [[[60]]]],
            dtype=torch.uint8,
        ),
        "rgb_frame_history": original_history.clone(),
        "rgb_frame_history_lengths": torch.tensor([3, 3], dtype=torch.long),
    }
    reset_obs = {
        "wrist_images": torch.tensor(
            [[[[30]]], [[[99]]]],
            dtype=torch.uint8,
        ),
    }

    env._update_rgb_frame_history(
        final_obs,
        reset_obs,
        np.array([False, True]),
    )

    assert final_obs["rgb_frame_history_lengths"].tolist() == [3, 1]
    assert final_obs["rgb_frame_history"][0].tolist() == original_history[0].tolist()
    assert final_obs["rgb_frame_history"][1].tolist() == [[[[99]]], [[[99]]], [[[99]]]]


def test_chunk_step_copies_reset_rgb_frame_history_after_auto_reset(monkeypatch):
    env = _make_chunk_history_test_env()
    env.auto_reset = True
    env.ignore_terminations = False
    env.action_map = {0: "forward"}

    step_frames = [
        torch.tensor([[[[10]]], [[[20]]]], dtype=torch.uint8),
        torch.tensor([[[[11]]], [[[21]]]], dtype=torch.uint8),
        torch.tensor([[[[12]]], [[[22]]]], dtype=torch.uint8),
    ]
    terminations = [
        torch.tensor([False, False]),
        torch.tensor([False, True]),
        torch.tensor([False, False]),
    ]
    truncations = [
        torch.tensor([False, False]),
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    ]
    step_calls = []

    def fake_step(actions, auto_reset=True):
        step_idx = len(step_calls)
        step_calls.append((actions.copy(), auto_reset))
        return (
            {"wrist_images": step_frames[step_idx].clone()},
            torch.zeros(2),
            terminations[step_idx],
            truncations[step_idx],
            {"step_idx": step_idx},
        )

    handler_final_observation = {"source": "handler_final_observation"}

    def fake_handle_auto_reset(dones, final_obs, infos):
        assert dones.tolist() == [False, True]
        assert final_obs["rgb_frame_history_lengths"].tolist() == [3, 3]
        return (
            {"wrist_images": torch.tensor([[[[50]]], [[[99]]]], dtype=torch.uint8)},
            {"final_observation": handler_final_observation, "final_info": infos},
        )

    monkeypatch.setattr(env, "step", fake_step)
    monkeypatch.setattr(env, "_handle_auto_reset", fake_handle_auto_reset)

    obs_list, _, _, _, infos_list = env.chunk_step(
        np.zeros((2, 3, 1), dtype=np.int64)
    )

    returned_obs = obs_list[-1]
    assert [auto_reset for _, auto_reset in step_calls] == [False, False, False]
    assert returned_obs["wrist_images"].tolist() == [[[[50]]], [[[99]]]]
    assert returned_obs["rgb_frame_history_lengths"].tolist() == [3, 1]
    assert returned_obs["rgb_frame_history"][0].tolist() == [
        [[[10]]],
        [[[11]]],
        [[[12]]],
    ]
    assert returned_obs["rgb_frame_history"][1].tolist() == [
        [[[99]]],
        [[[99]]],
        [[[99]]],
    ]
    assert infos_list[-1]["final_observation"] is handler_final_observation


def test_chunk_step_auto_reset_skips_rgb_history_when_model_type_missing(monkeypatch):
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace()
    env.auto_reset = True
    env.ignore_terminations = False
    env.action_map = {0: "forward"}
    env.num_envs = 2

    step_frames = [
        torch.tensor([[[[10]]], [[[20]]]], dtype=torch.uint8),
        torch.tensor([[[[11]]], [[[21]]]], dtype=torch.uint8),
    ]
    terminations = [
        torch.tensor([False, False]),
        torch.tensor([False, True]),
    ]
    truncations = [
        torch.tensor([False, False]),
        torch.tensor([False, False]),
    ]

    def fake_step(actions, auto_reset=True):
        step_idx = fake_step.calls
        fake_step.calls += 1
        return (
            {"wrist_images": step_frames[step_idx].clone()},
            torch.zeros(2),
            terminations[step_idx],
            truncations[step_idx],
            {"step_idx": step_idx},
        )

    fake_step.calls = 0

    def fake_handle_auto_reset(dones, final_obs, infos):
        assert dones.tolist() == [False, True]
        return (
            {"wrist_images": torch.tensor([[[[50]]], [[[99]]]], dtype=torch.uint8)},
            {"final_info": infos},
        )

    monkeypatch.setattr(env, "step", fake_step)
    monkeypatch.setattr(env, "_handle_auto_reset", fake_handle_auto_reset)

    obs_list, _, _, _, _ = env.chunk_step(np.zeros((2, 2, 1), dtype=np.int64))

    returned_obs = obs_list[-1]
    assert "rgb_frame_history" not in returned_obs
    assert "rgb_frame_history_lengths" not in returned_obs
    assert returned_obs["wrist_images"].tolist() == [[[[50]]], [[[99]]]]


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
        first_done_reward_mask=np.array([True, True, False, False]),
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
        first_done_reward_mask=np.array([False, False]),
    )

    assert reward.tolist() == [0.0, 0.0]


def test_habitat_weighted_reward_is_zero_for_simultaneous_termination_and_truncation():
    env = _make_reward_test_env(num_envs=1)
    env.dones_once = np.array([False])
    episode = {
        "success": torch.tensor([1.0]),
        "distance_to_goal": torch.tensor([0.0]),
        "ndtw": torch.tensor([1.0]),
    }

    reward = env._calc_step_reward(
        episode,
        first_done_reward_mask=np.array([False]),
    )

    assert reward.tolist() == [0.0]


def test_habitat_record_metrics_includes_ndtw_and_seeds_initial_distance_to_goal():
    env = object.__new__(HabitatEnv)
    env.env_config = SimpleNamespace(
        task=SimpleNamespace(
            measurements=SimpleNamespace(success=SimpleNamespace(success_distance=4.0))
        )
    )
    env._elapsed_steps = np.array([1, 2], dtype=np.int32)
    env.initial_distance_to_goal = np.array([np.nan, 5.0], dtype=np.float32)
    env.first_done_cached_mask = np.array([False, False])
    env.episode_info = None

    infos = {
        "distance_to_goal": [3.5, 3.0],
        "ndtw": [0.25, 0.75],
        "trajectory_Length": [2.0, 2.0],
        "oracle_success": [0.0, 1.0],
        "oracle_navigation_error": [3.5, 0.5],
    }

    recorded_infos = env._record_metrics(
        infos,
        terminations=np.array([False, True]),
        first_done_mask=np.array([False, True]),
    )

    assert env.initial_distance_to_goal.tolist() == [3.5, 5.0]
    assert recorded_infos["episode"]["success"].tolist() == [0.0, 1.0]
    assert recorded_infos["episode"]["spl"].tolist() == [0.0, 1.0]
    assert recorded_infos["episode"]["ndtw"].tolist() == [0.25, 0.75]
