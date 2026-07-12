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

import json
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

import rlinf.envs.habitat.habitat_env as habitat_env_module
from rlinf.envs.habitat.extensions import video
from rlinf.envs.habitat.extensions.subgoal import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)
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
    assert cfg.habitat.task.measurements.top_down_map.type == "TopDownMap"
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
    assert cfg.gt_path is None


def test_habitat_grpo_uninavid_uses_subgoal_progress_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg["algorithm"]
    assert "success_reward_coef" not in raw_cfg["algorithm"]
    assert "ndtw_reward_coef" not in raw_cfg["algorithm"]
    assert raw_cfg["algorithm"]["reward_mode"] == "subgoal_progress"
    assert raw_cfg["algorithm"]["progress_reward_coef"] == 20
    assert raw_cfg["algorithm"]["subgoal_success_reward_coef"] == 0
    assert raw_cfg["algorithm"]["subgoal_switch_distance"] == 2.0
    assert raw_cfg["algorithm"]["subgoal_success_distance"] == 2.0
    assert raw_cfg["algorithm"]["stop_success_reward_coef"] == 20
    assert raw_cfg["algorithm"]["final_success_distance"] == 2.0
    assert raw_cfg["algorithm"]["premature_stop_coeff"] == 0
    assert raw_cfg["algorithm"]["failure_stop_coeff"] == 10
    assert raw_cfg["algorithm"]["stall_patience"] == 9
    assert raw_cfg["algorithm"]["stall_penalty_coeff"] == 0.0
    assert raw_cfg["algorithm"]["stall_recovery_patience"] == 3
    assert raw_cfg["algorithm"]["stall_observation_patience"] == 6
    assert raw_cfg["algorithm"]["filter_rewards"] is False
    assert raw_cfg["algorithm"]["rewards_lower_bound"] == -10.0
    assert raw_cfg["algorithm"]["rewards_upper_bound"] == 10.0
    assert "success_reward_coef" not in raw_cfg["env"]["train"]
    assert "ndtw_reward_coef" not in raw_cfg["env"]["train"]
    assert raw_cfg["env"]["train"]["reward_mode"] == "${algorithm.reward_mode}"
    assert (
        raw_cfg["env"]["train"]["progress_reward_coef"]
        == "${algorithm.progress_reward_coef}"
    )
    assert (
        raw_cfg["env"]["train"]["subgoal_success_reward_coef"]
        == "${algorithm.subgoal_success_reward_coef}"
    )
    assert (
        raw_cfg["env"]["train"]["subgoal_switch_distance"]
        == "${algorithm.subgoal_switch_distance}"
    )
    assert (
        raw_cfg["env"]["train"]["subgoal_success_distance"]
        == "${algorithm.subgoal_success_distance}"
    )
    assert (
        raw_cfg["env"]["train"]["stop_success_reward_coef"]
        == "${algorithm.stop_success_reward_coef}"
    )
    assert (
        raw_cfg["env"]["train"]["final_success_distance"]
        == "${algorithm.final_success_distance}"
    )
    assert (
        raw_cfg["env"]["train"]["premature_stop_coeff"]
        == "${algorithm.premature_stop_coeff}"
    )
    assert (
        raw_cfg["env"]["train"]["failure_stop_coeff"]
        == "${algorithm.failure_stop_coeff}"
    )
    assert raw_cfg["env"]["train"]["stall_patience"] == "${algorithm.stall_patience}"
    assert (
        raw_cfg["env"]["train"]["stall_penalty_coeff"]
        == "${algorithm.stall_penalty_coeff}"
    )
    assert (
        raw_cfg["env"]["train"]["stall_recovery_patience"]
        == "${algorithm.stall_recovery_patience}"
    )
    assert (
        raw_cfg["env"]["train"]["stall_observation_patience"]
        == "${algorithm.stall_observation_patience}"
    )
    assert (
        raw_cfg["env"]["train"]["data_path"]
        == "${env.data_path_dir}/${env.train.split}/r2r_train_with_subgoals.json"
    )
    assert (
        raw_cfg["env"]["train"]["gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert "success_reward_coef" not in raw_cfg["env"]["eval"]
    assert "ndtw_reward_coef" not in raw_cfg["env"]["eval"]
    assert raw_cfg["env"]["eval"]["reward_mode"] == "${algorithm.reward_mode}"
    assert (
        raw_cfg["env"]["eval"]["progress_reward_coef"]
        == "${algorithm.progress_reward_coef}"
    )
    assert (
        raw_cfg["env"]["eval"]["subgoal_success_reward_coef"]
        == "${algorithm.subgoal_success_reward_coef}"
    )
    assert (
        raw_cfg["env"]["eval"]["subgoal_switch_distance"]
        == "${algorithm.subgoal_switch_distance}"
    )
    assert (
        raw_cfg["env"]["eval"]["subgoal_success_distance"]
        == "${algorithm.subgoal_success_distance}"
    )
    assert (
        raw_cfg["env"]["eval"]["stop_success_reward_coef"]
        == "${algorithm.stop_success_reward_coef}"
    )
    assert (
        raw_cfg["env"]["eval"]["final_success_distance"]
        == "${algorithm.final_success_distance}"
    )
    assert (
        raw_cfg["env"]["eval"]["premature_stop_coeff"]
        == "${algorithm.premature_stop_coeff}"
    )
    assert (
        raw_cfg["env"]["eval"]["failure_stop_coeff"]
        == "${algorithm.failure_stop_coeff}"
    )
    assert raw_cfg["env"]["eval"]["stall_patience"] == "${algorithm.stall_patience}"
    assert (
        raw_cfg["env"]["eval"]["stall_penalty_coeff"]
        == "${algorithm.stall_penalty_coeff}"
    )
    assert (
        raw_cfg["env"]["eval"]["stall_recovery_patience"]
        == "${algorithm.stall_recovery_patience}"
    )
    assert (
        raw_cfg["env"]["eval"]["stall_observation_patience"]
        == "${algorithm.stall_observation_patience}"
    )
    assert (
        raw_cfg["env"]["eval"]["data_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}.json.gz"
    )
    assert (
        raw_cfg["env"]["eval"]["gt_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_gt.json.gz"
    )


def test_habitat_r2r_ppo_uninavid_wires_step_cost_coeff_to_env_configs():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_ppo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert raw_cfg["algorithm"]["step_cost_coeff"] == 0.01
    assert (
        raw_cfg["env"]["train"]["step_cost_coeff"]
        == "${algorithm.step_cost_coeff}"
    )
    assert raw_cfg["env"]["eval"]["step_cost_coeff"] == "${algorithm.step_cost_coeff}"


def test_habitat_subgoal_progress_configs_declare_step_cost_coeff():
    expected_values = {
        "examples/embodiment/config/habitat_r2r_ppo_uninavid.yaml": 0.01,
        "examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml": 0.0,
        "examples/embodiment/config/habitat_rxr_ppo_uninavid.yaml": 0.0,
        "examples/embodiment/config/habitat_rxr_grpo_uninavid.yaml": 0.0,
    }

    for config_path, expected_value in expected_values.items():
        cfg = OmegaConf.load(config_path)
        raw_cfg = OmegaConf.to_container(cfg, resolve=False)

        assert raw_cfg["algorithm"]["reward_mode"] == "subgoal_progress"
        assert raw_cfg["algorithm"]["step_cost_coeff"] == expected_value
        assert raw_cfg["env"]["train"]["step_cost_coeff"] == "${algorithm.step_cost_coeff}"
        assert raw_cfg["env"]["eval"]["step_cost_coeff"] == "${algorithm.step_cost_coeff}"


def test_habitat_eval_uninavid_uses_weighted_reward_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_eval_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert "reward_coef" not in raw_cfg["algorithm"]
    assert raw_cfg["algorithm"]["success_reward_coef"] == 10.0
    assert raw_cfg["algorithm"]["ndtw_reward_coef"] == 5.0
    assert "reward_mode" not in raw_cfg["env"]["train"]
    assert (
        raw_cfg["env"]["train"]["success_reward_coef"]
        == "${algorithm.success_reward_coef}"
    )
    assert (
        raw_cfg["env"]["train"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    )
    assert raw_cfg["env"]["train"]["split"] == "train"
    assert (
        raw_cfg["env"]["train"]["gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert "reward_mode" not in raw_cfg["env"]["eval"]
    assert (
        raw_cfg["env"]["eval"]["success_reward_coef"]
        == "${algorithm.success_reward_coef}"
    )
    assert raw_cfg["env"]["eval"]["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert (
        raw_cfg["env"]["eval"]["gt_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_gt.json.gz"
    )


def test_random_episode_sequences_are_reproducible_by_seed():
    episodes = [
        SimpleNamespace(episode_id=str(episode_idx), scene_id=f"scene-{episode_idx}")
        for episode_idx in range(12)
    ]

    first = random_episode_sequences(
        episodes,
        seed=123,
        auto_reset=False,
        total_num_processes=2,
        num_group=3,
        total_num_envs=6,
        max_steps_per_rollout_epoch=128,
        max_episode_steps=64,
    )
    second = random_episode_sequences(
        episodes,
        seed=123,
        auto_reset=False,
        total_num_processes=2,
        num_group=3,
        total_num_envs=6,
        max_steps_per_rollout_epoch=128,
        max_episode_steps=64,
    )
    different_seed = random_episode_sequences(
        episodes,
        seed=456,
        auto_reset=False,
        total_num_processes=2,
        num_group=3,
        total_num_envs=6,
        max_steps_per_rollout_epoch=128,
        max_episode_steps=64,
    )

    flattened = [
        episode_id
        for process_sequences in first
        for group_sequence in process_sequences
        for episode_id in group_sequence
    ]

    assert first == second
    assert first != different_seed
    assert sorted(flattened, key=int) == [str(episode_idx) for episode_idx in range(12)]


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

    monkeypatch.setattr(
        habitat_env_module, "get_config", lambda *args, **kwargs: dummy_config
    )
    monkeypatch.setattr(
        habitat_env_module.habitat.datasets,
        "make_dataset",
        lambda *args, **kwargs: dummy_dataset,
    )
    monkeypatch.setattr(
        habitat_env_module,
        "random_episode_sequences",
        lambda *args, **kwargs: [[["7", "19"]]],
    )

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        init_params=SimpleNamespace(config_path="dummy-config.yaml"),
        split="train",
        data_path="/tmp/r2r/train/train.json.gz",
        gt_path="/tmp/r2r/train/train_gt.json.gz",
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
    env.reward_mode = "weighted_success_ndtw"
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


def test_habitat_video_frame_concatenates_rgb_and_topdown_map(monkeypatch):
    rgb = np.full((2, 3, 3), 7, dtype=np.uint8)
    topdown = np.full((2, 3, 3), 19, dtype=np.uint8)

    monkeypatch.setattr(
        video,
        "render_topdown_map",
        lambda info: topdown,
    )

    frame = video.build_rollout_video_frame(
        {"rgb": rgb, "instruction": {"text": "go"}},
        {"top_down_map": {"map": np.zeros((2, 3), dtype=np.uint8)}},
    )

    assert frame.shape[0] > 2
    assert frame.shape[1] == 6
    assert np.array_equal(frame[:2, :3], rgb)
    assert np.array_equal(frame[:2, 3:], topdown)


def test_habitat_video_frame_adds_instruction_panel(monkeypatch):
    rgb = np.full((20, 30, 3), 7, dtype=np.uint8)
    topdown = np.full((20, 30, 3), 19, dtype=np.uint8)

    monkeypatch.setattr(
        video,
        "render_topdown_map",
        lambda info: topdown,
    )

    frame = video.build_rollout_video_frame(
        {
            "rgb": rgb,
            "instruction": {"text": "walk down the hallway and stop by the chair"},
        },
        {"top_down_map": {"map": np.zeros((20, 30), dtype=np.uint8)}},
    )

    assert frame.shape[0] > rgb.shape[0]
    assert frame.shape[1] == rgb.shape[1] + topdown.shape[1]
    assert np.array_equal(frame[:20, :30], rgb)
    assert np.array_equal(frame[:20, 30:], topdown)
    assert np.any(frame[20:] < 255)


def test_habitat_video_instruction_panel_keeps_long_instruction_fixed_size():
    base_frame = np.full((20, 60, 3), 7, dtype=np.uint8)
    short = video.append_instruction_panel(base_frame, "turn left")
    long = video.append_instruction_panel(
        base_frame,
        " ".join(["walk through the corridor and continue toward the target"] * 20),
    )

    assert short.shape == long.shape
    assert long.shape[0] > base_frame.shape[0]
    assert np.array_equal(long[:20], base_frame)
    assert np.any(long[20:] < 255)


def test_habitat_video_instruction_display_is_not_all_caps():
    assert video.normalize_instruction_for_video("GO DOWN THE HALLWAY") == (
        "Go down the hallway"
    )
    assert video.normalize_instruction_for_video("Go down the hallway") == (
        "Go down the hallway"
    )


def test_habitat_episode_video_flushes_to_success_and_failure_dirs(
    tmp_path, monkeypatch
):
    render_images = {
        "episode_10": [np.full((2, 2, 3), 10, dtype=np.uint8)],
        "episode_11": [np.full((2, 2, 3), 11, dtype=np.uint8)],
    }
    calls = []

    def fake_save_rollout_video(rollout_images, output_dir, video_name, fps):
        calls.append((rollout_images, output_dir, video_name, fps))

    monkeypatch.setattr(
        video,
        "save_rollout_video",
        fake_save_rollout_video,
    )
    infos = {"episode": {"success": torch.tensor([1.0, 0.0])}}

    video.flush_rollout_videos(
        render_images,
        np.array([True, True]),
        infos,
        episode_ids=["10", "11"],
        video_cfg=SimpleNamespace(video_base_dir=str(tmp_path), fps=12),
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0][0][0], np.full((2, 2, 3), 10, dtype=np.uint8))
    assert calls[0][1:] == (str(tmp_path / "success"), "episode_10", 12)
    assert np.array_equal(calls[1][0][0], np.full((2, 2, 3), 11, dtype=np.uint8))
    assert calls[1][1:] == (str(tmp_path / "failure"), "episode_11", 12)
    assert render_images == {}


def test_habitat_episode_video_flush_uses_cfg_video_cfg_when_attr_missing(
    tmp_path, monkeypatch
):
    env = SimpleNamespace(
        cfg=SimpleNamespace(video_cfg=SimpleNamespace(video_base_dir=str(tmp_path)))
    )
    render_images = {"episode_12": [np.full((2, 2, 3), 12, dtype=np.uint8)]}
    calls = []

    monkeypatch.setattr(
        video,
        "save_rollout_video",
        lambda rollout_images, output_dir, video_name, fps: calls.append(
            (output_dir, video_name, fps)
        ),
    )

    video.flush_rollout_videos(
        render_images,
        np.array([True]),
        {"episode": {"success": [1.0]}},
        episode_ids=["12"],
        video_cfg=video.get_video_cfg(env),
    )

    assert calls == [(str(tmp_path / "success"), "episode_12", 2)]


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
    assert torch.equal(
        final_obs["rgb_frame_history"][:, 0], obs_list[0]["wrist_images"]
    )
    assert torch.equal(
        final_obs["rgb_frame_history"][:, 1], obs_list[1]["wrist_images"]
    )
    assert torch.equal(
        final_obs["rgb_frame_history"][:, 2], obs_list[2]["wrist_images"]
    )


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

    obs_list, _, _, _, infos_list = env.chunk_step(np.zeros((2, 3, 1), dtype=np.int64))

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
    env.metrics_cfg = SimpleNamespace(save_metrics=False)

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


def test_habitat_subgoal_reward_reset_initializes_tracker():
    env = object.__new__(HabitatEnv)
    env.num_envs = 2
    env.cfg = SimpleNamespace(
        reward_mode="subgoal_progress",
        progress_reward_coef=1.0,
        subgoal_success_reward_coef=6.0,
        subgoal_switch_distance=1.0,
        subgoal_success_distance=0.5,
        stop_success_reward_coef=10.0,
        final_success_distance=3.0,
        premature_stop_coeff=4.0,
        failure_stop_coeff=5.0,
        stall_patience=3,
        stall_penalty_coeff=1.0,
    )
    env.reward_mode = "subgoal_progress"
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=2,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            failure_stop_coeff=5.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
            stall_recovery_patience=2,
            stall_observation_patience=2,
        ),
    )
    env.env = SimpleNamespace(
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[4.0, 8.0], [3.0]]
        }
    )

    env._reset_subgoal_reward_state(np.array([0, 1]))

    assert env.subgoal_reward.num_goals.tolist() == [2, 1]
    assert env.subgoal_reward.num_subgoals.tolist() == [1, 0]
    assert env.subgoal_reward.previous_distance_to_active_subgoal.tolist() == [4.0, 3.0]


def test_habitat_subgoal_reward_dispatch_uses_dense_reward_and_attaches_metrics():
    env = object.__new__(HabitatEnv)
    env.num_envs = 1
    env.reward_mode = "subgoal_progress"
    env.episode_info = {}
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=1,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            failure_stop_coeff=5.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
            stall_recovery_patience=2,
            stall_observation_patience=2,
        ),
    )
    env.subgoal_reward.reset([0], [[4.0, 8.0]])
    env.env = SimpleNamespace(
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[3.0, 7.0]]
        }
    )
    infos = {"episode": {}}

    reward = env._calc_step_reward(
        episode={},
        first_done_reward_mask=np.array([False]),
        is_stop=np.array([False]),
        valid_reward_mask=np.array([True]),
        infos=infos,
    )

    assert reward.tolist() == [0.125]
    assert infos["episode"]["r_progress"].tolist() == [0.125]
    assert infos["episode"]["active_subgoal_index"].tolist() == [0.0]


def test_habitat_subgoal_reward_penalizes_truncation_without_stop():
    env = object.__new__(HabitatEnv)
    env.num_envs = 1
    env.reward_mode = "subgoal_progress"
    env.episode_info = {}
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=1,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            failure_stop_coeff=5.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
            stall_recovery_patience=2,
            stall_observation_patience=2,
        ),
    )
    env.subgoal_reward.reset([0], [[2.0]])
    env.env = SimpleNamespace(
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[3.5]]
        }
    )
    infos = {"episode": {}}

    reward = env._calc_step_reward(
        episode={},
        first_done_reward_mask=np.array([False]),
        is_stop=np.array([False]),
        truncations=np.array([True]),
        valid_reward_mask=np.array([True]),
        infos=infos,
    )

    assert reward.tolist() == [-5.75]
    assert infos["episode"]["r_stop"].tolist() == [-5.0]
    assert infos["episode"]["stop_action_ratio"].tolist() == [0.0]


def test_habitat_subgoal_reward_metrics_preserve_cached_first_done_values():
    env = object.__new__(HabitatEnv)
    env.num_envs = 1
    env.episode_info = {
        "success": torch.tensor([1.0]),
    }
    infos = {"episode": {k: v.clone() for k, v in env.episode_info.items()}}

    env._attach_subgoal_reward_metrics(
        infos,
        {
            "r_progress": np.array([0.25], dtype=np.float32),
            "active_subgoal_index": np.array([0.0], dtype=np.float32),
        },
        valid_reward_mask=np.array([True]),
    )
    env._attach_subgoal_reward_metrics(
        infos,
        {
            "r_progress": np.array([0.0], dtype=np.float32),
            "active_subgoal_index": np.array([1.0], dtype=np.float32),
        },
        valid_reward_mask=np.array([False]),
    )

    assert env.episode_info["r_progress"].tolist() == [0.25]
    assert infos["episode"]["r_progress"].tolist() == [0.25]
    assert env.episode_info["active_subgoal_index"].tolist() == [0.0]


def test_habitat_step_writes_subgoal_reward_metrics_after_current_step(tmp_path):
    env = object.__new__(HabitatEnv)
    env.num_envs = 1
    env.reward_mode = "subgoal_progress"
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        video_cfg=SimpleNamespace(save_video=False),
    )
    env.metrics_cfg = SimpleNamespace(
        save_metrics=True,
        metrics_base_dir=str(tmp_path),
    )
    env.env_config = SimpleNamespace(
        task=SimpleNamespace(
            measurements=SimpleNamespace(success=SimpleNamespace(success_distance=3.0))
        ),
        simulator=SimpleNamespace(
            agents=SimpleNamespace(
                main_agent=SimpleNamespace(
                    sim_sensors=SimpleNamespace(
                        depth_sensor=SimpleNamespace(normalize_depth=False)
                    )
                )
            )
        ),
    )
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env.max_episode_steps = 10
    env.ignore_terminations = False
    env.auto_reset = False
    env.first_done_cached_mask = np.zeros(1, dtype=bool)
    env.initial_distance_to_goal = np.array([4.0], dtype=np.float32)
    env.episode_info = None
    env.current_raw_obs = None
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=1,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            failure_stop_coeff=5.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
            stall_recovery_patience=2,
            stall_observation_patience=2,
        ),
    )
    env.subgoal_reward.reset([0], [[4.0]])
    env._normalize_depth = lambda actions, raw_obs: None
    env._wrap_obs = lambda raw_obs, info_lists=None: {}
    env.env = SimpleNamespace(
        step=lambda actions: (
            [{}],
            np.array([0.0], dtype=np.float32),
            np.array([False]),
            [
                {
                    "distance_to_goal": 3.0,
                    "ndtw": 0.5,
                    "trajectory_Length": 1.0,
                    "oracle_success": 0.0,
                    "oracle_navigation_error": 3.0,
                }
            ],
        ),
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[3.0]]
        },
        get_current_episode_metadata=lambda: {"episode_id": ["episode-1"]},
    )

    env.step(np.array(["stop"], dtype="U12"), auto_reset=False)

    metrics_file = tmp_path / "episode_episode-1.json"
    metrics = json.loads(metrics_file.read_text())
    assert metrics["r_progress"] == 0.25
    assert metrics["active_subgoal_index"] == 0.0
    assert "normalized_progress" in metrics


def test_habitat_weighted_reward_dispatch_remains_default():
    env = _make_reward_test_env(num_envs=1)
    env.reward_mode = "weighted_success_ndtw"
    episode = {
        "success": torch.tensor([1.0]),
        "distance_to_goal": torch.tensor([1.5]),
        "ndtw": torch.tensor([0.2]),
    }

    reward = env._calc_step_reward(
        episode,
        first_done_reward_mask=np.array([True]),
    )

    assert reward.tolist() == [6.0]
