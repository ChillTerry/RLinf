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

import rlinf.envs.habitat.habitat_env as habitat_env_module
from rlinf.envs.habitat.habitat_env import HabitatEnv


def test_habitat_packs_uninavid_chunk_rgb_history_into_wrist_images():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type="uninavid")
    obs_list = [
        {"wrist_images": torch.full((2, 4, 4, 3), 1, dtype=torch.uint8)},
        {"wrist_images": torch.full((2, 4, 4, 3), 2, dtype=torch.uint8)},
        {"wrist_images": torch.full((2, 4, 4, 3), 3, dtype=torch.uint8)},
    ]

    env._attach_uninavid_chunk_history(obs_list)

    assert "wrist_images_history" not in obs_list[-1]
    assert obs_list[-1]["wrist_images"].shape == (2, 3, 4, 4, 3)
    assert obs_list[-1]["wrist_images"][0, :, 0, 0, 0].tolist() == [1, 2, 3]


def test_habitat_does_not_attach_chunk_history_for_other_models():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type="openvla")
    obs_list = [
        {"wrist_images": torch.full((1, 4, 4, 3), 1, dtype=torch.uint8)},
        {"wrist_images": torch.full((1, 4, 4, 3), 2, dtype=torch.uint8)},
    ]

    env._attach_uninavid_chunk_history(obs_list)

    assert "wrist_images_history" not in obs_list[-1]


def test_habitat_model_type_access_allows_missing_field():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(video_cfg=SimpleNamespace(save_video=False))

    assert env._habitat_model_type() is None


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


def test_habitat_formats_vector_actions_for_current_habitat_api():
    env = object.__new__(HabitatEnv)

    formatted = env._format_habitat_actions(
        np.array(["move_forward", "turn_left"], dtype="U12")
    )

    assert formatted == [{"action": "move_forward"}, {"action": "turn_left"}]


def test_habitat_formats_singleton_action_dim_for_current_habitat_api():
    env = object.__new__(HabitatEnv)

    formatted = env._format_habitat_actions(
        np.array([["turn_left"], ["no_op"]], dtype="U12")
    )

    assert formatted == [{"action": "turn_left"}, {"action": "no_op"}]


def test_habitat_video_wrap_uses_rgb_when_top_down_map_is_missing():
    class EnvStub:
        def get_current_episode_metadata(self):
            return {"episode_id": ["7"]}

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        video_cfg=SimpleNamespace(save_video=True),
    )
    env.env = EnvStub()
    rgb = np.full((4, 5, 3), 13, dtype=np.uint8)

    obs = env._wrap_obs(
        [{"rgb": rgb, "instruction": {"text": "go straight"}}],
        [{}],
    )

    assert obs["main_images"].shape == (1, 4, 5, 3)
    assert obs["main_images"][0, 0, 0, 0].item() == 13


def test_habitat_uninavid_raw_rgb_opt_in_preserves_wrist_images(monkeypatch):
    class EnvStub:
        def get_current_episode_metadata(self):
            return {"episode_id": ["7"]}

    transformed_rgb = np.full((4, 5, 3), 99, dtype=np.uint8)

    def fake_observations_to_image(obs, info=None):
        return {"rgb": transformed_rgb}

    monkeypatch.setattr(
        habitat_env_module,
        "observations_to_image",
        fake_observations_to_image,
    )

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        uninavid_use_raw_rgb=True,
        video_cfg=SimpleNamespace(save_video=True),
    )
    env.env = EnvStub()
    raw_rgb = np.full((4, 5, 3), 13, dtype=np.uint8)

    obs = env._wrap_obs(
        [{"rgb": raw_rgb, "instruction": {"text": "go straight"}}],
        [{}],
    )

    assert obs["main_images"][0, 0, 0, 0].item() == 99
    assert obs["wrist_images"][0, 0, 0, 0].item() == 13


def test_habitat_chunk_step_squeezes_singleton_action_dim_before_stop_padding():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type="uninavid")
    env._elapsed_steps = np.array([0], dtype=np.int32)
    env.max_episode_steps = 10
    env.auto_reset = False
    env.ignore_terminations = False
    env.action_map = {
        0: "stop",
        1: "move_forward",
        2: "turn_left",
        3: "turn_right",
        4: "no_op",
    }
    stepped_actions = []

    def step(actions):
        stepped_actions.append(actions.tolist())
        return (
            {},
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )

    env.step = step

    env.chunk_step(np.array([[[0], [1], [2]]], dtype=np.int64))

    assert stepped_actions == [["stop"], ["no_op"], ["no_op"]]


def test_habitat_step_auto_resets_internal_done_even_when_terminations_are_ignored():
    class EnvStub:
        def step(self, actions):
            return (
                [{"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}],
                np.array([0.0], dtype=np.float32),
                np.array([True]),
                [{"distance_to_goal": 0.0}],
            )

    env = object.__new__(HabitatEnv)
    env.env = EnvStub()
    env._elapsed_steps = np.array([0], dtype=np.int32)
    env.max_episode_steps = 10
    env.ignore_terminations = True
    env.auto_reset = True
    env.dones_once = np.array([False])
    env.first_done_infos = None
    reset_masks = []

    env._normalize_depth = lambda actions, raw_obs: None
    env._record_metrics = lambda infos, terminations: {
        "episode": {"success": torch.tensor([1.0])}
    }
    env._calc_step_reward = lambda success: np.array([1.0], dtype=np.float32)
    env._save_metrics = lambda infos, masks: None
    env._overlay_first_done_episode_metrics = lambda infos: None
    env._wrap_obs = lambda raw_obs, info_lists: {
        "wrist_images": torch.zeros(1, 2, 2, 3)
    }

    def handle_auto_reset(dones, obs, infos):
        reset_masks.append(dones.copy())
        return obs, infos

    env._handle_auto_reset = handle_auto_reset

    _, _, terminations, _, _ = env.step(np.array(["move_forward"]))

    assert reset_masks == [np.array([True])]
    assert terminations.tolist() == [False]


def _make_uninavid_early_stop_env(distance_to_goal_fn, reset_distance_to_goal=1.0):
    class EnvStub:
        def __init__(self):
            self.actions = []
            self.step_count = 0

        def reset(self, env_idx):
            return [
                {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "instruction": {"text": ""},
                }
                for _ in np.asarray(env_idx).reshape(-1)
            ]

        def get_current_metrics(self, id=None):
            env_count = 1 if id is None else len(np.asarray(id).reshape(-1))
            return {"distance_to_goal": [reset_distance_to_goal] * env_count}

        def step(self, actions):
            self.step_count += 1
            self.actions.append(actions)
            return (
                [{"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}],
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                [{"distance_to_goal": distance_to_goal_fn(self.step_count)}],
            )

    stub = EnvStub()
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        uninavid_original_early_stop=True,
        uninavid_early_stop_rotation=25,
        uninavid_early_stop_steps=400,
    )
    env.env = stub
    env.num_envs = 1
    env._elapsed_steps = np.array([0], dtype=np.int32)
    env.initial_distance_to_goal = np.zeros(1, dtype=np.float32)
    env.max_episode_steps = 999
    env.ignore_terminations = False
    env.auto_reset = True
    env.dones_once = np.array([False])
    env.first_done_infos = None
    env.prev_step_reward = np.array([0.0], dtype=np.float32)
    env.current_raw_obs = None

    env._normalize_depth = lambda actions, raw_obs: None
    env._record_metrics = lambda infos, terminations: {
        "episode": {"success": torch.tensor([0.0])}
    }
    env._calc_step_reward = lambda success: np.array([0.0], dtype=np.float32)
    env._save_metrics = lambda infos, masks: None
    env._overlay_first_done_episode_metrics = lambda infos: None
    env._wrap_obs = lambda raw_obs, info_lists=None: {
        "wrist_images": torch.zeros(1, 2, 2, 3)
    }
    env._handle_auto_reset = lambda dones, obs, infos: (obs, infos)

    return env, stub


def test_habitat_reset_seeds_initial_distance_to_goal_from_current_metrics():
    class EnvStub:
        def reset(self, env_idx):
            return [
                {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "instruction": {"text": ""},
                }
                for _ in np.asarray(env_idx).reshape(-1)
            ]

        def get_current_metrics(self, id=None):
            assert np.asarray(id).tolist() == [1]
            return {"distance_to_goal": [7.5]}

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        uninavid_original_early_stop=True,
        video_cfg=SimpleNamespace(save_video=False),
    )
    env.env = EnvStub()
    env.num_envs = 2
    env._elapsed_steps = np.array([3, 4], dtype=np.int32)
    env.prev_step_reward = np.array([1.0, 2.0], dtype=np.float32)
    env.dones_once = np.array([True, True])
    env.first_done_infos = None
    env.current_raw_obs = [
        {"rgb": np.zeros((2, 2, 3), dtype=np.uint8), "instruction": {"text": ""}},
        None,
    ]
    env.initial_distance_to_goal = np.array([3.0, 0.0], dtype=np.float32)
    env.auto_reset = True
    env._uninavid_early_stop_last_distance_to_goal = np.array(
        [3.0, 2.0], dtype=np.float32
    )
    env._uninavid_early_stop_rotation_counts = np.array([5, 6], dtype=np.int32)
    env._wrap_obs = lambda raw_obs, info_lists=None: {"wrist_images": torch.zeros(2)}

    env.reset(env_idx=np.array([1]))

    assert env.initial_distance_to_goal.tolist() == [3.0, 7.5]
    assert env._uninavid_early_stop_last_distance_to_goal.tolist() == [3.0, 7.5]
    assert env._uninavid_early_stop_rotation_counts.tolist() == [5, 0]


def test_uninavid_original_early_stop_uses_reset_seeded_distance_baseline():
    env, stub = _make_uninavid_early_stop_env(lambda _step_count: 1.0)
    env.reset()

    for _ in range(26):
        _, _, terminations, _, _ = env.step(np.array(["move_forward"]))
        assert stub.actions[-1] == [{"action": "move_forward"}]
        assert terminations.tolist() == [False]

    _, _, terminations, _, _ = env.step(np.array(["move_forward"]))

    assert stub.actions[-1] == [{"action": "no_op"}]
    assert terminations.tolist() == [True]


def test_uninavid_original_step_limit_uses_pre_step_iter_counter():
    env, stub = _make_uninavid_early_stop_env(
        lambda step_count: float(1000 - step_count)
    )
    env.reset()
    env._elapsed_steps[:] = 400

    _, _, terminations, _, _ = env.step(np.array(["move_forward"]))

    assert stub.actions[-1] == [{"action": "move_forward"}]
    assert terminations.tolist() == [False]

    _, _, terminations, _, _ = env.step(np.array(["move_forward"]))

    assert stub.actions[-1] == [{"action": "no_op"}]
    assert terminations.tolist() == [True]


def test_habitat_chunk_step_does_not_leak_stop_padding_into_next_episode():
    class EnvStub:
        def __init__(self):
            self.episode_id = 1
            self.actions = []

        def step(self, actions):
            self.actions.append((self.episode_id, actions[0]["action"]))
            return (
                [{"rgb": np.zeros((2, 2, 3), dtype=np.uint8)}],
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                [{"distance_to_goal": 0.0, "trajectory_Length": 0.0}],
            )

        def reset(self, env_idx):
            self.episode_id += 1
            return [
                {
                    "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    "instruction": {"text": ""},
                }
            ]

        def get_current_episode_metadata(self):
            return {"episode_id": [str(self.episode_id)]}

        def get_current_metrics(self, id=None):
            return {"distance_to_goal": [1.0]}

    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        model_type="uninavid",
        video_cfg=SimpleNamespace(save_video=False),
    )
    env.env = EnvStub()
    env.num_envs = 1
    env._elapsed_steps = np.array([0], dtype=np.int32)
    env.prev_step_reward = np.array([0.0], dtype=np.float32)
    env.initial_distance_to_goal = np.full(1, np.nan, dtype=np.float32)
    env.max_episode_steps = 10
    env.auto_reset = True
    env.ignore_terminations = False
    env.dones_once = np.array([False])
    env.first_done_infos = None
    env.current_raw_obs = None
    env.action_map = {
        0: "stop",
        1: "move_forward",
        2: "turn_left",
        3: "turn_right",
        4: "no_op",
    }
    env._normalize_depth = lambda actions, raw_obs: None
    env._record_metrics = lambda infos, terminations: {
        "episode": {"success": torch.tensor([1.0])}
    }
    env._calc_step_reward = lambda success: np.array([0.0], dtype=np.float32)
    env._save_metrics = lambda infos, masks: None
    env._overlay_first_done_episode_metrics = lambda infos: None
    env._wrap_obs = lambda raw_obs, info_lists=None: {
        "wrist_images": torch.zeros(1, 2, 2, 3)
    }

    env.chunk_step(np.array([[[0], [1]]], dtype=np.int64))

    assert env.env.actions == [
        (1, "no_op"),
        (1, "no_op"),
    ]
    assert env.env.episode_id == 2


def test_habitat_save_metrics_preserves_first_file_for_revisited_episode(tmp_path):
    class EnvStub:
        def get_current_episode_metadata(self):
            return {"episode_id": ["episode-7"]}

    env = object.__new__(HabitatEnv)
    env.env = EnvStub()
    env.metrics_cfg = SimpleNamespace(
        save_metrics=True,
        metrics_base_dir=str(tmp_path),
    )
    env.dones_once = np.array([False])
    env.first_done_infos = None
    metric_save_masks = np.array([True])

    first_infos = {
        "episode": {
            "success": torch.tensor([1.0]),
            "spl": torch.tensor([0.75]),
        }
    }
    revisited_infos = {
        "episode": {
            "success": torch.tensor([0.0]),
            "spl": torch.tensor([0.25]),
        }
    }

    env._save_metrics(first_infos, metric_save_masks)
    env._save_metrics(revisited_infos, metric_save_masks)

    metrics_file = tmp_path / "episode_episode-7.json"
    with metrics_file.open() as f:
        metrics = json.load(f)

    assert metrics == {"success": 1.0, "spl": 0.75}
