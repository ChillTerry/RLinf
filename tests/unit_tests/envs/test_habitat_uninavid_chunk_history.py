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
