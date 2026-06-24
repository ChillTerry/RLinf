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

import gzip
import json

from omegaconf import OmegaConf

from rlinf.envs.habitat.habitat_env import build_habitat_overrides
from rlinf.envs.habitat.extensions import rxr_dataset  # noqa: F401


def test_habitat_rxr_grpo_uninavid_uses_rxr_paths_and_task_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_rxr_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert raw_cfg["runner"]["logger"]["experiment_name"] == (
        "habitat_rxr_grpo_uninavid"
    )
    assert raw_cfg["cluster"]["component_placement"]["actor,env,rollout"] == "4-7"
    assert raw_cfg["actor"]["global_batch_size"] == 512
    assert raw_cfg["actor"]["model"]["drop_overlong_train_metadata"] is True
    assert raw_cfg["env"]["data_path_dir"] == "VLN-CE/datasets/rxr"
    assert raw_cfg["env"]["rxr_role"] == "guide"
    assert raw_cfg["env"]["rxr_languages"] == ["en-US", "en-IN"]
    assert raw_cfg["env"]["train"]["rxr_roles"] == "${env.rxr_roles}"
    assert raw_cfg["env"]["train"]["rxr_languages"] == "${env.rxr_languages}"
    assert raw_cfg["env"]["train"]["data_path"] == (
        "${env.data_path_dir}/${env.train.split}/${env.train.split}_${env.rxr_role}_reachable.json.gz"
    )
    assert raw_cfg["env"]["train"]["gt_path"] == (
        "${env.data_path_dir}/${env.train.split}/${env.train.split}_${env.rxr_role}_gt_reachable.json.gz"
    )
    assert raw_cfg["env"]["train"]["init_params"]["config_path"] == (
        "rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml"
    )
    assert raw_cfg["env"]["eval"]["data_path"] == (
        "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_${env.rxr_role}_reachable.json.gz"
    )
    assert raw_cfg["env"]["eval"]["gt_path"] == (
        "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_${env.rxr_role}_gt_reachable.json.gz"
    )
    assert raw_cfg["env"]["eval"]["init_params"]["config_path"] == (
        "rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml"
    )
    assert raw_cfg["env"]["eval"]["rxr_roles"] == "${env.rxr_roles}"
    assert raw_cfg["env"]["eval"]["rxr_languages"] == "${env.rxr_languages}"


def test_habitat_rxr_uninavid_extension_config_has_rxr_dataset_fields():
    cfg = OmegaConf.load("rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert raw_cfg["habitat"]["dataset"]["type"] == "RxR-VLN-CE-v1"
    assert raw_cfg["habitat"]["simulator"]["turn_angle"] == 30
    assert raw_cfg["habitat"]["simulator"]["agents"]["main_agent"]["sim_sensors"][
        "rgb_sensor"
    ]["width"] == 640
    assert raw_cfg["habitat"]["simulator"]["agents"]["main_agent"]["sim_sensors"][
        "rgb_sensor"
    ]["height"] == 480
    assert raw_cfg["habitat"]["task"]["measurements"]["ndtw"]["GT_PATH"] == (
        "VLN-CE/datasets/rxr/{split}/{split}_guide_gt.json.gz"
    )


def test_habitat_rxr_dataset_fields_are_added_as_hydra_overrides():
    cfg = OmegaConf.create(
        {
            "split": "val_unseen",
            "data_path": "VLN-CE/datasets/rxr/val_unseen/val_unseen_guide.json.gz",
            "scenes_dir": "VLN-CE/scene_dataset",
            "gt_path": (
                "VLN-CE/datasets/rxr/val_unseen/val_unseen_guide_gt.json.gz"
            ),
            "rxr_roles": ["guide"],
            "rxr_languages": ["en-US", "en-IN"],
        }
    )

    overrides = build_habitat_overrides(cfg, max_episode_steps=100)

    assert "+habitat.dataset.ROLES=[guide]" in overrides
    assert "+habitat.dataset.LANGUAGES=[en-US,en-IN]" in overrides


def test_habitat_rxr_dataset_loader_reads_rxr_without_instruction_vocab():
    import habitat

    cfg = OmegaConf.create(
        {
            "split": "val_unseen",
            "data_path": (
                "VLN-CE/datasets/rxr/val_unseen/"
                "val_unseen_guide_reachable.json.gz"
            ),
            "scenes_dir": "VLN-CE/scene_dataset",
            "content_scenes": ["*"],
        }
    )

    dataset = habitat.datasets.make_dataset("RxR-VLN-CE-v1", config=cfg)

    assert len(dataset.episodes) > 0
    first_episode = dataset.episodes[0]
    assert first_episode.instruction.instruction_text
    assert first_episode.scene_id.startswith("VLN-CE/scene_dataset/")


def test_habitat_rxr_dataset_loader_filters_configured_languages(tmp_path):
    import habitat

    dataset_path = tmp_path / "train_guide.json.gz"
    scenes_dir = tmp_path / "scenes"
    scenes_dir.mkdir()
    episodes = [
        _rxr_episode("1", "English US", "en-US"),
        _rxr_episode("2", "English India", "en-IN"),
        _rxr_episode("3", "Hindi", "hi-IN"),
        _rxr_episode("4", "Telugu", "te-IN"),
    ]
    with gzip.open(dataset_path, "wt", encoding="utf-8") as file_obj:
        json.dump({"episodes": episodes}, file_obj)

    cfg = OmegaConf.create(
        {
            "split": "train",
            "data_path": str(dataset_path),
            "scenes_dir": str(scenes_dir),
            "content_scenes": ["*"],
            "LANGUAGES": ["en-US", "en-IN"],
        }
    )

    dataset = habitat.datasets.make_dataset("RxR-VLN-CE-v1", config=cfg)

    assert [episode.episode_id for episode in dataset.episodes] == ["1", "2"]
    assert [
        episode.instruction.instruction_text for episode in dataset.episodes
    ] == ["English US", "English India"]


def _rxr_episode(episode_id, instruction_text, language):
    return {
        "episode_id": episode_id,
        "trajectory_id": int(episode_id),
        "scene_id": "mp3d/example/example.glb",
        "info": {"role": "guide"},
        "instruction": {
            "instruction_id": episode_id,
            "instruction_text": instruction_text,
            "language": language,
        },
        "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
        "goals": [{"position": [1.0, 0.0, 0.0], "radius": 3.0}],
    }
