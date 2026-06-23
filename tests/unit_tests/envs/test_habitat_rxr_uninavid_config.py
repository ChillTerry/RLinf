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

from omegaconf import OmegaConf

from rlinf.envs.habitat.habitat_env import build_habitat_overrides
from rlinf.envs.habitat.extensions import rxr_dataset  # noqa: F401


def test_habitat_rxr_grpo_uninavid_uses_rxr_paths_and_task_config():
    cfg = OmegaConf.load("examples/embodiment/config/habitat_rxr_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert raw_cfg["runner"]["logger"]["experiment_name"] == (
        "habitat_rxr_grpo_uninavid"
    )
    assert raw_cfg["cluster"]["component_placement"]["actor"] == "4-6"
    assert raw_cfg["cluster"]["component_placement"]["rollout"] == 7
    assert raw_cfg["cluster"]["component_placement"]["env"] == 4
    assert raw_cfg["actor"]["global_batch_size"] == 504
    assert raw_cfg["env"]["data_path_dir"] == "VLN-CE/datasets/rxr"
    assert raw_cfg["env"]["rxr_role"] == "guide"
    assert raw_cfg["env"]["rxr_languages"] == ["en-US", "en-IN"]
    assert raw_cfg["env"]["train"]["rxr_roles"] == "${env.rxr_roles}"
    assert raw_cfg["env"]["train"]["rxr_languages"] == "${env.rxr_languages}"
    assert raw_cfg["env"]["train"]["data_path"] == (
        "${env.data_path_dir}/${env.train.split}/${env.train.split}_${env.rxr_role}.json.gz"
    )
    assert raw_cfg["env"]["train"]["ndtw_gt_path"] == (
        "${env.data_path_dir}/${env.train.split}/${env.train.split}_${env.rxr_role}_gt.json.gz"
    )
    assert raw_cfg["env"]["train"]["init_params"]["config_path"] == (
        "rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml"
    )
    assert raw_cfg["env"]["eval"]["data_path"] == (
        "${env.data_path_dir}/${env.eval.split}/tiny_${env.eval.split}_${env.rxr_role}.json.gz"
    )
    assert raw_cfg["env"]["eval"]["ndtw_gt_path"] == (
        "${env.data_path_dir}/${env.eval.split}/tiny_${env.eval.split}_${env.rxr_role}_gt.json.gz"
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
            "ndtw_gt_path": (
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
            "data_path": "VLN-CE/datasets/rxr/val_unseen/tiny_val_unseen_guide.json.gz",
            "scenes_dir": "VLN-CE/scene_dataset",
            "content_scenes": ["*"],
        }
    )

    dataset = habitat.datasets.make_dataset("RxR-VLN-CE-v1", config=cfg)

    assert len(dataset.episodes) == 738
    first_episode = dataset.episodes[0]
    assert first_episode.instruction.instruction_text
    assert first_episode.scene_id.startswith("VLN-CE/scene_dataset/")
