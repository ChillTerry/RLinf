# Copyright 2025 The RLinf Authors.
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

import pytest
from omegaconf import OmegaConf

import rlinf.config as config_module


class _StubHybridComponentPlacement:
    def __init__(self, config, cluster):
        del config, cluster

    def get_world_size(self, component: str) -> int:
        assert component == "env"
        return 1


def _make_habitat_cfg():
    return OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {
                    "actor,env,rollout": "0",
                },
            },
            "runner": {
                "only_eval": False,
                "val_check_interval": 1,
            },
            "algorithm": {
                "loss_type": "actor",
            },
            "rollout": {
                "pipeline_stage_num": 1,
            },
            "actor": {
                "model": {
                    "model_type": "navid",
                    "num_action_chunks": 1,
                },
            },
            "env": {
                "enable_offload": False,
                "train": {
                    "env_type": "habitat",
                    "total_num_envs": 1,
                    "group_size": 1,
                    "max_steps_per_rollout_epoch": 1,
                    "auto_reset": False,
                    "enable_offload": False,
                },
                "eval": {
                    "env_type": "habitat",
                    "total_num_envs": 1,
                    "group_size": 1,
                    "max_steps_per_rollout_epoch": 1,
                    "auto_reset": True,
                    "enable_offload": False,
                },
            },
        }
    )


@pytest.fixture(autouse=True)
def _stub_cluster_and_placement(monkeypatch):
    monkeypatch.setattr(config_module, "Cluster", lambda: object())
    monkeypatch.setattr(
        config_module,
        "HybridComponentPlacement",
        _StubHybridComponentPlacement,
    )


def test_validate_embodied_cfg_rejects_habitat_train_offload_with_auto_reset():
    cfg = _make_habitat_cfg()
    cfg.env.train.enable_offload = True
    cfg.env.train.auto_reset = True

    with pytest.raises(
        AssertionError,
        match="Habitat.*env\\.train\\.enable_offload.*env\\.train\\.auto_reset=False",
    ):
        config_module.validate_embodied_cfg(cfg)


def test_validate_embodied_cfg_rejects_habitat_top_level_env_offload_only():
    cfg = _make_habitat_cfg()
    cfg.env.enable_offload = True
    cfg.env.train.enable_offload = False
    cfg.env.eval.enable_offload = False

    with pytest.raises(
        AssertionError,
        match="Habitat.*top-level env\\.enable_offload.*(env\\.train\\.enable_offload|env\\.eval\\.enable_offload)",
    ):
        config_module.validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("section", "auto_reset"),
    [("train", False), ("eval", True)],
)
def test_validate_embodied_cfg_accepts_nested_habitat_offload_safe_cases(
    section: str, auto_reset: bool
):
    cfg = _make_habitat_cfg()
    cfg.env[section].enable_offload = True
    cfg.env[section].auto_reset = auto_reset

    validated_cfg = config_module.validate_embodied_cfg(cfg)

    assert validated_cfg is cfg
    assert validated_cfg.env[section].enable_offload is True
