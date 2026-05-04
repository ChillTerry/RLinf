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

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_uninavid_stage_one_config_allows_resume_override():
    config_dir = str(Path("examples/sft/config").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="uninavid_stage_1_sft",
            overrides=["runner.resume_dir=/tmp/uninavid-step-1"],
        )

    assert cfg.runner.resume_dir == "/tmp/uninavid-step-1"


def test_uninavid_habitat_eval_config_composes(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(Path("examples/embodiment").resolve()))
    config_dir = str(Path("examples/embodiment/config").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name="habitat_r2r_eval_uninavid")

    assert cfg.runner.only_eval is True
    assert cfg.env.eval.env_type == "habitat"
    assert cfg.env.eval.model_type == "uninavid"
    assert cfg.actor.model.model_type == "uninavid"
    assert cfg.actor.model.rollout_mode == "batched_feature_cache"
    assert cfg.actor.model.num_action_chunks == 4
    assert cfg.actor.model.action_dim == 1
    assert cfg.rollout.generation_backend == "huggingface"
    assert cfg.algorithm.sampling_params.do_sample is True
    assert cfg.algorithm.sampling_params.temperature_eval == 0.5
    assert cfg.algorithm.length_params.max_new_token == 1024
