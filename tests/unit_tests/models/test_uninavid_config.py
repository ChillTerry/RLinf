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
    assert cfg.env.eval.total_num_envs == 112
    assert cfg.env.eval.auto_reset is True
    assert cfg.env.eval.ignore_terminations is True
    assert cfg.actor.model.model_type == "uninavid"
    assert cfg.actor.model.rollout_mode == "batched_feature_cache"
    assert cfg.actor.model.num_action_chunks == 4
    assert cfg.actor.model.action_dim == 1
    assert cfg.rollout.generation_backend == "huggingface"
    assert cfg.rollout.collect_prev_infos is False
    assert cfg.rollout.recompute_logprobs is False
    assert cfg.algorithm.sampling_params.do_sample is False
    assert cfg.algorithm.sampling_params.temperature_train == 0.5
    assert cfg.algorithm.sampling_params.temperature_eval == 0.0
    assert cfg.algorithm.sampling_params.top_k == 50
    assert cfg.algorithm.sampling_params.top_p == 0.6
    assert cfg.algorithm.sampling_params.repetition_penalty == 1.0
    assert cfg.algorithm.sampling_params.add_BOS is False
    assert cfg.algorithm.length_params.max_new_token == 1024


def test_habitat_r2r_grpo_uninavid_config_uses_token_training_contract(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(Path("examples/embodiment").resolve()))
    config_dir = str(Path("examples/embodiment/config").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(config_name="habitat_r2r_grpo_uninavid")

    assert cfg.algorithm.adv_type == "grpo"
    assert cfg.algorithm.loss_type == "actor"
    assert cfg.algorithm.reward_type == "chunk_level"
    assert cfg.algorithm.logprob_type == "token_level"
    assert cfg.algorithm.entropy_type == "token_level"
    assert cfg.algorithm.loss_agg_func == "token-mean"
    assert cfg.algorithm.sampling_params.do_sample is True
    assert cfg.env.train.use_rel_reward is True
    assert cfg.env.eval.use_rel_reward is True
    assert cfg.actor.model.model_type == "uninavid"
    assert cfg.actor.model.num_action_chunks == 4
    assert cfg.actor.model.rollout_mode == "batched_feature_cache"
    assert cfg.actor.model.action_dim == 1
    assert (
        cfg.actor.model.model_path
        == "VLN-CE/models/uninavid_weights/uninavid-7b-full-224-video-fps-1-grid-2"
    )
    assert cfg.actor.model.vision_tower == "VLN-CE/models/uninavid_weights/eva_vit_g.pth"
    assert (
        cfg.actor.model.image_processor
        == "rlinf/models/embodiment/uninavid/processor/clip-patch14-224"
    )
    assert cfg.actor.model.precision == "bf16"
    assert cfg.rollout.collect_prev_infos is True
    assert cfg.rollout.recompute_logprobs is False
    assert cfg.rollout.generation_backend == "huggingface"
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    assert cfg.rollout.model.precision == cfg.actor.model.precision
    assert cfg.reward.use_reward_model is False
    assert cfg.runner.only_eval is False
    assert cfg.cluster.component_placement["actor,env,rollout"] == "0-1"
