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


def test_habitat_r2r_cma_extension_config_loads():
    cfg = OmegaConf.load("rlinf/envs/habitat/extensions/config/vlnce_r2r_cma.yaml")

    assert cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width == 224
    assert cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width == 256
    assert cfg.habitat.task.measurements.success.success_distance == 3.0


def test_habitat_eval_cma_uses_current_reward_config():
    raw_cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_eval_cma.yaml")
    raw_env_eval = OmegaConf.to_container(raw_cfg.env.eval, resolve=False)

    assert raw_cfg.defaults[2] == "model/cma@actor.model"
    assert raw_cfg.algorithm.reward_type == "action_level"
    assert raw_cfg.algorithm.logprob_type == "action_level"
    assert "reward_coef" not in raw_cfg.algorithm
    assert raw_env_eval["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_env_eval["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert (
        raw_env_eval["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.eval.split}/${env.eval.split}_gt.json.gz"
    )
    assert (
        raw_env_eval["init_params"]["config_path"]
        == "rlinf/envs/habitat/extensions/config/vlnce_r2r_cma.yaml"
    )


def test_habitat_grpo_cma_uses_current_reward_config():
    raw_cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_grpo_cma.yaml")
    raw_env_train = OmegaConf.to_container(raw_cfg.env.train, resolve=False)

    assert raw_cfg.defaults[2] == "model/cma@actor.model"
    assert raw_cfg.algorithm.reward_type == "action_level"
    assert raw_cfg.algorithm.logprob_type == "action_level"
    assert "reward_coef" not in raw_cfg.algorithm
    assert raw_cfg.rollout.collect_prev_infos is True
    assert raw_env_train["group_size"] == "${algorithm.group_size}"
    assert raw_env_train["success_reward_coef"] == "${algorithm.success_reward_coef}"
    assert raw_env_train["ndtw_reward_coef"] == "${algorithm.ndtw_reward_coef}"
    assert (
        raw_env_train["ndtw_gt_path"]
        == "${env.data_path_dir}/${env.train.split}/${env.train.split}_gt.json.gz"
    )
    assert (
        raw_env_train["init_params"]["config_path"]
        == "rlinf/envs/habitat/extensions/config/vlnce_r2r_cma.yaml"
    )
