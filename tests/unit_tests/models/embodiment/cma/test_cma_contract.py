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

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import EMBODIED_MODEL, SupportedModel
from rlinf.models import get_model
from rlinf.models.embodiment.cma.cma_action_model import CMAPolicy
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _RecordingPolicy:
    def __init__(self):
        self.calls = []

    def predict_action_batch(self, env_obs, **kwargs):
        self.calls.append(kwargs)
        actions = torch.zeros(2, 1, dtype=torch.long)
        return actions, {"forward_inputs": {"action": actions}}


def test_cma_builtin_model_registration_smoke():
    cfg = OmegaConf.load("examples/embodiment/config/model/cma.yaml")
    cfg.load_to_device = False
    cfg.rgb_encoder_config.cnn_type = "TorchVisionResNet18"
    cfg.rgb_encoder_config.trainable = False
    cfg.depth_encoder_config.trainable = False

    assert SupportedModel.CMA_POLICY in EMBODIED_MODEL

    model = get_model(cfg)

    assert isinstance(model, CMAPolicy)


def test_cma_rollout_predict_passes_eval_mode():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "cma"}},
            "algorithm": {"loss_type": "actor"},
        }
    )
    worker._train_sampling_params = {"temperature": 1.0}
    worker._eval_sampling_params = {"temperature": 0.0}
    worker._dagger_sampling_params = {"beta": 0.0}
    worker.expert_model = None
    worker.hf_model = _RecordingPolicy()
    worker._timer_metrics = {}

    actions, result = worker.predict(env_obs={}, mode="eval")

    assert actions.shape == (2, 1)
    assert result["expert_label_flag"] is False
    assert worker.hf_model.calls == [{"mode": "eval"}]


def test_cma_grpo_action_level_loss_contract():
    torch.manual_seed(0)

    num_steps = 3
    batch_size = 8
    group_size = 4

    rewards = torch.randn(num_steps, batch_size, 1)
    dones = torch.zeros(num_steps + 1, batch_size, 1, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones(num_steps, batch_size, 1, dtype=torch.bool)

    adv_outputs = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        reward_type="action_level",
        rewards=rewards,
        dones=dones,
        loss_mask=loss_mask,
        group_size=group_size,
        normalize_advantages=True,
    )
    advantages = adv_outputs["advantages"]

    assert advantages.shape == (num_steps, batch_size, 1)

    logprobs = torch.randn(batch_size, 1).float()
    old_logprobs = logprobs.detach() + 0.01 * torch.randn(batch_size, 1)
    micro_advantages = advantages[0].float()

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="action_level",
        reward_type="action_level",
        single_action_dim=1,
        logprobs=logprobs,
        old_logprobs=old_logprobs.float(),
        advantages=micro_advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        loss_mask=torch.ones_like(micro_advantages, dtype=torch.bool),
        max_episode_steps=128,
        critic_warmup=False,
    )

    assert loss.ndim == 0
    assert "actor/policy_loss" in metrics
