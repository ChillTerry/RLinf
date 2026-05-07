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

import inspect
from types import SimpleNamespace

import torch

from rlinf.config import SupportedModel
from rlinf.workers.rollout.hf import huggingface_worker
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def test_build_dataloader_has_uninavid_sft_branch():
    source = inspect.getsource(FSDPVlaSftWorker.build_dataloader)

    assert "SupportedModel.UNINAVID" in source
    assert "build_uninavid_sft_dataloader" in source


def test_get_train_model_output_uses_uninavid_sft_forward_path():
    source = inspect.getsource(FSDPVlaSftWorker.get_train_model_output)

    assert "SupportedModel.UNINAVID" in source
    assert "ForwardType.SFT" in source


class FakeAlgorithmConfig(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def predict_action_batch(self, *, env_obs, **kwargs):
        self.calls.append({"env_obs": env_obs, "kwargs": kwargs})
        return torch.zeros(1, 1, 1), {"forward_inputs": {}}


def make_rollout_worker(*, loss_type="grpo"):
    worker = object.__new__(huggingface_worker.MultiStepRolloutWorker)
    worker._timer_metrics = {}
    worker._train_sampling_params = {
        "max_new_tokens": 8,
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.9,
    }
    worker._eval_sampling_params = {
        "max_new_tokens": 4,
        "do_sample": False,
        "temperature": 0.1,
        "top_k": 1,
        "top_p": 1.0,
    }
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(
            model=SimpleNamespace(model_type=SupportedModel.UNINAVID.value)
        ),
        algorithm=FakeAlgorithmConfig(loss_type=loss_type),
    )
    worker.expert_model = None
    worker.hf_model = RecordingPolicy()
    return worker


def test_hf_rollout_worker_passes_mode_to_uninavid_without_dropping_sampling_kwargs():
    worker = make_rollout_worker()
    env_obs = {"task_descriptions": ["go"]}

    worker.predict(env_obs, mode="train")

    assert worker.hf_model.calls[-1]["kwargs"] == {
        **worker._train_sampling_params,
        "mode": "train",
    }

    worker.predict(env_obs, mode="eval")

    assert worker.hf_model.calls[-1]["kwargs"] == {
        **worker._eval_sampling_params,
        "mode": "eval",
    }


def test_hf_rollout_worker_dagger_passes_eval_mode_to_uninavid_with_sampling_kwargs():
    worker = make_rollout_worker(loss_type="embodied_dagger")

    worker.predict({"task_descriptions": ["go"]}, mode="train")

    assert worker.hf_model.calls[-1]["kwargs"] == {
        **worker._train_sampling_params,
        "mode": "eval",
    }
