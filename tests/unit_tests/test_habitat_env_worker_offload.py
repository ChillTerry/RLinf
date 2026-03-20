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

from unittest.mock import MagicMock
from typing import cast

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.env.env_worker import EnvWorker


def _make_env_worker(*, train_auto_reset: bool, eval_auto_reset: bool) -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"num_action_chunks": 1}},
            "algorithm": {"eval_rollout_epoch": 1},
            "env": {
                "train": {"auto_reset": train_auto_reset},
                "eval": {
                    "auto_reset": eval_auto_reset,
                    "enable_offload": True,
                },
            },
        }
    )
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 1
    worker.eval_env_list = []
    worker.env_list = []
    worker.last_obs_list = [{"states": torch.tensor([[999.0]])}]
    worker.last_intervened_info_list = cast(
        list[tuple[torch.Tensor | None, torch.Tensor | None]], [(None, None)]
    )
    worker.n_eval_chunk_steps = 1
    return worker


def _make_async_env_worker_cfg(
    *, train_enable_offload: bool, eval_enable_offload: bool
):
    return OmegaConf.create(
        {
            "env": {
                "train": {"enable_offload": train_enable_offload},
                "eval": {"enable_offload": eval_enable_offload},
            }
        }
    )


def test_bootstrap_step_uses_reset_when_train_auto_reset_is_disabled():
    worker = _make_env_worker(train_auto_reset=False, eval_auto_reset=True)
    reset_obs = {"states": torch.tensor([[1.0]])}
    worker.env_list = [
        MagicMock(
            reset=MagicMock(
                return_value=(
                    reset_obs,
                    {"final_observation": {"states": torch.tensor([[2.0]])}},
                )
            )
        )
    ]

    env_outputs = worker.bootstrap_step()

    worker.env_list[0].reset.assert_called_once_with()
    assert worker.env_list[0].is_start is True
    assert len(env_outputs) == 1
    assert torch.equal(env_outputs[0].obs["states"], reset_obs["states"])
    assert env_outputs[0].final_obs is not None
    assert torch.equal(env_outputs[0].final_obs["states"], torch.tensor([[2.0]]))
    assert env_outputs[0].dones is not None
    assert torch.equal(env_outputs[0].dones, torch.zeros((1, 1), dtype=torch.bool))


def test_evaluate_resets_before_each_eval_cycle_after_offload():
    worker = _make_env_worker(train_auto_reset=False, eval_auto_reset=True)
    reset_obs_1 = {"states": torch.tensor([[1.0]])}
    reset_obs_2 = {"states": torch.tensor([[2.0]])}
    eval_env = MagicMock()
    eval_env.reset.side_effect = [
        (reset_obs_1, {"final_observation": {"states": torch.tensor([[11.0]])}}),
        (reset_obs_2, {"final_observation": {"states": torch.tensor([[22.0]])}}),
    ]
    worker.eval_env_list = [eval_env]
    worker.recv_chunk_actions = MagicMock(return_value=torch.tensor([[0.0]]))
    worker.env_evaluate_step = MagicMock(
        return_value=(
            MagicMock(to_dict=MagicMock(return_value={"obs": {}})),
            {"success": torch.tensor([1.0])},
        )
    )
    worker.finish_rollout = MagicMock()
    sent_batches = []

    def _capture_send(_channel, env_batch, mode="train"):
        sent_batches.append((mode, env_batch))

    setattr(worker, "send_env_batch", _capture_send)

    worker.evaluate(MagicMock(), MagicMock())
    worker.evaluate(MagicMock(), MagicMock())

    assert eval_env.reset.call_count == 2
    assert eval_env.offload.call_count == 2
    assert [mode for mode, _ in sent_batches] == ["eval", "eval"]
    assert torch.equal(sent_batches[0][1]["obs"]["states"], reset_obs_1["states"])
    assert torch.equal(sent_batches[1][1]["obs"]["states"], reset_obs_2["states"])
    worker.env_evaluate_step.assert_called()


def test_async_env_worker_rejects_offload_with_existing_assertion(monkeypatch):
    def _fake_env_worker_init(self, cfg):
        self.enable_offload = (
            cfg.env.train.enable_offload or cfg.env.eval.enable_offload
        )

    monkeypatch.setattr(EnvWorker, "__init__", _fake_env_worker_init)

    with pytest.raises(AssertionError, match="Offload not supported in AsyncEnvWorker"):
        AsyncEnvWorker(
            _make_async_env_worker_cfg(
                train_enable_offload=True, eval_enable_offload=False
            )
        )


def test_async_env_worker_rejects_eval_offload_with_existing_assertion(monkeypatch):
    def _fake_env_worker_init(self, cfg):
        self.enable_offload = (
            cfg.env.train.enable_offload or cfg.env.eval.enable_offload
        )

    monkeypatch.setattr(EnvWorker, "__init__", _fake_env_worker_init)

    with pytest.raises(AssertionError, match="Offload not supported in AsyncEnvWorker"):
        AsyncEnvWorker(
            _make_async_env_worker_cfg(
                train_enable_offload=False, eval_enable_offload=True
            )
        )
