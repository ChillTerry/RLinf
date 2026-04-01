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

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.models.embodiment.navid import get_model
from rlinf.models.embodiment.navid.navid_action_model import NaVidForRLActionPrediction


class _DummyTokenizer:
    pad_token_id = 0


class _DummyImageProcessor:
    pass


class _DummyNaVidModel(nn.Module):
    def __init__(self, hidden_size: int = 2):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.config = SimpleNamespace(hidden_size=hidden_size)

    def update_prompt(self, prompts=None):
        self.prompts = prompts


def _make_policy(*, add_value_head: bool = True) -> NaVidForRLActionPrediction:
    policy = NaVidForRLActionPrediction(
        tokenizer=_DummyTokenizer(),
        model=_DummyNaVidModel(),
        image_processor=_DummyImageProcessor(),
        action_dim=1,
        num_action_chunks=3,
        add_value_head=add_value_head,
        hidden_size=2,
        max_prompt_length=8,
    )
    if add_value_head:
        policy.value_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            policy.value_head.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return policy


def test_navid_get_model_forwards_ppo_config(monkeypatch):
    captured_kwargs = {}

    def _stub_from_pretrained(cls, **kwargs):
        captured_kwargs.update(kwargs)
        return "navid-model"

    monkeypatch.setattr(
        NaVidForRLActionPrediction,
        "from_pretrained",
        classmethod(_stub_from_pretrained),
    )

    cfg = OmegaConf.create(
        {
            "model_path": "/tmp/navid",
            "action_dim": 1,
            "num_action_chunks": 3,
            "add_value_head": True,
            "max_prompt_length": 512,
            "max_history_len": 120,
        }
    )

    model = get_model(cfg, torch_dtype=torch.bfloat16)

    assert model == "navid-model"
    assert captured_kwargs["add_value_head"] is True
    assert captured_kwargs["max_prompt_length"] == 512
    assert captured_kwargs["max_history_len"] == 120
    assert captured_kwargs["torch_dtype"] == torch.bfloat16


def test_predict_action_batch_returns_response_level_values_and_token_logprobs(
    monkeypatch,
):
    policy = _make_policy(add_value_head=True)
    device = next(policy.model.parameters()).device

    monkeypatch.setattr(policy, "preprocess_env_obs", lambda env_obs: env_obs)
    monkeypatch.setattr(
        policy,
        "_get_generation_params",
        lambda **kwargs: {"do_sample": True, "temperature": 0.2, "max_new_tokens": 3},
    )
    monkeypatch.setattr(
        policy,
        "_build_prompts_and_convs",
        lambda **kwargs: (
            ["prompt_a", "prompt_b"],
            ["question_a", "question_b"],
            [object(), object()],
        ),
    )
    monkeypatch.setattr(policy, "_build_special_token_tensors", lambda **kwargs: None)
    monkeypatch.setattr(
        policy,
        "_tokenize_and_pad_prompts",
        lambda **kwargs: (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device),
            torch.ones((2, 4), dtype=torch.long, device=device),
        ),
    )
    monkeypatch.setattr(
        policy, "_build_stopping_criteria", lambda **kwargs: ("stop", object())
    )
    monkeypatch.setattr(
        policy,
        "_preprocess_new_frames",
        lambda **kwargs: torch.zeros((2, 3, 2, 2), dtype=torch.float32, device=device),
    )
    monkeypatch.setattr(
        policy,
        "_accumulate_history_frames",
        lambda **kwargs: [torch.zeros((1, 3, 2, 2), device=device) for _ in range(2)],
    )
    monkeypatch.setattr(
        policy,
        "_pad_history_frames_for_model",
        lambda **kwargs: kwargs["images_for_model"],
    )
    monkeypatch.setattr(
        policy,
        "_generate",
        lambda **kwargs: SimpleNamespace(
            scores=(
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 1.0], [0.3, 0.1, 0.2, 0.9, 0.4]],
                    device=device,
                ),
                torch.tensor(
                    [[0.3, 0.5, 0.1, 1.2, 0.4], [0.1, 0.2, 1.1, 0.4, 0.5]],
                    device=device,
                ),
                torch.tensor(
                    [[0.1, 0.2, 1.5, 0.3, 0.4], [2.0, 0.1, 0.2, 0.3, 0.4]],
                    device=device,
                ),
            ),
            sequences=torch.tensor(
                [
                    [1, 2, 3, 4, 4, 3, 2],
                    [5, 6, 7, 8, 3, 0, 0],
                ],
                device=device,
            ),
            hidden_states=(
                (
                    torch.tensor(
                        [
                            [[1.0, 2.0], [3.0, 4.0]],
                            [[5.0, 6.0], [7.0, 8.0]],
                        ],
                        device=device,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        policy, "_decode_generated_texts", lambda **kwargs: ["forward 25", "stop"]
    )
    monkeypatch.setattr(
        policy,
        "_parse_actions_from_texts",
        lambda **kwargs: np.full((2, 3, 1), "move_forward", dtype="<U12"),
    )

    env_obs = {
        "main_images": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        "task_descriptions": ["task_a", "task_b"],
        "states": torch.tensor([11, 22]),
    }

    _, result = policy.predict_action_batch(env_obs=env_obs)

    assert result["prev_logprobs"].shape == (2, 3)
    assert result["prev_values"].shape == (2, 1)
    assert torch.allclose(
        result["prev_values"],
        torch.tensor([[3.0], [7.0]], device=device),
    )
    assert torch.equal(
        result["forward_inputs"]["response_mask"],
        torch.tensor([[1, 1, 1], [1, 0, 0]], device=device),
    )
    assert torch.allclose(result["prev_logprobs"][1, 1:], torch.zeros(2, device=device))


def test_default_forward_returns_response_level_values(monkeypatch):
    policy = _make_policy(add_value_head=True)

    def _stub_forward_teacher_forcing_multimodal(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
    ):
        del input_ids, attention_mask, pixel_values, output_hidden_states
        batch_size, seq_len = labels.shape
        logits = torch.randn((batch_size, seq_len, 8), dtype=torch.float32)
        hidden = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [3.0, 0.0],
                    [10.0, 1.0],
                    [30.0, 1.0],
                    [40.0, 1.0],
                    [50.0, 1.0],
                ],
                [
                    [1.0, 1.0],
                    [2.0, 1.0],
                    [3.0, 1.0],
                    [20.0, 2.0],
                    [60.0, 2.0],
                    [70.0, 2.0],
                    [80.0, 2.0],
                ],
            ],
            dtype=torch.float32,
        )
        outputs = SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,),
            attentions=None,
            past_key_values=None,
        )
        return outputs, labels

    monkeypatch.setattr(
        policy,
        "_forward_teacher_forcing_multimodal",
        _stub_forward_teacher_forcing_multimodal,
    )

    forward_inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "pixel_values": torch.zeros((2, 1, 3, 2, 2), dtype=torch.float32),
        "response_token_ids": torch.tensor([[4, 5, 6], [6, 5, 4]], dtype=torch.long),
        "response_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
    }

    output = policy.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_values=True,
    )

    assert output["logprobs"].shape == (2, 3)
    assert output["values"].shape == (2, 1)
    assert torch.allclose(output["values"], torch.tensor([[10.0], [20.0]]))
