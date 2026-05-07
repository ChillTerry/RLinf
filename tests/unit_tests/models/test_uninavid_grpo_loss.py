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

import pytest
import torch

from rlinf.algorithms.losses import compute_grpo_actor_loss_fn
from rlinf.models.embodiment.uninavid.rl_loss import (
    prepare_uninavid_token_level_loss_inputs,
)
from rlinf.utils.utils import masked_mean


def test_uninavid_token_loss_inputs_mask_padding_tokens():
    logprobs = torch.tensor([[[-0.1], [-0.2], [-9.0]]])
    old_logprobs = torch.tensor([[[-0.1], [-0.2], [-8.0]]])
    advantages = torch.tensor([1.5])
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)

    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        response_mask=response_mask,
    )

    assert prepared["loss_mask"].tolist() == [[[True], [True], [False]]]
    assert prepared["advantages"].shape == (1, 3, 1)
    assert prepared["advantages"][0, :, 0].tolist() == [1.5, 1.5, 1.5]


def test_uninavid_token_loss_inputs_combine_transition_and_response_masks():
    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=torch.zeros(2, 3, 1),
        old_logprobs=torch.zeros(2, 3, 1),
        advantages=torch.ones(2),
        response_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        sample_loss_mask=torch.tensor([[1], [0]], dtype=torch.bool),
    )

    assert prepared["loss_mask"].tolist() == [
        [[True], [True], [False]],
        [[False], [False], [False]],
    ]


def test_uninavid_token_loss_inputs_reject_shape_mismatch():
    with pytest.raises(ValueError, match="response_mask"):
        prepare_uninavid_token_level_loss_inputs(
            logprobs=torch.zeros(1, 3, 1),
            old_logprobs=torch.zeros(1, 3, 1),
            advantages=torch.zeros(1),
            response_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_uninavid_token_loss_inputs_convert_loss_tensors_to_float32():
    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=torch.zeros(1, 2, 1, dtype=torch.float16),
        old_logprobs=torch.ones(1, 2, 1, dtype=torch.bfloat16),
        advantages=torch.ones(1, dtype=torch.float64),
        response_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    assert prepared["logprobs"].dtype == torch.float32
    assert prepared["old_logprobs"].dtype == torch.float32
    assert prepared["advantages"].dtype == torch.float32


def test_uninavid_token_loss_inputs_reject_entropy_shape_mismatch():
    with pytest.raises(ValueError, match="entropy"):
        prepare_uninavid_token_level_loss_inputs(
            logprobs=torch.zeros(1, 3, 1),
            old_logprobs=torch.zeros(1, 3, 1),
            advantages=torch.zeros(1),
            response_mask=torch.ones(1, 3, dtype=torch.bool),
            entropy=torch.zeros(1, 2, 1),
        )


def test_uninavid_token_loss_path_keeps_later_valid_tokens_when_first_sample_masked():
    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=torch.zeros(2, 2, 1),
        old_logprobs=torch.zeros(2, 2, 1),
        advantages=torch.ones(2),
        response_mask=torch.tensor([[0, 0], [1, 1]], dtype=torch.bool),
    )

    loss, _ = compute_grpo_actor_loss_fn(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_func=masked_mean,
        fast_path_zero_loss_mask=False,
        **prepared,
    )
    fast_path_loss, _ = compute_grpo_actor_loss_fn(
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_agg_func=masked_mean,
        fast_path_zero_loss_mask=True,
        **prepared,
    )

    assert loss.item() == pytest.approx(-1.0)
    assert fast_path_loss.item() == pytest.approx(0.0)
