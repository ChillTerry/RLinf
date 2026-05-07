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

from __future__ import annotations

from typing import Dict, Optional

import torch


def prepare_uninavid_token_level_loss_inputs(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sample_loss_mask: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if logprobs.dim() != 3 or logprobs.shape[-1] != 1:
        raise ValueError("UniNaVid logprobs must have shape [batch, response_len, 1].")
    if old_logprobs.shape != logprobs.shape:
        raise ValueError("UniNaVid old_logprobs must match logprobs shape.")
    if response_mask.shape != logprobs.shape[:-1]:
        raise ValueError(
            "UniNaVid response_mask must have shape [batch, response_len]."
        )

    logprobs = logprobs.float()
    old_logprobs = old_logprobs.float()
    mask = response_mask.to(torch.bool).unsqueeze(-1)

    if sample_loss_mask is not None:
        sample_loss_mask = sample_loss_mask.to(torch.bool)
        while sample_loss_mask.dim() < mask.dim():
            sample_loss_mask = sample_loss_mask.unsqueeze(-1)
        if sample_loss_mask.shape[1] == 1:
            sample_loss_mask = sample_loss_mask.expand(-1, mask.shape[1], -1)
        if sample_loss_mask.shape != mask.shape:
            raise ValueError(
                "UniNaVid sample_loss_mask must be broadcastable to response_mask."
            )
        mask = mask & sample_loss_mask

    if advantages.dim() == 1:
        advantages = advantages.view(-1, 1, 1)
    elif advantages.dim() == 2:
        advantages = advantages.unsqueeze(-1)
    elif advantages.dim() != 3:
        raise ValueError("UniNaVid advantages must be rank 1, 2, or 3.")

    if advantages.shape[0] != logprobs.shape[0]:
        raise ValueError(
            "UniNaVid advantages batch size must match logprobs batch size."
        )
    if advantages.shape[1] == 1:
        advantages = advantages.expand(-1, logprobs.shape[1], -1)
    if advantages.shape != logprobs.shape:
        raise ValueError("UniNaVid advantages must be broadcastable to logprobs shape.")

    prepared = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages.float(),
        "loss_mask": mask,
    }
    if entropy is not None:
        if entropy.shape != logprobs.shape:
            raise ValueError("UniNaVid entropy must match logprobs shape.")
        prepared["entropy"] = entropy
    return prepared
