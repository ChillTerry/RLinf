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

from typing import Optional

import torch

from rlinf.scheduler import Worker


def prepare_uninavid_token_level_loss_inputs(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    sample_loss_mask: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
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


def _zero_actor_diagnostics() -> dict[str, float]:
    return {
        "actor/effective_word_ratio": 0.0,
        "actor/response_len_mean": 0.0,
        "actor/response_len_std": 0.0,
        "actor/advantage_abs_mean": 0.0,
        "actor/advantage_std": 0.0,
        "actor/log_ratio_abs_mean": 0.0,
        "actor/log_ratio_p95_abs": 0.0,
        "actor/approx_kl_k2": 0.0,
        "actor/clip_low_fraction": 0.0,
        "actor/clip_high_fraction": 0.0,
        "actor/valid_token_count": 0.0,
    }


def _zero_actor_diagnostic_stats(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "response_count": torch.tensor(0.0, device=device),
        "parsed_action_char_sum": torch.tensor(0.0, device=device),
        "response_alpha_char_sum": torch.tensor(0.0, device=device),
        "response_len_sum": torch.tensor(0.0, device=device),
        "response_len_sq_sum": torch.tensor(0.0, device=device),
        "valid_token_count": torch.tensor(0.0, device=device),
        "advantage_abs_sum": torch.tensor(0.0, device=device),
        "advantage_sum": torch.tensor(0.0, device=device),
        "advantage_sq_sum": torch.tensor(0.0, device=device),
        "log_ratio_abs_sum": torch.tensor(0.0, device=device),
        "log_ratio_sq_sum": torch.tensor(0.0, device=device),
        "clip_low_count": torch.tensor(0.0, device=device),
        "clip_high_count": torch.tensor(0.0, device=device),
    }


def compute_uninavid_actor_diagnostic_stats(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    response_mask: torch.Tensor,
    parsed_action_char_count: torch.Tensor,
    response_alpha_char_count: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    with torch.no_grad():
        device = logprobs.device
        stats = _zero_actor_diagnostic_stats(device)
        log_ratio_abs_values = torch.empty(0, dtype=torch.float32, device=device)

        loss_mask = loss_mask.to(torch.bool)
        valid_token_count = loss_mask.sum()
        if valid_token_count.item() == 0:
            return _zero_actor_diagnostic_stats(device), log_ratio_abs_values

        response_mask = response_mask.to(device=device, dtype=torch.bool)
        sample_has_valid_loss = loss_mask.reshape(loss_mask.shape[0], -1).any(dim=1)
        response_len = response_mask.float().sum(dim=1)[sample_has_valid_loss]
        stats["response_count"] = torch.tensor(
            float(response_len.numel()), device=device
        )
        stats["response_len_sum"] = response_len.sum()
        stats["response_len_sq_sum"] = (response_len**2).sum()
        stats["parsed_action_char_sum"] = parsed_action_char_count.to(
            device=device,
            dtype=torch.float32,
        )[sample_has_valid_loss].sum()
        stats["response_alpha_char_sum"] = response_alpha_char_count.to(
            device=device,
            dtype=torch.float32,
        )[sample_has_valid_loss].sum()

        valid_advantages = advantages[loss_mask]
        stats["valid_token_count"] = valid_token_count.float()
        stats["advantage_abs_sum"] = valid_advantages.abs().sum()
        stats["advantage_sum"] = valid_advantages.sum()
        stats["advantage_sq_sum"] = (valid_advantages**2).sum()

        log_ratio = logprobs - old_logprobs
        valid_log_ratio = log_ratio[loss_mask]
        valid_ratio = valid_log_ratio.exp()
        log_ratio_abs_values = valid_log_ratio.abs().float()
        stats["log_ratio_abs_sum"] = log_ratio_abs_values.sum()
        stats["log_ratio_sq_sum"] = (valid_log_ratio**2).sum()
        stats["clip_low_count"] = (valid_ratio < 1.0 - clip_ratio_low).float().sum()
        stats["clip_high_count"] = (valid_ratio > 1.0 + clip_ratio_high).float().sum()
        stats = {key: value.detach() for key, value in stats.items()}
        return stats, log_ratio_abs_values.detach()


def merge_uninavid_actor_diagnostic_stats(
    stats_list: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not stats_list:
        return _zero_actor_diagnostic_stats(Worker.torch_platform.current_device())

    merged = {
        key: torch.stack([stats[key] for stats in stats_list]).sum()
        for key in stats_list[0]
    }
    return merged


def all_reduce_uninavid_actor_diagnostic_stats(
    stats: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = Worker.torch_platform.current_device()
    stats = {key: value.to(device=device) for key, value in stats.items()}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for value in stats.values():
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
    return stats


def gather_uninavid_log_ratio_abs_values(
    log_ratio_abs_values: torch.Tensor,
) -> torch.Tensor:
    device = Worker.torch_platform.current_device()
    values = log_ratio_abs_values.to(device=device, dtype=torch.float32).reshape(-1)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return values

    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor([values.numel()], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, local_size)
    max_size = int(torch.stack(sizes).max().item())
    padded = torch.zeros(max_size, dtype=torch.float32, device=device)
    if values.numel() > 0:
        padded[: values.numel()] = values
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded)
    return torch.cat(
        [tensor[: int(size.item())] for tensor, size in zip(gathered, sizes)],
        dim=0,
    )


def finalize_uninavid_actor_diagnostics(
    stats: dict[str, torch.Tensor | float],
    log_ratio_abs_values: torch.Tensor | None = None,
) -> dict[str, float]:
    metrics = _zero_actor_diagnostics()

    response_alpha_char_sum = float(stats["response_alpha_char_sum"])
    if response_alpha_char_sum > 0.0:
        metrics["actor/effective_word_ratio"] = (
            float(stats["parsed_action_char_sum"]) / response_alpha_char_sum
        )

    response_count = float(stats["response_count"])
    if response_count > 0.0:
        response_len_mean = float(stats["response_len_sum"]) / response_count
        response_len_var = (
            float(stats["response_len_sq_sum"]) / response_count - response_len_mean**2
        )
        metrics["actor/response_len_mean"] = response_len_mean
        metrics["actor/response_len_std"] = max(response_len_var, 0.0) ** 0.5

    valid_token_count = float(stats["valid_token_count"])
    if valid_token_count == 0.0:
        return metrics

    advantage_mean = float(stats["advantage_sum"]) / valid_token_count
    advantage_var = (
        float(stats["advantage_sq_sum"]) / valid_token_count - advantage_mean**2
    )
    metrics["actor/advantage_abs_mean"] = (
        float(stats["advantage_abs_sum"]) / valid_token_count
    )
    metrics["actor/advantage_std"] = max(advantage_var, 0.0) ** 0.5
    metrics["actor/log_ratio_abs_mean"] = (
        float(stats["log_ratio_abs_sum"]) / valid_token_count
    )
    if log_ratio_abs_values is not None and log_ratio_abs_values.numel() > 0:
        metrics["actor/log_ratio_p95_abs"] = torch.quantile(
            log_ratio_abs_values.float(),
            0.95,
        ).item()
    metrics["actor/approx_kl_k2"] = (
        0.5 * float(stats["log_ratio_sq_sum"]) / valid_token_count
    )
    metrics["actor/clip_low_fraction"] = (
        float(stats["clip_low_count"]) / valid_token_count
    )
    metrics["actor/clip_high_fraction"] = (
        float(stats["clip_high_count"]) / valid_token_count
    )
    metrics["actor/valid_token_count"] = valid_token_count
    return metrics


def compute_uninavid_actor_diagnostics(**kwargs) -> dict[str, float]:
    stats, log_ratio_abs_values = compute_uninavid_actor_diagnostic_stats(**kwargs)
    return finalize_uninavid_actor_diagnostics(stats, log_ratio_abs_values)
