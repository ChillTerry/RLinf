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

from typing import Mapping, Optional, Sequence

import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.scheduler import Worker


def _build_uninavid_action_token_mask(
    *,
    logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    action_token_mask: Optional[torch.Tensor] = None,
    sample_loss_mask: Optional[torch.Tensor] = None,
    action_token_slot_ids: Optional[torch.Tensor] = None,
    valid_action_slots: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logprobs.dim() != 3 or logprobs.shape[-1] != 1:
        raise ValueError("UniNaVid logprobs must have shape [batch, response_len, 1].")
    if response_mask.shape != logprobs.shape[:-1]:
        raise ValueError(
            "UniNaVid response_mask must have shape [batch, response_len]."
        )

    mask = response_mask.to(torch.bool).unsqueeze(-1)
    if action_token_mask is not None:
        if action_token_mask.shape != response_mask.shape:
            raise ValueError(
                "UniNaVid action_token_mask must have shape [batch, response_len]."
            )
        mask = mask & action_token_mask.to(torch.bool).unsqueeze(-1)

    if (action_token_slot_ids is None) != (valid_action_slots is None):
        raise ValueError(
            "action_token_slot_ids and valid_action_slots must be provided together."
        )
    if action_token_slot_ids is not None:
        if action_token_slot_ids.shape != response_mask.shape:
            raise ValueError(
                "action_token_slot_ids must have shape [batch, response_len]."
            )
        if (
            valid_action_slots.dim() != 2
            or valid_action_slots.shape[0] != mask.shape[0]
        ):
            raise ValueError(
                "valid_action_slots must have shape [batch, num_action_chunks]."
            )
        slot_ids = action_token_slot_ids.to(device=mask.device, dtype=torch.long)
        slot_in_range = (slot_ids >= 0) & (slot_ids < valid_action_slots.shape[1])
        gathered_slots = valid_action_slots.to(
            device=mask.device, dtype=torch.bool
        ).gather(1, slot_ids.clamp(min=0, max=valid_action_slots.shape[1] - 1))
        mask = mask & (slot_in_range & gathered_slots).unsqueeze(-1)

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
    return mask


def aggregate_uninavid_action_logprobs(
    *,
    logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    action_token_mask: Optional[torch.Tensor] = None,
    sample_loss_mask: Optional[torch.Tensor] = None,
    action_token_slot_ids: Optional[torch.Tensor] = None,
    valid_action_slots: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate action-token log probabilities into joint action chunks."""
    token_mask = _build_uninavid_action_token_mask(
        logprobs=logprobs,
        response_mask=response_mask,
        action_token_mask=action_token_mask,
        sample_loss_mask=sample_loss_mask,
        action_token_slot_ids=action_token_slot_ids,
        valid_action_slots=valid_action_slots,
    )
    chunk_logprobs = torch.where(token_mask, logprobs.float(), 0.0).sum(
        dim=1,
        keepdim=True,
    )
    chunk_loss_mask = token_mask.any(dim=1, keepdim=True)
    return chunk_logprobs, chunk_loss_mask


def prepare_uninavid_token_level_loss_inputs(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    action_token_mask: Optional[torch.Tensor] = None,
    sample_loss_mask: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    action_token_slot_ids: Optional[torch.Tensor] = None,
    valid_action_slots: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    if old_logprobs.shape != logprobs.shape:
        raise ValueError("UniNaVid old_logprobs must match logprobs shape.")

    logprobs = logprobs.float()
    old_logprobs = old_logprobs.float()
    mask = _build_uninavid_action_token_mask(
        logprobs=logprobs,
        response_mask=response_mask,
        action_token_mask=action_token_mask,
        sample_loss_mask=sample_loss_mask,
        action_token_slot_ids=action_token_slot_ids,
        valid_action_slots=valid_action_slots,
    )

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


def prepare_uninavid_chunk_level_loss_inputs(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    action_token_mask: Optional[torch.Tensor] = None,
    sample_loss_mask: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    action_token_slot_ids: Optional[torch.Tensor] = None,
    valid_action_slots: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Prepare PPO inputs for a joint action represented by multiple tokens."""
    if old_logprobs.shape != logprobs.shape:
        raise ValueError("UniNaVid old_logprobs must match logprobs shape.")

    chunk_logprobs, chunk_loss_mask = aggregate_uninavid_action_logprobs(
        logprobs=logprobs,
        response_mask=response_mask,
        action_token_mask=action_token_mask,
        sample_loss_mask=sample_loss_mask,
        action_token_slot_ids=action_token_slot_ids,
        valid_action_slots=valid_action_slots,
    )
    old_chunk_logprobs, old_chunk_loss_mask = aggregate_uninavid_action_logprobs(
        logprobs=old_logprobs,
        response_mask=response_mask,
        action_token_mask=action_token_mask,
        sample_loss_mask=sample_loss_mask,
        action_token_slot_ids=action_token_slot_ids,
        valid_action_slots=valid_action_slots,
    )
    if not torch.equal(chunk_loss_mask, old_chunk_loss_mask):
        raise RuntimeError("UniNaVid new and old chunk loss masks must match.")

    if advantages.dim() == 1:
        advantages = advantages.view(-1, 1, 1)
    elif advantages.dim() == 2:
        advantages = advantages.unsqueeze(-1)
    elif advantages.dim() != 3:
        raise ValueError("UniNaVid advantages must be rank 1, 2, or 3.")
    if advantages.shape != chunk_logprobs.shape:
        raise ValueError(
            "UniNaVid chunk-level advantages must have shape [batch, 1, 1]."
        )

    prepared = {
        "logprobs": chunk_logprobs,
        "old_logprobs": old_chunk_logprobs,
        "advantages": advantages.float(),
        "loss_mask": chunk_loss_mask,
    }
    if entropy is not None:
        if entropy.shape != logprobs.shape:
            raise ValueError("UniNaVid entropy must match logprobs shape.")
        chunk_entropy, _ = aggregate_uninavid_action_logprobs(
            logprobs=entropy,
            response_mask=response_mask,
            action_token_mask=action_token_mask,
            sample_loss_mask=sample_loss_mask,
            action_token_slot_ids=action_token_slot_ids,
            valid_action_slots=valid_action_slots,
        )
        prepared["entropy"] = chunk_entropy
    return prepared


def compute_uninavid_actor_critic_loss(
    *,
    actor_loss_mask: torch.Tensor,
    critic_loss_mask: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    value_loss_coeff: float = 1.0,
    loss_mask_sum: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    actor_loss, actor_metrics = compute_ppo_actor_loss(
        loss_mask=actor_loss_mask,
        **kwargs,
    )
    critic_loss, critic_metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=value_clip,
        huber_delta=huber_delta,
        loss_mask=critic_loss_mask,
        loss_mask_sum=loss_mask_sum,
        max_episode_steps=max_episode_steps,
    )

    metrics = {}
    metrics.update(actor_metrics)
    metrics.update(critic_metrics)
    scaled_critic_loss = float(value_loss_coeff) * critic_loss
    metrics["critic/value_loss_scaled"] = scaled_critic_loss.detach()
    return actor_loss + scaled_critic_loss, metrics


def _per_chunk_masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError("Per-chunk values and mask must have identical shapes.")
    flat_values = values.reshape(values.shape[0], -1)
    flat_mask = mask.to(torch.bool).reshape(mask.shape[0], -1)
    counts = flat_mask.sum(dim=1)
    sums = torch.where(flat_mask, flat_values, 0.0).sum(dim=1)
    return torch.where(counts > 0, sums / counts.clamp_min(1), 0.0)


def compute_uninavid_per_chunk_ppo_losses(
    *,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    actor_loss_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: Optional[float] = None,
    clip_log_ratio_min: Optional[float] = None,
    clip_log_ratio_max: Optional[float] = None,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    prev_values: Optional[torch.Tensor] = None,
    critic_loss_mask: Optional[torch.Tensor] = None,
    value_clip: Optional[float] = None,
    huber_delta: Optional[float] = None,
    entropy: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return one actor/critic/entropy/KL value for every compacted chunk."""
    if not (
        logprobs.shape
        == old_logprobs.shape
        == advantages.shape
        == actor_loss_mask.shape
    ):
        raise ValueError("Weighted UniNaVid actor inputs must have identical shapes.")
    log_ratio = logprobs.float() - old_logprobs.float()
    if clip_log_ratio_min is not None:
        log_ratio = log_ratio.clamp(min=clip_log_ratio_min)
    if clip_log_ratio_max is not None:
        log_ratio = log_ratio.clamp(max=clip_log_ratio_max)
    ratio = torch.exp(log_ratio)
    clipped_ratio = ratio.clamp(1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss = torch.maximum(
        -advantages.float() * ratio, -advantages.float() * clipped_ratio
    )
    if clip_ratio_c is not None:
        if clip_ratio_c <= 1.0:
            raise ValueError("clip_ratio_c must be greater than 1.")
        dual_clip = torch.sign(advantages) * clip_ratio_c * advantages
        policy_loss = torch.minimum(policy_loss, dual_clip)

    per_chunk = {
        "actor_loss": _per_chunk_masked_mean(policy_loss, actor_loss_mask),
        "approx_kl": _per_chunk_masked_mean(-log_ratio, actor_loss_mask),
        "clip_fraction": _per_chunk_masked_mean(
            (ratio != clipped_ratio).float(), actor_loss_mask
        ),
    }
    if entropy is not None:
        per_chunk["entropy"] = _per_chunk_masked_mean(entropy, actor_loss_mask)
    else:
        per_chunk["entropy"] = torch.zeros_like(per_chunk["actor_loss"])

    critic_inputs = (values, returns, prev_values, critic_loss_mask)
    if all(value is None for value in critic_inputs):
        per_chunk["critic_loss"] = torch.zeros_like(per_chunk["actor_loss"])
        return per_chunk
    if any(value is None for value in critic_inputs):
        raise ValueError("Weighted critic inputs must be provided together.")
    if value_clip is None or huber_delta is None:
        raise ValueError("Weighted critic loss requires value_clip and huber_delta.")
    if not (
        values.shape == returns.shape == prev_values.shape == critic_loss_mask.shape
    ):
        raise ValueError("Weighted UniNaVid critic inputs must have identical shapes.")
    clipped_values = prev_values + (values - prev_values).clamp(-value_clip, value_clip)
    value_loss = torch.maximum(
        huber_loss(returns - values, huber_delta),
        huber_loss(returns - clipped_values, huber_delta),
    )
    per_chunk["critic_loss"] = _per_chunk_masked_mean(value_loss, critic_loss_mask)
    return per_chunk


def aggregate_uninavid_weighted_ppo_losses(
    per_chunk_losses: Mapping[str, torch.Tensor],
    *,
    train_chunk_weights: torch.Tensor,
    value_loss_coeff: float = 1.0,
    entropy_bonus: float = 0.0,
    critic_warmup: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Aggregate fixed chunk weights; microbatch losses are intended to be summed."""
    actor_loss = per_chunk_losses["actor_loss"]
    weights = train_chunk_weights.to(device=actor_loss.device, dtype=actor_loss.dtype)
    if weights.shape != actor_loss.shape:
        raise ValueError("train_chunk_weights must have one value per chunk.")
    weighted_actor = (weights * actor_loss).sum()
    if critic_warmup:
        weighted_actor = weighted_actor * 0.0
    weighted_critic = (weights * per_chunk_losses["critic_loss"]).sum()
    weighted_entropy = (weights * per_chunk_losses["entropy"]).sum()
    total_loss = (
        weighted_actor
        + float(value_loss_coeff) * weighted_critic
        - float(entropy_bonus) * weighted_entropy
    )
    metrics = {
        "actor/policy_loss": weighted_actor.detach(),
        "critic/value_loss": weighted_critic.detach(),
        "actor/entropy_loss": weighted_entropy.detach(),
        "actor/approx_kl": (weights * per_chunk_losses["approx_kl"]).sum().detach(),
        "actor/clip_fraction": (weights * per_chunk_losses["clip_fraction"])
        .sum()
        .detach(),
        "actor/total_loss": total_loss.detach(),
        "curriculum/train_weight_mass": weights.sum().detach(),
    }
    return total_loss, metrics


def mean_uninavid_training_metrics(
    metrics: Mapping[str, Sequence[torch.Tensor | float]],
) -> dict[str, float]:
    """Average scalar training metrics without passing CUDA tensors to NumPy."""
    means = {}
    for key, values in metrics.items():
        if not values:
            raise ValueError(f"Metric {key!r} has no values to aggregate.")
        if isinstance(values[0], torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"Metric {key!r} mixes tensor and scalar values.")
            if not all(value.numel() == 1 for value in values):
                raise ValueError(f"Metric {key!r} must contain scalar tensors.")
            means[key] = torch.stack(
                [value.detach().reshape(()) for value in values]
            ).float().mean().item()
        else:
            if any(isinstance(value, torch.Tensor) for value in values):
                raise TypeError(f"Metric {key!r} mixes tensor and scalar values.")
            means[key] = sum(float(value) for value in values) / len(values)
    return means


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


def compute_uninavid_reference_drift_diagnostics(
    *,
    logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, float]:
    if ref_logprobs.shape != logprobs.shape:
        raise ValueError("UniNaVid ref_logprobs must match logprobs shape.")
    if loss_mask.shape != logprobs.shape:
        raise ValueError("UniNaVid reference drift loss_mask must match logprobs.")

    with torch.no_grad():
        loss_mask = loss_mask.to(torch.bool)
        valid_token_count = loss_mask.sum()
        if valid_token_count.item() == 0:
            return {
                "actor/ref_log_ratio_abs_mean": 0.0,
                "actor/ref_log_ratio_p95_abs": 0.0,
                "actor/ref_approx_kl_k2": 0.0,
                "actor/ref_ratio_abs": 0.0,
                "actor/ref_valid_token_count": 0.0,
            }

        log_ratio = logprobs.float() - ref_logprobs.float()
        valid_log_ratio = log_ratio[loss_mask]
        valid_log_ratio_abs = valid_log_ratio.abs()
        valid_ratio = valid_log_ratio.exp()

        return {
            "actor/ref_log_ratio_abs_mean": valid_log_ratio_abs.mean().item(),
            "actor/ref_log_ratio_p95_abs": torch.quantile(
                valid_log_ratio_abs.float(),
                0.95,
            ).item(),
            "actor/ref_approx_kl_k2": (
                0.5 * (valid_log_ratio.float() ** 2).mean()
            ).item(),
            "actor/ref_ratio_abs": (valid_ratio - 1.0).abs().mean().item(),
            "actor/ref_valid_token_count": float(valid_token_count.item()),
        }
