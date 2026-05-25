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

import math

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.uninavid.rl_loss import (
    compute_uninavid_actor_diagnostic_stats,
    compute_uninavid_actor_diagnostics,
    compute_uninavid_reference_drift_diagnostics,
    finalize_uninavid_actor_diagnostics,
    gather_uninavid_log_ratio_abs_values,
    merge_uninavid_actor_diagnostic_stats,
    prepare_uninavid_token_level_loss_inputs,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def test_prepare_loss_inputs_restricts_mask_to_action_tokens():
    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=torch.zeros((1, 4, 1), dtype=torch.float32),
        old_logprobs=torch.zeros((1, 4, 1), dtype=torch.float32),
        advantages=torch.ones((1, 1, 1), dtype=torch.float32),
        response_mask=torch.tensor([[True, True, True, False]]),
        action_token_mask=torch.tensor([[True, False, True, True]]),
        sample_loss_mask=torch.tensor([[[True], [True], [False], [True]]]),
    )

    assert prepared["loss_mask"].tolist() == [[[True], [False], [False], [False]]]


def test_actor_diagnostics_use_only_masked_token_positions():
    metrics = compute_uninavid_actor_diagnostics(
        logprobs=torch.tensor([[[0.2], [0.0], [-0.1]]], dtype=torch.float32),
        old_logprobs=torch.tensor([[[0.0], [0.5], [-0.4]]], dtype=torch.float32),
        advantages=torch.tensor([[[1.0], [-10.0], [-2.0]]], dtype=torch.float32),
        loss_mask=torch.tensor([[[True], [False], [True]]]),
        response_mask=torch.tensor([[True, True, False]]),
        parsed_action_char_count=torch.tensor([11]),
        response_alpha_char_count=torch.tensor([22]),
        clip_ratio_low=0.1,
        clip_ratio_high=0.2,
    )

    valid_log_ratio = torch.tensor([0.2, 0.3], dtype=torch.float32)
    valid_ratio = valid_log_ratio.exp()
    valid_advantages = torch.tensor([1.0, -2.0], dtype=torch.float32)

    assert metrics["actor/effective_word_ratio"] == 0.5
    assert metrics["actor/response_len_mean"] == 2.0
    assert metrics["actor/response_len_std"] == 0.0
    assert metrics["actor/advantage_abs_mean"] == valid_advantages.abs().mean().item()
    assert metrics["actor/advantage_std"] == valid_advantages.std(unbiased=False).item()
    assert metrics["actor/log_ratio_abs_mean"] == valid_log_ratio.abs().mean().item()
    assert (
        metrics["actor/log_ratio_p95_abs"]
        == torch.quantile(
            valid_log_ratio.abs(),
            0.95,
        ).item()
    )
    assert metrics["actor/approx_kl_k2"] == (0.5 * (valid_log_ratio**2).mean()).item()
    assert metrics["actor/clip_low_fraction"] == 0.0
    assert (
        metrics["actor/clip_high_fraction"]
        == (valid_ratio > 1.0 + 0.2).float().mean().item()
    )
    assert metrics["actor/valid_token_count"] == 2.0


def test_actor_diagnostics_empty_valid_tokens_are_zero_and_non_nan():
    metrics = compute_uninavid_actor_diagnostics(
        logprobs=torch.zeros((1, 2, 1), dtype=torch.float32),
        old_logprobs=torch.zeros((1, 2, 1), dtype=torch.float32),
        advantages=torch.ones((1, 2, 1), dtype=torch.float32),
        loss_mask=torch.zeros((1, 2, 1), dtype=torch.bool),
        response_mask=torch.ones((1, 2), dtype=torch.bool),
        parsed_action_char_count=torch.tensor([7]),
        response_alpha_char_count=torch.tensor([14]),
        clip_ratio_low=0.1,
        clip_ratio_high=0.2,
    )

    for value in metrics.values():
        assert value == 0.0
        assert not math.isnan(value)


def test_actor_diagnostics_advantage_metrics_ignore_masked_positions():
    metrics = compute_uninavid_actor_diagnostics(
        logprobs=torch.zeros((1, 3, 1), dtype=torch.float32),
        old_logprobs=torch.zeros((1, 3, 1), dtype=torch.float32),
        advantages=torch.tensor([[[1.0], [100.0], [3.0]]], dtype=torch.float32),
        loss_mask=torch.tensor([[[True], [False], [True]]]),
        response_mask=torch.tensor([[True, True, True]]),
        parsed_action_char_count=torch.tensor([0]),
        response_alpha_char_count=torch.tensor([10]),
        clip_ratio_low=0.1,
        clip_ratio_high=0.2,
    )

    assert metrics["actor/advantage_abs_mean"] == 2.0
    assert metrics["actor/advantage_std"] == 1.0


def test_actor_diagnostic_stats_merge_before_finalize_avoids_microbatch_average():
    common = {
        "clip_ratio_low": 0.1,
        "clip_ratio_high": 0.2,
    }
    stats_a, ratios_a = compute_uninavid_actor_diagnostic_stats(
        logprobs=torch.tensor([[[0.0], [0.2]]], dtype=torch.float32),
        old_logprobs=torch.zeros((1, 2, 1), dtype=torch.float32),
        advantages=torch.tensor([[[1.0], [3.0]]], dtype=torch.float32),
        loss_mask=torch.tensor([[[True], [True]]]),
        response_mask=torch.tensor([[True, True]]),
        parsed_action_char_count=torch.tensor([7]),
        response_alpha_char_count=torch.tensor([14]),
        **common,
    )
    stats_b, ratios_b = compute_uninavid_actor_diagnostic_stats(
        logprobs=torch.tensor([[[0.4]]], dtype=torch.float32),
        old_logprobs=torch.zeros((1, 1, 1), dtype=torch.float32),
        advantages=torch.tensor([[[5.0]]], dtype=torch.float32),
        loss_mask=torch.tensor([[[True]]]),
        response_mask=torch.tensor([[True]]),
        parsed_action_char_count=torch.tensor([0]),
        response_alpha_char_count=torch.tensor([0]),
        **common,
    )

    metrics = finalize_uninavid_actor_diagnostics(
        merge_uninavid_actor_diagnostic_stats([stats_a, stats_b]),
        torch.cat([ratios_a, ratios_b]),
    )

    assert metrics["actor/valid_token_count"] == 3.0
    assert metrics["actor/advantage_abs_mean"] == 3.0
    assert math.isclose(metrics["actor/log_ratio_abs_mean"], 0.2, rel_tol=1e-6)
    assert math.isclose(metrics["actor/effective_word_ratio"], 0.5)


def test_actor_response_diagnostics_ignore_samples_without_valid_loss_tokens():
    metrics = compute_uninavid_actor_diagnostics(
        logprobs=torch.zeros((2, 2, 1), dtype=torch.float32),
        old_logprobs=torch.zeros((2, 2, 1), dtype=torch.float32),
        advantages=torch.ones((2, 2, 1), dtype=torch.float32),
        loss_mask=torch.tensor([[[True], [True]], [[False], [False]]]),
        response_mask=torch.tensor([[True, True], [True, True]]),
        parsed_action_char_count=torch.tensor([7, 100]),
        response_alpha_char_count=torch.tensor([14, 100]),
        clip_ratio_low=0.1,
        clip_ratio_high=0.2,
    )

    assert metrics["actor/effective_word_ratio"] == 0.5
    assert metrics["actor/response_len_mean"] == 2.0
    assert metrics["actor/valid_token_count"] == 2.0


def test_gather_log_ratio_abs_values_without_dist_returns_flat_values(monkeypatch):
    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.rl_loss.Worker.torch_platform.current_device",
        lambda: torch.device("cpu"),
    )

    values = gather_uninavid_log_ratio_abs_values(torch.tensor([[0.1], [0.2]]))

    torch.testing.assert_close(values, torch.tensor([0.1, 0.2]))


def test_reference_drift_diagnostics_use_only_masked_token_positions():
    metrics = compute_uninavid_reference_drift_diagnostics(
        logprobs=torch.tensor([[[0.2], [0.0], [-0.3]]], dtype=torch.float32),
        ref_logprobs=torch.tensor([[[0.0], [0.4], [-0.1]]], dtype=torch.float32),
        loss_mask=torch.tensor([[[True], [False], [True]]]),
    )

    valid_log_ratio = torch.tensor([0.2, -0.2], dtype=torch.float32)
    valid_ratio = valid_log_ratio.exp()

    assert math.isclose(
        metrics["actor/ref_log_ratio_abs_mean"],
        valid_log_ratio.abs().mean().item(),
        rel_tol=1e-6,
    )
    assert math.isclose(
        metrics["actor/ref_log_ratio_p95_abs"],
        torch.quantile(valid_log_ratio.abs(), 0.95).item(),
        rel_tol=1e-6,
    )
    assert math.isclose(
        metrics["actor/ref_approx_kl_k2"],
        (0.5 * (valid_log_ratio**2).mean()).item(),
        rel_tol=1e-6,
    )
    assert math.isclose(
        metrics["actor/ref_ratio_abs"],
        (valid_ratio - 1.0).abs().mean().item(),
        rel_tol=1e-6,
    )
    assert metrics["actor/ref_valid_token_count"] == 2.0


def test_reference_drift_logprobs_are_requested_only_for_uninavid_habitat():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "env": {"train": {"env_type": "habitat"}},
            "algorithm": {
                "kl_beta": 0.0,
                "reinpp_kl_beta": 0.0,
                "log_reference_drift": True,
            },
        }
    )
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = cfg
    actor.kl_beta = 0.0
    actor.reinpp_kl_beta = 0.0

    assert actor._should_log_uninavid_reference_drift()
    assert actor._should_compute_ref_logprobs()

    actor.cfg.env.train.env_type = "libero"

    assert not actor._should_log_uninavid_reference_drift()
    assert not actor._should_compute_ref_logprobs()


def test_reference_drift_logprobs_are_not_requested_without_env_config():
    cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "uninavid"}},
            "algorithm": {
                "kl_beta": 0.0,
                "reinpp_kl_beta": 0.0,
                "log_reference_drift": True,
            },
        }
    )
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = cfg
    actor.kl_beta = 0.0
    actor.reinpp_kl_beta = 0.0

    assert not actor._should_log_uninavid_reference_drift()
    assert not actor._should_compute_ref_logprobs()
