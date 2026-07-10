import math
import warnings

import pytest
import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss
from rlinf.utils.metric_utils import compute_rollout_metrics
from rlinf.utils.utils import masked_mean_ratio


class _CpuPlatform:
    @staticmethod
    def current_device():
        return torch.device("cpu")


def test_critic_value_clip_ratio_uses_raw_delta_and_valid_mask():
    _, metrics = compute_ppo_critic_loss(
        values=torch.tensor([[0.3], [100.0]], dtype=torch.float32),
        returns=torch.zeros((2, 1), dtype=torch.float32),
        prev_values=torch.zeros((2, 1), dtype=torch.float32),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.tensor([[True], [False]]),
    )

    assert metrics["critic/value_clip_ratio"].item() == 1.0
    assert metrics["critic/value_delta_abs_mean"].item() == pytest.approx(0.3)
    assert metrics["critic/value_delta_abs_p95"].item() == pytest.approx(0.3)


def test_critic_explained_variance_uses_biased_var_without_singleton_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, metrics = compute_ppo_critic_loss(
            values=torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            returns=torch.tensor([[1.0], [3.0]], dtype=torch.float32),
            prev_values=torch.zeros((2, 1), dtype=torch.float32),
            value_clip=0.2,
            huber_delta=10.0,
            loss_mask=torch.tensor([[True], [False]]),
        )

    assert math.isnan(metrics["critic/explained_variance"].item())
    assert not any("degrees of freedom" in str(warning.message) for warning in caught)


def test_actor_fast_path_zero_loss_mask_checks_entire_mask():
    logprobs = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=torch.tensor([[0.0], [2.0]], dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_mask=torch.tensor([[False], [True]]),
        fast_path_zero_loss_mask=True,
    )

    assert loss.item() == pytest.approx(-2.0)
    assert metrics["actor/policy_loss"].item() == pytest.approx(-2.0)


def test_rollout_metrics_ignore_invalid_padded_positions(monkeypatch):
    import rlinf.utils.metric_utils as metric_utils

    monkeypatch.setattr(metric_utils.Worker, "torch_platform", _CpuPlatform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, op=None: tensor)

    metrics = compute_rollout_metrics(
        {
            "rewards": torch.tensor([[[1.0, 100.0], [3.0, 300.0]]]),
            "advantages": torch.tensor([[[2.0], [200.0]]]),
            "returns": torch.tensor([[[4.0], [400.0]]]),
            "loss_mask": torch.tensor([[[True, False], [False, False]]]),
        }
    )

    assert metrics["rewards"] == pytest.approx(1.0)
    assert metrics["advantages_mean"] == pytest.approx(2.0)
    assert metrics["advantages_max"] == pytest.approx(2.0)
    assert metrics["advantages_min"] == pytest.approx(2.0)
    assert metrics["returns_mean"] == pytest.approx(4.0)
    assert metrics["returns_max"] == pytest.approx(4.0)
    assert metrics["returns_min"] == pytest.approx(4.0)


def test_masked_mean_ratio_is_finite_when_ratio_is_zero_on_masked_positions():
    result = masked_mean_ratio(
        values=torch.tensor([1.0, 2.0]),
        mask=torch.tensor([False, True]),
        loss_mask_ratio=torch.tensor([0.0, 2.0]),
    )

    assert torch.isfinite(result)
    assert result.item() == pytest.approx(0.5)
