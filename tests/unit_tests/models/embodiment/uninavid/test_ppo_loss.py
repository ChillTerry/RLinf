import torch

from rlinf.models.embodiment.uninavid.rl_loss import (
    compute_uninavid_actor_critic_loss,
)
from rlinf.utils.utils import masked_mean


def test_uninavid_actor_critic_loss_uses_token_actor_mask_and_chunk_critic_mask():
    logprobs = torch.zeros(2, 3, 1, dtype=torch.float32, requires_grad=True)
    old_logprobs = torch.zeros_like(logprobs)
    advantages = torch.ones_like(logprobs)
    token_loss_mask = torch.tensor(
        [[[True], [False], [True]], [[True], [True], [False]]]
    )

    values = torch.zeros(2, 1, dtype=torch.float32, requires_grad=True)
    returns = torch.ones(2, 1, dtype=torch.float32)
    prev_values = torch.zeros(2, 1, dtype=torch.float32)
    chunk_loss_mask = torch.tensor([[True], [False]])
    loss_mask_sum = torch.tensor([[1.0], [1.0]])

    loss, metrics = compute_uninavid_actor_critic_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        actor_loss_mask=token_loss_mask,
        values=values,
        returns=returns,
        prev_values=prev_values,
        critic_loss_mask=chunk_loss_mask,
        loss_mask_sum=loss_mask_sum,
        max_episode_steps=2,
        loss_agg_func=masked_mean,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        value_clip=0.2,
        huber_delta=10.0,
        critic_warmup=False,
    )

    loss.backward()

    assert logprobs.grad is not None
    assert values.grad is not None
    assert "actor/policy_loss" in metrics
    assert "critic/value_loss" in metrics

