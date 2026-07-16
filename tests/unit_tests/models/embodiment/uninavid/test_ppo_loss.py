import torch

from rlinf.models.embodiment.uninavid.rl_loss import (
    aggregate_uninavid_weighted_ppo_losses,
    compute_uninavid_actor_critic_loss,
    compute_uninavid_per_chunk_ppo_losses,
    prepare_uninavid_token_level_loss_inputs,
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


def test_uninavid_token_loss_masks_action_tokens_after_executed_prefix():
    prepared = prepare_uninavid_token_level_loss_inputs(
        logprobs=torch.zeros(1, 6, 1),
        old_logprobs=torch.zeros(1, 6, 1),
        advantages=torch.ones(1, 1, 1),
        response_mask=torch.ones(1, 6, dtype=torch.bool),
        action_token_mask=torch.tensor([[False, True, False, True, True, True]]),
        action_token_slot_ids=torch.tensor([[-1, 0, -1, 1, 2, 3]]),
        valid_action_slots=torch.tensor([[True, True, True, False]]),
    )

    assert prepared["loss_mask"].squeeze(-1).tolist() == [
        [False, True, False, True, True, False]
    ]


def _weighted_loss(logprobs, values, weights, advantages, returns):
    per_chunk = compute_uninavid_per_chunk_ppo_losses(
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=advantages.expand_as(logprobs),
        actor_loss_mask=torch.ones_like(logprobs, dtype=torch.bool),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
        values=values,
        returns=returns,
        prev_values=torch.zeros_like(values),
        critic_loss_mask=torch.ones_like(values, dtype=torch.bool),
        value_clip=0.2,
        huber_delta=10.0,
        entropy=torch.full_like(logprobs, 0.25),
    )
    return aggregate_uninavid_weighted_ppo_losses(
        per_chunk,
        train_chunk_weights=weights,
        value_loss_coeff=0.5,
        entropy_bonus=0.01,
    )[0]


def test_weighted_microbatch_sums_match_full_global_batch_loss_and_gradients():
    full_logprobs = torch.tensor([[[0.1], [0.2]], [[-0.1], [0.3]]], requires_grad=True)
    full_values = torch.tensor([[0.2], [0.4]], requires_grad=True)
    weights = torch.tensor([0.6, 0.4])
    advantages = torch.tensor([[[1.0]], [[-0.5]]])
    returns = torch.tensor([[1.0], [0.0]])
    full_loss = _weighted_loss(
        full_logprobs,
        full_values,
        weights,
        advantages,
        returns,
    )
    full_loss.backward()

    split_logprobs = full_logprobs.detach().clone().requires_grad_(True)
    split_values = full_values.detach().clone().requires_grad_(True)
    split_loss = sum(
        _weighted_loss(
            split_logprobs[index : index + 1],
            split_values[index : index + 1],
            weights[index : index + 1],
            advantages[index : index + 1],
            returns[index : index + 1],
        )
        for index in range(2)
    )
    split_loss.backward()

    torch.testing.assert_close(split_loss, full_loss)
    torch.testing.assert_close(split_logprobs.grad, full_logprobs.grad)
    torch.testing.assert_close(split_values.grad, full_values.grad)
