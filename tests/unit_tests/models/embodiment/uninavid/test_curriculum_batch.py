import torch

from rlinf.data.embodied_io_struct import RolloutEpochSpec
from rlinf.models.embodiment.uninavid.curriculum_batch import (
    ProcessedEpochBatch,
    compact_curriculum_epochs,
    compute_grpo_epoch_advantages,
    compute_truncation_aware_gae,
    compute_valid_action_slots,
)


def _spec(epoch, bucket, chunks, weight):
    return RolloutEpochSpec(
        epoch_index=epoch,
        bucket_id=bucket,
        horizon_steps=chunks * 4,
        n_chunk_steps=chunks,
        curriculum_weight=weight,
        policy_version=3,
    )


def test_valid_action_slots_keep_terminal_action_and_drop_suffix():
    dones = torch.tensor(
        [
            [False, False, True, False],
            [False, False, False, False],
        ]
    )

    assert compute_valid_action_slots(dones).tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_termination_gae_does_not_bootstrap_true_terminal():
    advantages, returns = compute_truncation_aware_gae(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.5], [0.6], [10.0]]),
        terminations=torch.tensor([[False], [False], [True]]),
        dones=torch.tensor([[False], [False], [True]]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[2.5], [1.4]]))
    torch.testing.assert_close(returns, torch.tensor([[3.0], [2.0]]))


def test_truncation_gae_bootstraps_final_value_but_stops_epoch_recursion():
    advantages, returns = compute_truncation_aware_gae(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.5], [0.6], [10.0]]),
        terminations=torch.tensor([[False], [False], [False]]),
        dones=torch.tensor([[False], [False], [True]]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[12.5], [11.4]]))
    torch.testing.assert_close(returns, torch.tensor([[13.0], [12.0]]))


def test_grpo_advantage_uses_complete_groups_and_valid_prefix_scores():
    rewards = torch.tensor(
        [
            [[1.0], [1.0], [2.0], [2.0]],
            [[9.0], [2.0], [9.0], [4.0]],
        ]
    )
    valid = torch.tensor([[True, True, True, True], [False, True, False, True]])

    advantages = compute_grpo_epoch_advantages(
        rewards=rewards,
        valid_chunk_mask=valid,
        group_size=2,
    )

    expected = torch.tensor([-0.7071063, 0.7071063, -0.7071066, 0.7071066])
    torch.testing.assert_close(advantages[0, :, 0], expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(advantages[0], advantages[1])


def test_compact_removes_padding_and_preserves_nested_field_alignment_and_mass():
    short_values = torch.tensor(
        [
            [[0], [1]],
            [[2], [3]],
            [[4], [5]],
        ]
    )
    short = ProcessedEpochBatch(
        spec=_spec(0, "B0", 3, 0.6),
        trajectory_ids=("A", "B"),
        fields={
            "value": short_values,
            "forward_inputs": {"token": short_values + 10},
        },
        valid_chunk_mask=torch.tensor([[True, True], [True, True], [False, True]]),
    )
    long_values = torch.arange(4).reshape(4, 1, 1) + 100
    long = ProcessedEpochBatch(
        spec=_spec(1, "B1", 4, 0.4),
        trajectory_ids=("C",),
        fields={
            "value": long_values,
            "forward_inputs": {"token": long_values + 10},
        },
        valid_chunk_mask=torch.ones(4, 1, dtype=torch.bool),
    )

    compact = compact_curriculum_epochs((short, long))

    assert compact["value"].flatten().tolist() == [0, 1, 2, 3, 5, 100, 101, 102, 103]
    assert compact["forward_inputs"]["token"].flatten().tolist() == [
        10,
        11,
        12,
        13,
        15,
        110,
        111,
        112,
        113,
    ]
    assert compact["trajectory_ids"] == ("A", "B", "A", "B", "B", "C", "C", "C", "C")
    weights = compact["chunk_weights"]
    torch.testing.assert_close(
        weights[:5], torch.tensor([0.15, 0.1, 0.15, 0.1, 0.1], dtype=torch.float64)
    )
    torch.testing.assert_close(weights[5:], torch.full((4,), 0.1, dtype=torch.float64))
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0, dtype=torch.float64))
