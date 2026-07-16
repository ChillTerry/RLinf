import pytest
import torch

from rlinf.data.embodied_io_struct import RolloutEpochSpec
from rlinf.models.embodiment.uninavid.curriculum_batch import (
    ProcessedEpochBatch,
    compact_curriculum_epochs,
    compute_grpo_epoch_advantages,
    compute_truncation_aware_gae,
    compute_valid_action_slots,
    materialize_curriculum_rank_batch,
    plan_curriculum_global_batches,
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


def _planner_batch(bucket_ids, trajectory_ids, weights):
    count = len(bucket_ids)
    return {
        "value": torch.arange(count).reshape(count, 1),
        "loss_mask": torch.ones(count, 1, dtype=torch.bool),
        "forward_inputs": {"token": torch.arange(count).reshape(count, 1) + 10},
        "bucket_ids": tuple(bucket_ids),
        "trajectory_ids": tuple(trajectory_ids),
        "chunk_weights": torch.tensor(weights, dtype=torch.float64),
    }


def test_global_planner_mixes_buckets_consumes_once_and_uses_minimum_padding():
    compact = _planner_batch(
        ["B0"] * 5 + ["B1"] * 4,
        ["A", "A", "A", "B", "B", "C", "C", "D", "D"],
        [0.12] * 5 + [0.1] * 4,
    )

    plan = plan_curriculum_global_batches(
        compact,
        global_batch_size=4,
        world_size=2,
        micro_batch_size=1,
        seed=7,
    )

    assert plan.num_updates == 3
    assert plan.alignment_padding_count == 3
    consumed = []
    for global_batch in plan.global_batches:
        real_slots = [slot for slot in global_batch.sample_slots if slot is not None]
        consumed.extend(real_slots)
        assert {compact["bucket_ids"][slot] for slot in real_slots} == {"B0", "B1"}
        assert [len(rank_slots) for rank_slots in global_batch.rank_slots] == [2, 2]
    assert sorted(consumed) == list(range(9))
    assert (
        sum(global_batch.bucket_mass["B0"] for global_batch in plan.global_batches)
        == 0.6
    )
    assert (
        sum(global_batch.bucket_mass["B1"] for global_batch in plan.global_batches)
        == 0.4
    )


def test_global_planner_rejects_bucket_without_one_chunk_per_update():
    compact = _planner_batch(
        ["B0"] * 2 + ["B1"] * 7,
        [f"T{index}" for index in range(9)],
        [0.3] * 2 + [0.4 / 7] * 7,
    )

    with pytest.raises(ValueError, match="B0 has 2 valid chunks"):
        plan_curriculum_global_batches(
            compact,
            global_batch_size=4,
            world_size=2,
            micro_batch_size=1,
            seed=0,
        )


def test_rank_materialization_zeroes_alignment_padding_and_scales_weight():
    compact = _planner_batch(
        ["B0", "B1", "B0"],
        ["A", "B", "C"],
        [0.25, 0.5, 0.25],
    )
    plan = plan_curriculum_global_batches(
        compact,
        global_batch_size=4,
        world_size=2,
        micro_batch_size=1,
        seed=0,
    )

    rank_batches = [
        materialize_curriculum_rank_batch(
            compact,
            training_plan=plan,
            update_index=0,
            rank=rank,
        )
        for rank in range(2)
    ]

    assert [batch["value"].shape[0] for batch in rank_batches] == [2, 2]
    assert (
        sum(int(batch["alignment_padding_mask"].sum()) for batch in rank_batches) == 1
    )
    for batch in rank_batches:
        padding = batch["alignment_padding_mask"]
        assert batch["chunk_weights"][padding].eq(0).all()
        assert batch["train_chunk_weights"][padding].eq(0).all()
        assert batch["loss_mask"][padding].eq(False).all()


@pytest.mark.parametrize(
    ("count", "expected_updates"),
    [(4, 1), (8, 2)],
)
def test_global_planner_handles_exact_global_batch_multiples(count, expected_updates):
    compact = _planner_batch(
        ["B0", "B1"] * (count // 2),
        [f"T{index}" for index in range(count)],
        [1.0 / count] * count,
    )

    plan = plan_curriculum_global_batches(
        compact,
        global_batch_size=4,
        world_size=2,
        micro_batch_size=1,
        seed=0,
    )

    assert plan.num_updates == expected_updates
    assert plan.alignment_padding_count == 0


def test_bucket_queue_round_robins_trajectories_before_exhausting_one():
    compact = _planner_batch(
        ["B0"] * 6 + ["B1"] * 4,
        ["A", "A", "A", "B", "B", "B", "C", "C", "D", "D"],
        [0.1] * 10,
    )
    plan = plan_curriculum_global_batches(
        compact,
        global_batch_size=5,
        world_size=1,
        micro_batch_size=1,
        seed=2,
    )

    b0_trajectory_order = [
        compact["trajectory_ids"][slot]
        for batch in plan.global_batches
        for slot in batch.sample_slots
        if slot is not None and compact["bucket_ids"][slot] == "B0"
    ]
    assert b0_trajectory_order in (
        ["A", "B", "A", "B", "A", "B"],
        ["B", "A", "B", "A", "B", "A"],
    )
