import pytest
import torch

from rlinf.data.embodied_io_struct import EpochTrajectoryBatch, RolloutEpochSpec


def test_rollout_epoch_contract_accepts_variable_time_dimensions():
    batches = []
    for epoch_index, n_chunk_steps in enumerate((2, 5, 3)):
        spec = RolloutEpochSpec(
            epoch_index=epoch_index,
            bucket_id=f"B{epoch_index}",
            horizon_steps=n_chunk_steps * 4,
            n_chunk_steps=n_chunk_steps,
            curriculum_weight=1 / 3,
            policy_version=7,
        )
        batches.append(
            EpochTrajectoryBatch(
                spec=spec,
                trajectory_ids=[f"trajectory-{epoch_index}"],
                actions=torch.zeros(n_chunk_steps, 1, 4),
                rewards=torch.zeros(n_chunk_steps, 1, 4),
                versions=torch.full((n_chunk_steps, 1, 1), 7.0),
            )
        )

    assert [batch.actions.shape[0] for batch in batches] == [2, 5, 3]


def test_epoch_batch_rejects_tensor_time_dimension_different_from_spec():
    spec = RolloutEpochSpec(
        epoch_index=0,
        bucket_id="B0",
        horizon_steps=8,
        n_chunk_steps=2,
        curriculum_weight=1.0,
        policy_version=0,
    )

    with pytest.raises(ValueError, match="spec.n_chunk_steps"):
        EpochTrajectoryBatch(
            spec=spec,
            trajectory_ids=["trajectory-0"],
            actions=torch.zeros(3, 1, 4),
        )
