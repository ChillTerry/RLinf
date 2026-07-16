import asyncio
from types import MethodType, SimpleNamespace

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EpochTrajectoryBatch, RolloutEpochSpec
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _spec(epoch_index, chunks, weight=1.0):
    return RolloutEpochSpec(
        epoch_index=epoch_index,
        bucket_id=f"B{epoch_index}",
        horizon_steps=chunks * 4,
        n_chunk_steps=chunks,
        curriculum_weight=weight,
        policy_version=0,
    )


def test_rollout_worker_uses_epoch_specific_chunk_counts():
    async def run_epoch(chunks):
        worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
        worker.num_pipeline_stages = 1
        worker.collect_prev_infos = True
        worker.version = 0
        worker._timer_metrics = {}
        worker.cfg = SimpleNamespace(
            actor=SimpleNamespace(
                model=SimpleNamespace(num_action_chunks=4),
            )
        )
        worker.update_dagger_beta = lambda: None
        receive_count = 0
        predict_count = 0
        send_count = 0

        async def recv_env_output(self, _channel):
            nonlocal receive_count
            receive_count += 1
            return {"obs": {"states": torch.tensor([1])}, "final_obs": None}

        def predict(_obs):
            nonlocal predict_count
            predict_count += 1
            return torch.zeros(1, 4, 1), {
                "prev_logprobs": torch.zeros(1, 2, 1),
                "prev_values": torch.zeros(1, 1),
                "forward_inputs": {"action": torch.zeros(1, 4)},
                "expert_label_flag": False,
            }

        def send_rollout_result(_channel, _result, mode="train"):
            nonlocal send_count
            assert mode == "train"
            send_count += 1

        worker.recv_env_output = MethodType(recv_env_output, worker)
        worker.predict = predict
        worker.get_bootstrap_values = lambda _obs: None
        worker.send_rollout_result = send_rollout_result
        await worker.generate_one_epoch(
            None,
            None,
            n_chunk_steps=chunks,
            first_env_outputs=[
                {"obs": {"states": torch.tensor([1])}, "final_obs": None}
            ],
        )
        return receive_count, predict_count, send_count

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = [loop.run_until_complete(run_epoch(chunks)) for chunks in (2, 5, 3)]

    assert results == [(2, 3, 3), (5, 6, 6), (3, 4, 4)]


def _epoch_batch(spec, rank, batch_size=1):
    chunks = spec.n_chunk_steps
    return EpochTrajectoryBatch(
        spec=spec,
        trajectory_ids=[f"rank{rank}-trajectory{index}" for index in range(batch_size)],
        actions=torch.full((chunks, batch_size, 4), rank),
        rewards=torch.zeros(chunks, batch_size, 4),
        terminations=torch.zeros(chunks + 1, batch_size, 4, dtype=torch.bool),
        truncations=torch.zeros(chunks + 1, batch_size, 4, dtype=torch.bool),
        dones=torch.zeros(chunks + 1, batch_size, 4, dtype=torch.bool),
        prev_logprobs=torch.zeros(chunks, batch_size, 4, 1),
        prev_values=torch.zeros(chunks + 1, batch_size, 1),
        versions=torch.zeros(chunks, batch_size, 4, 1),
        forward_inputs={
            "response_mask": torch.ones(chunks, batch_size, 4, dtype=torch.bool),
        },
    )


def test_actor_merge_keeps_variable_epoch_time_boundaries():
    specs = [
        _spec(index, chunks, weight=1 / 3) for index, chunks in enumerate((2, 5, 3))
    ]
    gathered = [[_epoch_batch(spec, rank) for spec in specs] for rank in range(2)]

    merged = EmbodiedFSDPActor._merge_curriculum_actor_epochs(gathered)

    assert [epoch.actions.shape for epoch in merged] == [
        torch.Size([2, 2, 4]),
        torch.Size([5, 2, 4]),
        torch.Size([3, 2, 4]),
    ]
    assert merged[0].actions[:, 0].eq(0).all()
    assert merged[0].actions[:, 1].eq(1).all()


def test_actor_merge_rebalances_unequal_rank_trajectory_counts():
    spec = _spec(0, 2)
    gathered = [
        [_epoch_batch(spec, rank=0, batch_size=2)],
        [_epoch_batch(spec, rank=1, batch_size=1)],
    ]

    merged = EmbodiedFSDPActor._merge_curriculum_actor_epochs(gathered)

    assert merged[0].actions.shape == torch.Size([2, 3, 4])
    assert merged[0].trajectory_ids == [
        "rank0-trajectory0",
        "rank0-trajectory1",
        "rank1-trajectory0",
    ]


def test_actor_curriculum_advantage_compacts_without_fixed_t_reshape(monkeypatch):
    spec = _spec(0, 2)
    epoch = _epoch_batch(spec, rank=0)
    epoch.rewards[:] = 1.0
    epoch.truncations[-1, :, -1] = True
    epoch.dones[-1, :, -1] = True
    epoch.forward_inputs.update(
        {
            "action_token_mask": torch.ones(2, 1, 4, dtype=torch.bool),
            "action_token_slot_ids": torch.arange(4).reshape(1, 1, 4).expand(2, 1, 4),
            "valid_action_slots": torch.ones(2, 1, 4, dtype=torch.bool),
            "response_ids": torch.ones(2, 1, 4, dtype=torch.long),
            "prompt_inputs_embeds": torch.zeros(2, 1, 1, 1),
            "prompt_attention_mask": torch.ones(2, 1, 1, dtype=torch.bool),
            "parsed_action_char_count": torch.ones(2, 1),
            "response_alpha_char_count": torch.ones(2, 1),
            "action": torch.zeros(2, 1, 4),
        }
    )
    actor = EmbodiedFSDPActor.__new__(EmbodiedFSDPActor)
    actor.curriculum_epoch_batches = [epoch]
    actor.curriculum_compacted_batch = None
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "adv_type": "gae",
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "normalize_advantages": False,
                "rollout_epoch": 1,
                "group_size": 1,
            }
        }
    )
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_actor_worker.process_nested_dict_for_adv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixed-T reshape must not run")
        ),
    )

    metrics = actor._compute_curriculum_advantages_and_returns()

    assert actor.curriculum_compacted_batch["advantages"].shape[0] == 2
    assert metrics["curriculum/compaction_keep_ratio"] == 1.0


def test_env_curriculum_rates_are_identical_global_rank_means():
    success, timeout = EnvWorker._mean_curriculum_rates(
        [
            {"success": {"B0": 0.25, "B1": 0.5}, "timeout": {"B0": 0.5}},
            {"success": {"B0": 0.75, "B1": 0.0}, "timeout": {"B0": 0.0}},
        ]
    )

    assert success == {"B0": 0.5, "B1": 0.25}
    assert timeout == {"B0": 0.25}
