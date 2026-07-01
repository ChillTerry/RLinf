from types import SimpleNamespace

from omegaconf import OmegaConf

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_uninavid_rollout_requests_values_for_gae():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_type": "gae", "loss_type": "actor_critic"},
            "actor": {"model": {"add_value_head": True}},
        }
    )

    worker = SimpleNamespace(cfg=cfg)

    assert MultiStepRolloutWorker._should_calculate_values_for_uninavid(
        worker,
        mode="train",
    )


def test_uninavid_rollout_does_not_request_values_for_grpo():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_type": "grpo", "loss_type": "actor"},
            "actor": {"model": {"add_value_head": False}},
        }
    )

    worker = SimpleNamespace(cfg=cfg)

    assert not MultiStepRolloutWorker._should_calculate_values_for_uninavid(
        worker,
        mode="train",
    )

