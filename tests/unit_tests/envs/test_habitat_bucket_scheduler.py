import pytest

from rlinf.envs.habitat.extensions.bucket_scheduler import (
    BucketCurriculumScheduler,
    CurriculumStage,
    validate_curriculum_stages,
)


def _scheduler(**kwargs):
    defaults = {
        "bucket_ids": ("B0", "B1", "B2"),
        "stages": (CurriculumStage(3, (0.63, 0.27, 0.10)),),
        "rollout_epoch": 4,
        "curriculum_interval": 10,
        "seed": 42,
    }
    defaults.update(kwargs)
    return BucketCurriculumScheduler(**defaults)


def test_quota_assigns_every_active_bucket_and_uses_largest_deficit():
    scheduler = _scheduler()

    assert scheduler.allocate_epoch_quotas() == {"B0": 2, "B1": 1, "B2": 1}


def test_long_term_quota_tracks_feasible_weights():
    scheduler = _scheduler(
        stages=(CurriculumStage(3, (0.5, 0.3, 0.2)),),
        rollout_epoch=10,
    )
    totals = dict.fromkeys(scheduler.active_bucket_ids, 0)

    for _ in range(100):
        for bucket_id, count in scheduler.allocate_epoch_quotas().items():
            totals[bucket_id] += count

    assert totals == {"B0": 500, "B1": 300, "B2": 200}


def test_checkpoint_resume_preserves_future_schedule():
    scheduler = _scheduler()
    scheduler.build_epoch_bucket_order()
    scheduler.record_bucket_metrics(success_rates={"B2": 0.25})
    state = scheduler.state_dict()
    expected = [scheduler.build_epoch_bucket_order() for _ in range(5)]

    restored = _scheduler()
    restored.load_state_dict(state)

    assert [restored.build_epoch_bucket_order() for _ in range(5)] == expected
    assert restored.state_dict()["success"] == {"B2": 0.25}


def test_stage_with_more_buckets_than_rollout_epochs_fails():
    with pytest.raises(ValueError, match="exceeding rollout_epoch"):
        validate_curriculum_stages(
            (CurriculumStage(3, (0.5, 0.3, 0.2)),), rollout_epoch=2
        )
