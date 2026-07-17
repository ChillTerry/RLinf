import copy

import pytest

from rlinf.envs.habitat.extensions.bucket_scheduler import (
    BucketCurriculumScheduler,
    build_round_robin_bucket_order,
    normalize_bucket_curriculum_plan,
)


def _plan():
    plan = normalize_bucket_curriculum_plan(
        bucket_step_range_map={
            "short": [20, 40],
            "medium": [40, 60],
            "long": [60, 80],
        },
        bucket_max_steps=[40, 80, 120],
        curriculum_stages_map={
            "stage_a": [3, 1, 0],
            "stage_b": [1, 1, 2],
            "stage_c": [2, 0, 2],
            "stage_d": [1, 2, 1],
        },
        rollout_epoch=4,
        curriculum_interval=50,
        effective_num_action_chunks=4,
        bucket_schedule_seed=42,
    )
    for bucket_index, bucket_id in enumerate(plan["bucket_ids"]):
        plan["bucket_plans"][bucket_id]["episode_ids"] = [
            f"{bucket_id}-{index}" for index in range(5 + bucket_index)
        ]
    return plan


def test_normalized_plan_preserves_map_declaration_order():
    plan = _plan()

    assert plan["bucket_ids"] == ["short", "medium", "long"]
    assert [stage["stage_name"] for stage in plan["stages"]] == [
        "stage_a",
        "stage_b",
        "stage_c",
        "stage_d",
    ]
    assert plan["bucket_plans"]["long"]["upper_inclusive"] is True


def test_round_robin_order_exactly_matches_explicit_quota():
    assert build_round_robin_bucket_order(
        ["b1", "b2", "b3", "b4"], [5, 3, 0, 0]
    ) == ["b1", "b2", "b1", "b2", "b1", "b2", "b1", "b1"]


def test_stage_is_derived_only_from_global_step_and_saturates():
    scheduler = BucketCurriculumScheduler(plan=_plan(), total_group_streams=4)

    assert [scheduler.stage_index(step) for step in (49, 50, 99, 100, 149, 150)] == [
        0,
        1,
        1,
        2,
        2,
        3,
    ]
    assert scheduler.stage_index(10_000) == 3
    assert scheduler.active_bucket_ids(100) == ("short", "long")


def test_quota_is_the_only_target_weight_source():
    scheduler = BucketCurriculumScheduler(plan=_plan(), total_group_streams=4)

    assert scheduler.target_weights(0) == {
        "short": 0.75,
        "medium": 0.25,
        "long": 0.0,
    }
    assert scheduler.build_epoch_bucket_order(50) == [
        "short",
        "medium",
        "long",
        "long",
    ]


def test_global_cursor_wraps_and_checkpoint_resume_preserves_next_assignment():
    scheduler = BucketCurriculumScheduler(plan=_plan(), total_group_streams=4)

    assert scheduler.allocate_episode_ids("short") == (
        "short-0",
        "short-1",
        "short-2",
        "short-3",
    )
    state = scheduler.state_dict()
    expected = scheduler.allocate_episode_ids("short")

    restored = BucketCurriculumScheduler(plan=_plan(), total_group_streams=4)
    restored.load_state_dict(state)

    assert expected == ("short-4", "short-0", "short-1", "short-2")
    assert restored.allocate_episode_ids("short") == expected


def test_disabled_bucket_cursor_stays_fixed_until_reenabled():
    scheduler = BucketCurriculumScheduler(plan=_plan(), total_group_streams=4)
    scheduler.allocate_episode_ids("medium")
    cursor = scheduler.bucket_cursors["medium"]

    scheduler.build_epoch_bucket_order(100)

    assert scheduler.bucket_cursors["medium"] == cursor
    scheduler.allocate_episode_ids("medium")
    assert scheduler.bucket_cursors["medium"] != cursor


def test_checkpoint_rejects_changed_episode_order_or_invalid_cursor():
    plan = _plan()
    scheduler = BucketCurriculumScheduler(plan=plan, total_group_streams=4)
    state = scheduler.state_dict()
    changed_plan = copy.deepcopy(plan)
    changed_plan["bucket_plans"]["short"]["episode_ids"].reverse()

    with pytest.raises(ValueError, match="does not match"):
        BucketCurriculumScheduler(
            plan=changed_plan, total_group_streams=4
        ).load_state_dict(state)

    state["bucket_cursors"]["short"] = 5
    with pytest.raises(ValueError, match="outside"):
        scheduler.load_state_dict(state)


def test_cursor_requires_enough_unique_global_stream_episodes():
    plan = _plan()
    plan["bucket_plans"]["short"]["episode_ids"] = ["a", "b", "c"]
    scheduler = BucketCurriculumScheduler(plan=plan, total_group_streams=4)

    with pytest.raises(ValueError, match="4 global group streams"):
        scheduler.allocate_episode_ids("short")
