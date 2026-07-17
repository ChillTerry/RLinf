# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from rlinf.envs.habitat.extensions.allocator import (
    EpisodeRecord,
    build_action_length_buckets,
    filter_episodes_by_gt_action_length,
    get_trimmed_episode_count,
    trim_episode_ids,
    trim_episode_records,
)
from rlinf.envs.habitat.extensions.bucket_scheduler import (
    normalize_bucket_curriculum_plan,
)


def _episode(episode_id):
    return SimpleNamespace(episode_id=str(episode_id))


def test_get_trimmed_episode_count_uses_shared_train_and_eval_rules():
    assert (
        get_trimmed_episode_count(
            17,
            mode="train",
            total_num_processes=2,
            num_group=4,
            total_num_envs=8,
            max_steps_per_rollout_epoch=128,
            max_episode_steps=64,
        )
        == 16
    )
    assert (
        get_trimmed_episode_count(
            50,
            mode="eval",
            total_num_processes=2,
            num_group=4,
            total_num_envs=8,
            max_steps_per_rollout_epoch=128,
            max_episode_steps=64,
        )
        == 16
    )


def test_trim_episode_records_and_ids_share_the_same_count_rule():
    records = [
        EpisodeRecord(episode_id=episode_id, scene_id="scene", weight=1)
        for episode_id in range(17)
    ]
    episode_ids = list(range(17))

    trimmed_records = trim_episode_records(
        records,
        mode="train",
        total_num_processes=2,
        num_group=4,
        total_num_envs=8,
        max_steps_per_rollout_epoch=128,
        max_episode_steps=64,
    )
    trimmed_ids = trim_episode_ids(
        episode_ids,
        mode="train",
        total_num_processes=2,
        num_group=4,
        total_num_envs=8,
        max_steps_per_rollout_epoch=128,
        max_episode_steps=64,
    )

    assert [record.episode_id for record in trimmed_records] == trimmed_ids


def test_filter_episodes_by_gt_action_length_supports_dict_and_object_episodes():
    episodes = [
        {"episode_id": "1"},
        SimpleNamespace(episode_id="2"),
        {"episode_id": "3"},
    ]
    gt_data = {
        "1": {"actions": [0, 1]},
        "2": {"actions": [0, 1, 2]},
        "3": {"actions": [0, 1, 2, 3]},
    }

    result = filter_episodes_by_gt_action_length(
        episodes,
        gt_data,
        min_action_length=3,
        max_action_length=3,
    )

    kept_ids = [
        str(ep["episode_id"]) if isinstance(ep, dict) else ep.episode_id
        for ep in result.episodes
    ]
    assert kept_ids == ["2"]
    assert result.dropped_below_ids == ("1",)
    assert result.dropped_above_ids == ("3",)


def test_filter_episodes_by_gt_action_length_requires_matching_gt():
    with pytest.raises(KeyError, match="Missing GT for episode_id=1"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {},
            min_action_length=0,
            max_action_length=3,
        )


def test_filter_episodes_by_gt_action_length_requires_actions_field():
    with pytest.raises(KeyError, match="Missing GT actions for episode_id=1"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {"1": {}},
            min_action_length=0,
            max_action_length=3,
        )


def test_filter_episodes_by_gt_action_length_rejects_invalid_bounds():
    with pytest.raises(ValueError, match="GT action length bounds"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {"1": {"actions": []}},
            min_action_length=4,
            max_action_length=-1,
        )


def _bucket_plan():
    return normalize_bucket_curriculum_plan(
        bucket_step_range_map={
            "bucket_1": [20, 40],
            "bucket_2": [40, 60],
            "bucket_3": [60, 80],
            "bucket_4": [80, 100],
        },
        bucket_max_steps=[40, 80, 120, 160],
        curriculum_stages_map={"stage": [1, 1, 1, 1]},
        rollout_epoch=4,
        curriculum_interval=50,
        effective_num_action_chunks=4,
        bucket_schedule_seed=42,
    )


def test_explicit_action_length_bucket_boundaries_and_gap_diagnostics():
    lengths = [20, 39, 40, 59, 60, 79, 80, 100]
    episodes = [_episode(index) for index in range(len(lengths))]
    gt_data = {
        str(index): {"actions": [0] * length} for index, length in enumerate(lengths)
    }

    plan = _bucket_plan()
    buckets, gaps = build_action_length_buckets(
        episodes,
        gt_data,
        bucket_ids=plan["bucket_ids"],
        bucket_plans=plan["bucket_plans"],
        bucket_schedule_seed=42,
        minimum_bucket_size=2,
    )

    assert gaps == ()
    assert [set(bucket.episode_ids) for bucket in buckets] == [
        {"0", "1"},
        {"2", "3"},
        {"4", "5"},
        {"6", "7"},
    ]
    assert [(bucket.horizon_steps, bucket.n_chunk_steps) for bucket in buckets] == [
        (40, 10),
        (80, 20),
        (120, 30),
        (160, 40),
    ]


def test_action_length_bucket_rejects_too_few_episodes_without_duplication():
    plan = normalize_bucket_curriculum_plan(
        bucket_step_range_map={"bucket": [0, 10]},
        bucket_max_steps=[40],
        curriculum_stages_map={"stage": [1]},
        rollout_epoch=1,
        curriculum_interval=1,
        effective_num_action_chunks=4,
        bucket_schedule_seed=42,
    )
    with pytest.raises(ValueError, match="4 global group streams"):
        build_action_length_buckets(
            [_episode(0), _episode(1), _episode(2)],
            {
                "0": {"actions": [0]},
                "1": {"actions": [0, 1]},
                "2": {"actions": [0, 1, 2]},
            },
            bucket_ids=plan["bucket_ids"],
            bucket_plans=plan["bucket_plans"],
            bucket_schedule_seed=42,
            minimum_bucket_size=4,
        )


def test_action_length_bucket_reports_gap_and_rejects_empty_bucket():
    plan = normalize_bucket_curriculum_plan(
        bucket_step_range_map={"low": [0, 2], "high": [4, 6]},
        bucket_max_steps=[4, 8],
        curriculum_stages_map={"stage": [1, 0]},
        rollout_epoch=1,
        curriculum_interval=1,
        effective_num_action_chunks=4,
        bucket_schedule_seed=42,
    )
    episodes = [_episode(index) for index in range(3)]
    gt_data = {
        "0": {"actions": [0]},
        "1": {"actions": [0, 0, 0]},
        "2": {"actions": [0, 0, 0, 0]},
    }

    buckets, gaps = build_action_length_buckets(
        episodes,
        gt_data,
        bucket_ids=plan["bucket_ids"],
        bucket_plans=plan["bucket_plans"],
        bucket_schedule_seed=42,
        minimum_bucket_size=1,
    )

    assert [bucket.bucket_id for bucket in buckets] == ["low", "high"]
    assert gaps == (("1", 3),)
