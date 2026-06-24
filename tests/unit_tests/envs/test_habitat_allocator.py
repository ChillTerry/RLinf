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
    filter_episodes_by_gt_action_length,
    get_trimmed_episode_count,
    trim_episode_ids,
    trim_episode_records,
)


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

    kept, dropped = filter_episodes_by_gt_action_length(
        episodes,
        gt_data,
        max_action_length=3,
    )

    kept_ids = [
        str(ep["episode_id"]) if isinstance(ep, dict) else ep.episode_id
        for ep in kept
    ]
    assert kept_ids == ["1", "2"]
    assert dropped == ["3"]


def test_filter_episodes_by_gt_action_length_requires_matching_gt():
    with pytest.raises(KeyError, match="Missing GT for episode_id=1"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {},
            max_action_length=3,
        )


def test_filter_episodes_by_gt_action_length_requires_actions_field():
    with pytest.raises(KeyError, match="Missing GT actions for episode_id=1"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {"1": {}},
            max_action_length=3,
        )


def test_filter_episodes_by_gt_action_length_rejects_negative_threshold():
    with pytest.raises(ValueError, match="max_action_length must be non-negative"):
        filter_episodes_by_gt_action_length(
            [{"episode_id": "1"}],
            {"1": {"actions": []}},
            max_action_length=-1,
        )
