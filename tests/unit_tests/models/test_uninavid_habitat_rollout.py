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

import numpy as np
import torch

from rlinf.models.embodiment.uninavid.nav_rollout import (
    NO_OP_ACTION_ID,
    UNINAVID_NAV_PROMPT_TEMPLATE,
    UniNaVidNavCache,
    build_navigation_prompt,
    empty_rollout_metadata,
    get_slot_cache,
    parse_uninavid_actions,
    select_slot_rgb_frames,
)


def test_uninavid_nav_cache_resets_when_episode_id_changes():
    caches = {
        0: UniNaVidNavCache(
            episode_id=10,
            feat_cache=torch.ones(2, 4, 3),
            long_feat_cache=torch.ones(1, 3),
            weight=5,
            new_frames=2,
        ),
        1: UniNaVidNavCache(
            episode_id=20,
            feat_cache=torch.full((1, 4, 3), 7.0),
            long_feat_cache=None,
            weight=2,
            new_frames=1,
        ),
    }

    changed = get_slot_cache(caches, slot_id=0, episode_id=11)

    assert changed.episode_id == 11
    assert changed.feat_cache is None
    assert changed.long_feat_cache is None
    assert changed.weight == 1
    assert changed.new_frames == 0
    torch.testing.assert_close(caches[1].feat_cache, torch.full((1, 4, 3), 7.0))
    assert caches[1].episode_id == 20


def test_uninavid_nav_cache_is_slot_isolated():
    caches = {}
    slot_zero = get_slot_cache(caches, slot_id=0, episode_id=1)
    slot_one = get_slot_cache(caches, slot_id=1, episode_id=1)

    slot_zero.feat_cache = torch.ones(1, 4, 2)
    slot_zero.new_frames = 1

    assert slot_one.feat_cache is None
    assert slot_one.new_frames == 0
    assert caches[0] is slot_zero
    assert caches[1] is slot_one


def test_uninavid_action_parser_maps_text_to_habitat_ids():
    parsed = parse_uninavid_actions("Forward, left. right; stop then forward", 4)

    assert parsed.dtype == torch.long
    assert parsed.shape == (4, 1)
    assert parsed.squeeze(-1).tolist() == [1, 2, 3, 0]


def test_uninavid_action_parser_pads_unknown_and_after_stop():
    parsed = parse_uninavid_actions("spin forward stop left", 4)

    assert parsed.squeeze(-1).tolist() == [1, 0, NO_OP_ACTION_ID, NO_OP_ACTION_ID]


def test_uninavid_prompt_matches_original_template():
    prompt = build_navigation_prompt("Walk to the kitchen.")

    assert prompt == UNINAVID_NAV_PROMPT_TEMPLATE.format("Walk to the kitchen.")


def test_uninavid_select_slot_rgb_frames_prefers_chunk_history():
    env_obs = {
        "wrist_images": torch.zeros(2, 3, 4, 4, 3, dtype=torch.uint8),
        "wrist_images_history": torch.stack(
            [
                torch.full((2, 4, 4, 3), 11, dtype=torch.uint8),
                torch.full((2, 4, 4, 3), 22, dtype=torch.uint8),
            ],
            dim=1,
        ),
    }

    frames = select_slot_rgb_frames(env_obs, slot_id=1)

    assert len(frames) == 2
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert frames[0].shape == (4, 4, 3)
    assert frames[0][0, 0, 0] == 11
    assert frames[1][0, 0, 0] == 22


def test_uninavid_empty_rollout_metadata_has_no_training_terms():
    metadata = empty_rollout_metadata()

    assert metadata == {
        "prev_logprobs": None,
        "prev_values": None,
        "forward_inputs": {},
    }
