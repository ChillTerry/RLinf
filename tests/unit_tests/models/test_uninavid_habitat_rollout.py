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

from rlinf.models.embodiment.uninavid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
)
from rlinf.models.embodiment.uninavid.nav_rollout import (
    HABITAT_NAV_ACTION_TO_ID,
    NO_OP_ACTION_ID,
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


def test_uninavid_public_action_mapping_includes_no_op_aliases():
    assert HABITAT_NAV_ACTION_TO_ID["no-op"] == NO_OP_ACTION_ID
    assert HABITAT_NAV_ACTION_TO_ID["no_op"] == NO_OP_ACTION_ID


def test_uninavid_prompt_matches_original_template():
    prompt = build_navigation_prompt("Walk to the kitchen.")
    expected_prompt = (
        "Imagine you are a robot programmed for navigation tasks. You have been given "
        "a video of historical observations and an image of the current observation "
        "<image>. Your assigned task is: 'Walk to the kitchen.'. Analyze this series "
        "of images to determine your next four actions. The predicted action should "
        "be one of the following: forward, left, right, or stop."
    )

    assert prompt == expected_prompt


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


class FakeSequentialBackbone:
    def __init__(self):
        self.feat_cache = None
        self.long_feat_cache = None
        self.weight = 1
        self.new_frames = 0

    def initialize_online_inference_nav_feat_cache(self):
        self.feat_cache = None
        self.long_feat_cache = None
        self.weight = 1
        self.new_frames = 0


class FakeSequentialModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(mm_use_im_start_end=False, run_type="train")
        self.backbone = FakeSequentialBackbone()
        self.generated_text = ["forward left", "right stop"]
        self.generate_calls = []
        self.prompt_updates = []

    def get_model(self):
        return self.backbone

    def update_prompt(self, prompts):
        self.prompt_updates.append(prompts)

    def generate(self, input_ids, images, **kwargs):
        self.generate_calls.append(
            {
                "input_ids_shape": tuple(input_ids.shape),
                "num_images": len(images),
                "new_frames": self.backbone.new_frames,
                "run_type": self.config.run_type,
                "kwargs": kwargs,
            }
        )
        call_index = len(self.generate_calls) - 1
        generated_ids = torch.tensor(
            [[101, 102, call_index]],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat([input_ids, generated_ids], dim=1)


class FakeSequentialTokenizer:
    def __init__(self, texts):
        self.texts = texts
        self.bos_token_id = 1
        self.tokenized_texts = []

    def __call__(self, text, return_tensors=None):
        self.tokenized_texts.append(text)
        token_ids = [1] + [ord(ch) % 100 + 2 for ch in text]
        if return_tensors == "pt":
            token_ids = torch.tensor([token_ids], dtype=torch.long)
        return SimpleNamespace(input_ids=token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        index = int(token_ids[0][-1].item())
        return [self.texts[index]]


class FakeSequentialImageProcessor:
    def preprocess(self, images, return_tensors=None):
        array = np.asarray(images)
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float()
        return {"pixel_values": tensor}


def test_uninavid_predict_action_batch_returns_habitat_chunk_shape():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeSequentialModel()
    policy = UniNaVidForActionPrediction(
        tokenizer=FakeSequentialTokenizer(model.generated_text),
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(rollout_mode="sequential_cache", num_action_chunks=4)
    env_obs = {
        "wrist_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "task_descriptions": ["go to room one", "go to room two"],
        "states": torch.tensor([100, 200]),
    }

    actions, metadata = policy.predict_action_batch(env_obs=env_obs, mode="eval")

    assert actions.shape == (2, 4, 1)
    assert actions.dtype == torch.long
    assert actions[0].squeeze(-1).tolist() == [1, 2, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert actions[1].squeeze(-1).tolist() == [3, 0, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert metadata == empty_rollout_metadata()
    assert len(model.generate_calls) == 2


def test_uninavid_navigation_input_ids_include_image_start_end_tokens_when_enabled():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeSequentialModel()
    model.config.mm_use_im_start_end = True
    tokenizer = FakeSequentialTokenizer(model.generated_text)
    policy = UniNaVidForActionPrediction(
        tokenizer=tokenizer,
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(rollout_mode="sequential_cache", num_action_chunks=4)

    policy._build_navigation_input_ids(build_navigation_prompt("go to room one"))

    assert any(DEFAULT_IM_START_TOKEN in text for text in tokenizer.tokenized_texts)
    assert any(DEFAULT_IM_END_TOKEN in text for text in tokenizer.tokenized_texts)


def test_uninavid_sequential_cache_swaps_slot_state():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeSequentialModel()
    policy = UniNaVidForActionPrediction(
        tokenizer=FakeSequentialTokenizer(model.generated_text),
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(rollout_mode="sequential_cache", num_action_chunks=4)
    policy._nav_caches[0] = UniNaVidNavCache(
        episode_id=100,
        feat_cache=torch.ones(1, 4, 2),
        long_feat_cache=torch.ones(1, 2),
        weight=3,
        new_frames=1,
    )
    env_obs = {
        "wrist_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "task_descriptions": ["go to room one", "go to room two"],
        "states": torch.tensor([100, 200]),
    }

    policy.predict_action_batch(env_obs=env_obs, mode="eval")

    assert policy._nav_caches[0].episode_id == 100
    assert policy._nav_caches[0].feat_cache is not None
    assert policy._nav_caches[0].weight == 3
    assert policy._nav_caches[1].episode_id == 200
    assert policy._nav_caches[1].feat_cache is None
