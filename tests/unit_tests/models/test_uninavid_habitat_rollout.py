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
import pytest
import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.models.embodiment.uninavid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
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


def test_uninavid_action_parser_pads_fewer_than_four_actions():
    parsed = parse_uninavid_actions("forward right", 4)

    assert parsed.squeeze(-1).tolist() == [
        HABITAT_NAV_ACTION_TO_ID["forward"],
        HABITAT_NAV_ACTION_TO_ID["right"],
        HABITAT_NAV_ACTION_TO_ID["no-op"],
        HABITAT_NAV_ACTION_TO_ID["no-op"],
    ]


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


def test_uninavid_select_slot_rgb_frames_accepts_chronological_wrist_image_sequence():
    env_obs = {
        "wrist_images": torch.stack(
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


def test_env_output_preserves_chronological_wrist_images_without_history_payload():
    env_output = EnvOutput(
        obs={
            "wrist_images": torch.stack(
                [
                    torch.full((1, 4, 4, 3), 11, dtype=torch.uint8),
                    torch.full((1, 4, 4, 3), 22, dtype=torch.uint8),
                ],
                dim=1,
            ),
            "states": torch.tensor([7]),
            "task_descriptions": ["go"],
        }
    )

    obs_dict = env_output.to_dict()["obs"]

    assert "wrist_images" in obs_dict
    assert obs_dict["wrist_images"].shape == (1, 2, 4, 4, 3)
    assert obs_dict["wrist_images"][0, 0, 0, 0, 0].item() == 11
    assert obs_dict["wrist_images"][0, 1, 0, 0, 0].item() == 22
    assert obs_dict["wrist_images_history"] is None


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
                "run_type": getattr(self.config, "run_type", None),
                "kwargs": kwargs,
            }
        )
        call_index = len(self.generate_calls) - 1
        self.backbone.feat_cache = torch.full((1, 2), call_index + 10.0)
        self.backbone.long_feat_cache = torch.full((1, 1), call_index + 20.0)
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


class FakeBatchedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(512, 3)
        with torch.no_grad():
            values = torch.arange(512 * 3, dtype=torch.float32).view(512, 3)
            self.embed_tokens.weight.copy_(values / 1000)
        self.mm_projector = torch.nn.Identity()
        self.feat_cache = "global-feat-sentinel"
        self.long_feat_cache = "global-long-sentinel"
        self.weight = 99
        self.new_frames = 88


class FakeBatchedVisionTower:
    def __call__(self, images):
        batch_size = images.shape[0]
        base = images.float().mean(dim=(1, 2, 3)).view(batch_size, 1, 1)
        patch_offsets = torch.arange(65, dtype=torch.float32, device=images.device).view(
            1,
            65,
            1,
        )
        channel_offsets = torch.tensor(
            [0.0, 0.25, 0.5],
            dtype=torch.float32,
            device=images.device,
        ).view(1, 1, 3)
        return base + patch_offsets + channel_offsets


class FakeBatchedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(
            compress_type="grid:2",
            mm_use_im_start_end=False,
            mm_vision_select_feature="patch",
            run_type="train",
        )
        self.backbone = FakeBatchedBackbone()
        self.vision_tower = FakeBatchedVisionTower()
        self.generate_calls = []
        self.prompt_updates = []

    def get_model(self):
        return self.backbone

    def get_vision_tower(self):
        return self.vision_tower

    def update_prompt(self, prompts):
        self.prompt_updates.append(prompts)

    def generate(self, *, inputs_embeds, attention_mask, use_cache, **kwargs):
        self.generate_calls.append(
            {
                "inputs_embeds_shape": tuple(inputs_embeds.shape),
                "attention_mask_shape": tuple(attention_mask.shape),
                "attention_mask": attention_mask.detach().clone(),
                "use_cache": use_cache,
                "run_type": getattr(self.config, "run_type", None),
                "kwargs": kwargs,
            }
        )
        return torch.tensor(
            [[10], [11]],
            dtype=torch.long,
            device=inputs_embeds.device,
        )


class FakeBatchedTokenizer(FakeSequentialTokenizer):
    def __init__(self):
        super().__init__(["forward right", "left stop"])

    def __call__(self, text, return_tensors=None):
        self.tokenized_texts.append(text)
        token_ids = [self.bos_token_id] + [ord(ch) % 200 + 2 for ch in text]
        if return_tensors == "pt":
            token_ids = torch.tensor([token_ids], dtype=torch.long)
        return SimpleNamespace(input_ids=token_ids)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        return [self.texts[row_index] for row_index in range(token_ids.shape[0])]


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

    actions, metadata = policy.predict_action_batch(
        env_obs=env_obs,
        mode="eval",
        temperature=0.5,
    )

    assert actions.shape == (2, 4, 1)
    assert actions.dtype == torch.long
    assert actions[0].squeeze(-1).tolist() == [1, 2, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert actions[1].squeeze(-1).tolist() == [3, 0, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert metadata == empty_rollout_metadata()
    assert len(model.generate_calls) == 2
    assert [call["run_type"] for call in model.generate_calls] == ["eval", "eval"]
    assert [call["kwargs"]["use_cache"] for call in model.generate_calls] == [
        True,
        True,
    ]
    assert [call["kwargs"]["temperature"] for call in model.generate_calls] == [
        0.5,
        0.5,
    ]
    assert [call["new_frames"] for call in model.generate_calls] == [
        len(select_slot_rgb_frames(env_obs, 0)),
        len(select_slot_rgb_frames(env_obs, 1)),
    ]
    assert model.prompt_updates == [
        [
            [
                build_navigation_prompt("go to room one")
                .replace(DEFAULT_IMAGE_TOKEN, "")
                .replace("\n", "")
            ]
        ],
        [
            [
                build_navigation_prompt("go to room two")
                .replace(DEFAULT_IMAGE_TOKEN, "")
                .replace("\n", "")
            ]
        ],
    ]


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
    torch.testing.assert_close(
        policy._nav_caches[0].feat_cache,
        torch.full((1, 2), 10.0),
    )
    torch.testing.assert_close(
        policy._nav_caches[0].long_feat_cache,
        torch.full((1, 1), 20.0),
    )
    assert policy._nav_caches[0].weight == 3
    assert policy._nav_caches[1].episode_id == 200
    torch.testing.assert_close(
        policy._nav_caches[1].feat_cache,
        torch.full((1, 2), 11.0),
    )
    torch.testing.assert_close(
        policy._nav_caches[1].long_feat_cache,
        torch.full((1, 1), 21.0),
    )
    assert not torch.equal(
        policy._nav_caches[0].feat_cache,
        policy._nav_caches[1].feat_cache,
    )


def test_uninavid_predict_action_batch_rejects_training_terms():
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

    with pytest.raises(NotImplementedError):
        policy.predict_action_batch(env_obs={}, calculate_logprobs=True)
    with pytest.raises(NotImplementedError):
        policy.predict_action_batch(env_obs={}, calculate_values=True)


def test_uninavid_predict_action_batch_rejects_unknown_rollout_mode():
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
    policy.cfg = SimpleNamespace(rollout_mode="bad_mode", num_action_chunks=4)

    with pytest.raises(ValueError):
        policy.predict_action_batch(env_obs={})


def test_uninavid_batched_feature_cache_uses_single_generate_call(monkeypatch):
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeSequentialModel()
    model.generated_text = ["forward right", "left stop"]
    tokenizer = FakeSequentialTokenizer(model.generated_text)
    policy = UniNaVidForActionPrediction(
        tokenizer=tokenizer,
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(
        rollout_mode="batched_feature_cache",
        num_action_chunks=4,
    )

    def fake_generate_batched_navigation_texts(env_obs, generation_kwargs):
        assert len(env_obs["task_descriptions"]) == 2
        return ["forward right", "left stop"]

    monkeypatch.setattr(
        policy,
        "_generate_batched_navigation_texts",
        fake_generate_batched_navigation_texts,
    )
    env_obs = {
        "wrist_images": torch.zeros(2, 4, 4, 3, dtype=torch.uint8),
        "task_descriptions": ["go to room one", "go to room two"],
        "states": torch.tensor([100, 200]),
    }

    actions, metadata = policy.predict_action_batch(env_obs=env_obs, mode="eval")

    assert actions.shape == (2, 4, 1)
    assert actions[0].squeeze(-1).tolist() == [1, 3, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert actions[1].squeeze(-1).tolist() == [2, 0, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert metadata == empty_rollout_metadata()


def test_uninavid_llava_forward_accepts_inputs_embeds_without_images():
    from rlinf.models.embodiment.uninavid.model.language_model.llava_llama_vid import (
        LlavaConfig,
        LlavaLlamaAttForCausalLM,
    )

    config = LlavaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlavaLlamaAttForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    output = model(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        images=None,
        return_dict=True,
    )

    assert output.logits.shape == (batch_size, seq_len, config.vocab_size)


def test_uninavid_llava_forward_inputs_embeds_fast_path_skips_multimodal_prep(
    monkeypatch,
):
    from rlinf.models.embodiment.uninavid.model.language_model.llava_llama_vid import (
        LlavaConfig,
        LlavaLlamaAttForCausalLM,
    )

    config = LlavaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = LlavaLlamaAttForCausalLM(config)
    model.eval()

    def fail_if_multimodal_prep_runs(*args, **kwargs):
        raise AssertionError("inputs_embeds fast path should bypass multimodal prep")

    monkeypatch.setattr(
        model,
        "prepare_inputs_labels_for_multimodal",
        fail_if_multimodal_prep_runs,
    )

    batch_size = 2
    seq_len = 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    output = model(
        input_ids=None,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        images=None,
        return_dict=True,
    )

    assert output.logits.shape == (batch_size, seq_len, config.vocab_size)


def test_uninavid_llava_generate_accepts_inputs_embeds_without_images():
    from rlinf.models.embodiment.uninavid.model.language_model.llava_llama_vid import (
        LlavaConfig,
        LlavaLlamaAttForCausalLM,
    )

    config = LlavaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlavaLlamaAttForCausalLM(config)
    model.eval()

    batch_size = 2
    seq_len = 5
    inputs_embeds = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    generated = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        images=None,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
    )

    assert generated.dtype == torch.long
    assert generated.shape[0] == batch_size
    assert generated.shape[1] >= 2


def test_uninavid_pad_navigation_embeds_left_pads_shorter_rows():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    policy = UniNaVidForActionPrediction(
        tokenizer=FakeBatchedTokenizer(),
        model=FakeBatchedModel(),
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    short_embed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    long_embed = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    batch, attention_mask = policy._pad_navigation_embeds([short_embed, long_embed])

    assert batch.shape == (2, 3, 2)
    assert attention_mask.tolist() == [[0, 1, 1], [1, 1, 1]]
    torch.testing.assert_close(batch[0, 0], torch.zeros(2))
    torch.testing.assert_close(batch[0, 1:], short_embed)
    torch.testing.assert_close(batch[0, -1], short_embed[-1])
    torch.testing.assert_close(batch[1], long_embed)


def test_uninavid_nav_size_rejects_unsupported_compress_type():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeBatchedModel()
    model.config.compress_type = "unsupported"
    policy = UniNaVidForActionPrediction(
        tokenizer=FakeBatchedTokenizer(),
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="Unsupported Uni-NaVid compress_type"):
        policy._nav_size()


def test_uninavid_batched_feature_cache_real_helper_batches_and_updates_slots():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeBatchedModel()
    tokenizer = FakeBatchedTokenizer()
    policy = UniNaVidForActionPrediction(
        tokenizer=tokenizer,
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(
        rollout_mode="batched_feature_cache",
        num_action_chunks=4,
    )
    env_obs = {
        "wrist_images": torch.stack(
            [
                torch.zeros(4, 4, 3, dtype=torch.uint8),
                torch.full((4, 4, 3), 20, dtype=torch.uint8),
            ]
        ),
        "task_descriptions": ["go", "go to room two"],
        "states": torch.tensor([100, 200]),
    }

    actions, metadata = policy.predict_action_batch(
        env_obs=env_obs,
        mode="eval",
        temperature=0.5,
    )

    assert actions.shape == (2, 4, 1)
    assert actions[0].squeeze(-1).tolist() == [1, 3, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert actions[1].squeeze(-1).tolist() == [2, 0, NO_OP_ACTION_ID, NO_OP_ACTION_ID]
    assert metadata == empty_rollout_metadata()

    assert len(model.generate_calls) == 1
    generate_call = model.generate_calls[0]
    assert generate_call["inputs_embeds_shape"][0] == 2
    assert generate_call["attention_mask_shape"] == generate_call["inputs_embeds_shape"][:2]
    assert generate_call["attention_mask"].shape[0] == 2
    assert generate_call["attention_mask"][0, 0].item() == 0
    assert generate_call["attention_mask"][0, -1].item() == 1
    assert generate_call["attention_mask"][1].tolist() == [
        1
    ] * generate_call["attention_mask"].shape[1]
    assert generate_call["use_cache"] is True
    assert generate_call["run_type"] == "eval"
    assert generate_call["kwargs"]["temperature"] == 0.5
    assert model.config.run_type == "train"

    assert model.prompt_updates == [
        [
            [
                build_navigation_prompt("go")
                .replace(DEFAULT_IMAGE_TOKEN, "")
                .replace("\n", "")
            ],
            [
                build_navigation_prompt("go to room two")
                .replace(DEFAULT_IMAGE_TOKEN, "")
                .replace("\n", "")
            ],
        ]
    ]

    assert set(policy._nav_caches) == {0, 1}
    assert policy._nav_caches[0].episode_id == 100
    assert policy._nav_caches[1].episode_id == 200
    assert policy._nav_caches[0].feat_cache.shape == (1, 4, 3)
    assert policy._nav_caches[1].feat_cache.shape == (1, 4, 3)
    assert not torch.equal(
        policy._nav_caches[0].feat_cache,
        policy._nav_caches[1].feat_cache,
    )
    assert policy._nav_caches[0].new_frames == 1
    assert policy._nav_caches[1].new_frames == 1

    assert model.backbone.feat_cache == "global-feat-sentinel"
    assert model.backbone.long_feat_cache == "global-long-sentinel"
    assert model.backbone.weight == 99
    assert model.backbone.new_frames == 88


def test_uninavid_batched_feature_cache_strictly_restores_absent_run_type():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    model = FakeBatchedModel()
    delattr(model.config, "run_type")
    policy = UniNaVidForActionPrediction(
        tokenizer=FakeBatchedTokenizer(),
        model=model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    policy.cfg = SimpleNamespace(
        rollout_mode="batched_feature_cache",
        num_action_chunks=4,
    )
    env_obs = {
        "wrist_images": torch.stack(
            [
                torch.zeros(4, 4, 3, dtype=torch.uint8),
                torch.full((4, 4, 3), 20, dtype=torch.uint8),
            ]
        ),
        "task_descriptions": ["go to room one", "go to room two"],
        "states": torch.tensor([100, 200]),
    }

    policy.predict_action_batch(env_obs=env_obs, mode="eval")

    assert model.generate_calls[0]["run_type"] == "eval"
    assert not hasattr(model.config, "run_type")


def test_uninavid_sequential_cache_strictly_restores_run_type_state():
    from rlinf.models.embodiment.uninavid.uninavid_action_model import (
        UniNaVidForActionPrediction,
    )

    env_obs = {
        "wrist_images": torch.zeros(1, 4, 4, 3, dtype=torch.uint8),
        "task_descriptions": ["go to room one"],
        "states": torch.tensor([100]),
    }

    none_model = FakeSequentialModel()
    none_model.config.run_type = None
    none_policy = UniNaVidForActionPrediction(
        tokenizer=FakeSequentialTokenizer(none_model.generated_text),
        model=none_model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    none_policy.cfg = SimpleNamespace(
        rollout_mode="sequential_cache",
        num_action_chunks=4,
    )

    none_policy.predict_action_batch(env_obs=env_obs)

    assert none_model.generate_calls[0]["run_type"] == "eval"
    assert none_model.config.run_type is None

    absent_model = FakeSequentialModel()
    delattr(absent_model.config, "run_type")
    absent_policy = UniNaVidForActionPrediction(
        tokenizer=FakeSequentialTokenizer(absent_model.generated_text),
        model=absent_model,
        image_processor=FakeSequentialImageProcessor(),
        torch_dtype=torch.float32,
    )
    absent_policy.cfg = SimpleNamespace(
        rollout_mode="sequential_cache",
        num_action_chunks=4,
    )

    absent_policy.predict_action_batch(env_obs=env_obs)

    assert absent_model.generate_calls[0]["run_type"] == "eval"
    assert not hasattr(absent_model.config, "run_type")


def test_uninavid_process_grid_reduces_patch_grid():
    from rlinf.models.embodiment.uninavid.model.uninavid_arch import (
        process_grid,
    )

    visual = torch.arange(16 * 2, dtype=torch.float32).view(1, 16, 2)

    reduced = process_grid(visual, grid_size=2)

    expected = torch.tensor(
        [
            [
                [5.0, 6.0],
                [9.0, 10.0],
                [21.0, 22.0],
                [25.0, 26.0],
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(reduced, expected)


def test_uninavid_online_nav_cache_compresses_slot_without_global_state():
    from rlinf.models.embodiment.uninavid.model.uninavid_arch import (
        build_navigation_visual_tokens,
        update_online_nav_cache,
    )

    cache = UniNaVidNavCache(episode_id=1)
    update_online_nav_cache(cache, torch.ones(2, 4, 3), new_frames=2)
    update_online_nav_cache(cache, torch.full((1, 4, 3), 2.0), new_frames=1)

    tokens, lengths = build_navigation_visual_tokens(cache, nav_size=4)

    expected_cache = torch.cat(
        [torch.ones(2, 4, 3), torch.full((1, 4, 3), 2.0)],
        dim=0,
    )
    torch.testing.assert_close(cache.feat_cache, expected_cache)
    torch.testing.assert_close(tokens, expected_cache.reshape(-1, 3))
    assert lengths == [4, 4, 4]
    assert cache.new_frames == 1


def test_uninavid_navigation_visual_tokens_compress_long_memory_branch():
    from rlinf.models.embodiment.uninavid.model.uninavid_arch import (
        build_navigation_visual_tokens,
    )

    cache = UniNaVidNavCache(
        episode_id=1,
        feat_cache=torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0]],
                [[0.0, 3.0], [0.0, 3.0]],
            ],
            dtype=torch.float32,
        ),
        long_feat_cache=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        weight=2,
        new_frames=1,
    )

    tokens, lengths = build_navigation_visual_tokens(
        cache,
        nav_size=2,
        length_threshold=2,
    )

    expected_long_cache = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    expected_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [0.0, 3.0],
            [0.0, 3.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(cache.long_feat_cache, expected_long_cache)
    assert cache.weight == 3
    torch.testing.assert_close(tokens, expected_tokens)
    assert lengths == [1, 2, 2]
