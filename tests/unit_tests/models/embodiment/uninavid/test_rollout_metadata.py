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

import json
from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.uninavid.nav_rollout import (
    build_action_token_slot_ids,
    select_slot_rgb_frames,
)
from rlinf.models.embodiment.uninavid.uninavid_action_model import (
    UniNaVidForActionPrediction,
)


def test_select_slot_rgb_frames_uses_rgb_frame_history_lengths():
    env_obs = {
        "rgb_frame_history": torch.tensor(
            [
                [[[[1]]], [[[2]]], [[[3]]]],
                [[[[4]]], [[[5]]], [[[6]]]],
            ],
            dtype=torch.uint8,
        ),
        "rgb_frame_history_lengths": torch.tensor([2, 1], dtype=torch.long),
    }

    slot0 = select_slot_rgb_frames(env_obs, 0)
    slot1 = select_slot_rgb_frames(env_obs, 1)

    assert [frame.tolist() for frame in slot0] == [[[[1]]], [[[2]]]]
    assert [frame.tolist() for frame in slot1] == [[[[4]]]]


def test_select_slot_rgb_frames_requires_rgb_frame_history():
    with pytest.raises(KeyError, match="rgb_frame_history"):
        select_slot_rgb_frames({"wrist_images": torch.zeros((1, 1, 1, 1))}, 0)


def test_select_slot_rgb_frames_requires_rgb_frame_history_lengths():
    with pytest.raises(KeyError, match="rgb_frame_history_lengths"):
        select_slot_rgb_frames(
            {"rgb_frame_history": torch.zeros((1, 1, 1, 1, 1), dtype=torch.uint8)},
            0,
        )


def test_action_token_slot_ids_stop_before_synthetic_suffix():
    slot_ids = build_action_token_slot_ids(
        "forward, left, stop, right",
        [99, 6375, 98, 1563, 9847, 1492],
        num_action_chunks=4,
    )

    assert slot_ids == [-1, 0, -1, 1, 2, -1]


class _Tokenizer:
    pad_token_id = 0

    def batch_decode(self, response_ids, skip_special_tokens=True):
        return ["stop" for _ in range(response_ids.shape[0])]


class _GenerateModel:
    def __init__(self, outputs):
        self.config = SimpleNamespace()
        self.outputs = outputs
        self.generate_kwargs = None

    def parameters(self):
        yield torch.zeros((), requires_grad=True)

    def update_prompt(self, prompts):
        self.prompts = prompts

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return self.outputs


def test_uninavid_generation_scores_compute_prev_logprobs_and_response_mask():
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    scores = torch.tensor(
        [
            [[0.0, 2.0, -1.0], [1.0, 0.0, 3.0], [4.0, 0.0, -1.0]],
        ],
        dtype=torch.float32,
    )
    response_ids = torch.tensor([[1, 2, 0]], dtype=torch.long)

    prev_logprobs, response_mask = policy._compute_generation_score_logprobs(
        generated_scores=scores,
        response_ids=response_ids,
    )

    expected = torch.log_softmax(scores, dim=-1).gather(
        -1,
        response_ids.unsqueeze(-1),
    )
    expected = expected * response_mask.unsqueeze(-1)
    torch.testing.assert_close(prev_logprobs, expected)
    assert response_mask.tolist() == [[True, True, False]]
    assert prev_logprobs[0, 2, 0].item() == 0.0


def test_uninavid_response_token_stats_logging_writes_jsonl(tmp_path, monkeypatch):
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    policy.cfg = SimpleNamespace(
        log_response_token_stats=True,
        response_token_stats_path=str(tmp_path / "response_stats.jsonl"),
        response_token_stats_max_samples=10,
    )
    monkeypatch.setenv("RANK", "6")

    env_obs = {
        "states": torch.tensor([101, 102], dtype=torch.long),
        "task_descriptions": ["go left", "go right"],
        "languages": ["en-US", "en-IN"],
    }
    response_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    policy._maybe_log_response_token_stats(
        env_obs=env_obs,
        output_texts=["forward, left", "right, stop"],
        response_mask=response_mask,
        mode="train",
        num_action_chunks=4,
    )
    policy._maybe_log_response_token_stats(
        env_obs=env_obs,
        output_texts=["stop", "forward"],
        response_mask=torch.tensor([[True], [True]]),
        mode="train",
        num_action_chunks=4,
    )

    lines = (
        (tmp_path / "response_stats_rank_6.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = [json.loads(line) for line in lines]

    assert records[0] == {
        "split": None,
        "mode": "train",
        "rank": 6,
        "global_step": None,
        "episode_id": 101,
        "step": 0,
        "slot_id": 0,
        "language": "en-US",
        "response_token_len": 2,
        "parsed_actions": ["forward", "left"],
        "parsed_action_count": 2,
        "response_text": "forward, left",
    }
    assert records[1]["episode_id"] == 102
    assert records[1]["step"] == 0
    assert records[1]["response_token_len"] == 3
    assert records[2]["episode_id"] == 101
    assert records[2]["step"] == 1


def test_uninavid_masked_logprob_gather_skips_negative_infinity_pad_logits():
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    logits = torch.tensor(
        [
            [[0.0, 2.0, -1.0], [0.0, float("-inf"), 3.0]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([[1, 1]], dtype=torch.long)
    mask = torch.tensor([[True, False]])

    token_logprobs = policy._gather_masked_token_logprobs(
        logits=logits,
        target=target,
        mask=mask,
    )
    loss = token_logprobs.sum()
    loss.backward()

    assert torch.isfinite(token_logprobs).all()
    assert token_logprobs[0, 1, 0].item() == 0.0
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1].eq(0).all()


def test_uninavid_generate_batch_outputs_can_return_scores(monkeypatch):
    score_step_1 = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    score_step_2 = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32)
    model = _GenerateModel(
        outputs=SimpleNamespace(
            sequences=torch.tensor([[9, 1, 2]], dtype=torch.long),
            scores=(score_step_1, score_step_2),
        )
    )
    model.config.compress_type = "mean"
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=model,
        image_processor=None,
    )

    monkeypatch.setattr(
        policy,
        "_build_navigation_input_ids",
        lambda navigation_prompt: torch.tensor([[9, 8, 7]], dtype=torch.long),
    )
    monkeypatch.setattr(
        policy,
        "_encode_rgb_frames_for_slot",
        lambda rgb_frames: torch.ones((1, 2, 2), dtype=torch.float32),
    )
    monkeypatch.setattr(
        policy,
        "_update_slot_feature_cache",
        lambda cache, visual_features, new_frames: torch.ones(
            (1, 2), dtype=torch.float32
        ),
    )
    monkeypatch.setattr(
        "rlinf.models.embodiment.uninavid.uninavid_action_model.build_navigation_visual_tokens",
        lambda cache, nav_size: (
            torch.ones((1, 2), dtype=torch.float32),
            [1],
        ),
    )
    monkeypatch.setattr(
        policy,
        "_build_navigation_inputs_embeds",
        lambda input_ids, history_tokens, history_lengths, current_tokens: torch.ones(
            (3, 2),
            dtype=torch.float32,
        ),
    )
    monkeypatch.setattr(
        policy,
        "_pad_navigation_embeds",
        lambda embeds: (
            torch.stack(embeds, dim=0),
            torch.ones((len(embeds), embeds[0].shape[0]), dtype=torch.long),
        ),
    )

    env_obs = {
        "task_descriptions": ["go to the chair"],
        "states": torch.tensor([1], dtype=torch.long),
        "rgb_frame_history": torch.zeros((1, 1, 1, 1, 3), dtype=torch.uint8),
        "rgb_frame_history_lengths": torch.tensor([1], dtype=torch.long),
    }

    output_texts, inputs_embeds, attention_mask, response_ids, generated_scores = (
        policy._generate_batch_outputs(
            env_obs,
            generation_kwargs={"max_new_tokens": 2},
            return_scores=True,
        )
    )

    assert model.generate_kwargs["return_dict_in_generate"] is True
    assert model.generate_kwargs["output_scores"] is True
    assert output_texts == ["stop"]
    assert inputs_embeds.shape == (1, 3, 2)
    assert attention_mask.tolist() == [[1, 1, 1]]
    assert response_ids.tolist() == [[1, 2]]
    torch.testing.assert_close(
        generated_scores,
        torch.stack((score_step_1, score_step_2), dim=1),
    )


def test_uninavid_pads_response_ids_masks_and_generation_logprobs_together():
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    response_ids = torch.tensor([[4, 5]], dtype=torch.long)
    response_mask = torch.tensor([[True, True]])
    prev_logprobs = torch.tensor([[[-0.1], [-0.2]]], dtype=torch.float32)

    padded_ids, padded_mask, padded_logprobs = (
        policy._pad_response_inputs_with_logprobs(
            response_ids=response_ids,
            response_mask=response_mask,
            prev_logprobs=prev_logprobs,
            target_len=4,
        )
    )

    assert padded_ids.tolist() == [[4, 5, 0, 0]]
    assert padded_mask.tolist() == [[True, True, False, False]]
    torch.testing.assert_close(
        padded_logprobs,
        torch.tensor([[[-0.1], [-0.2], [0.0], [0.0]]], dtype=torch.float32),
    )


def test_uninavid_train_rollout_uses_generation_scores_without_recompute(monkeypatch):
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    prompt_inputs_embeds = torch.ones((1, 3, 2), dtype=torch.float32)
    prompt_attention_mask = torch.ones((1, 3), dtype=torch.long)
    response_ids = torch.tensor([[1, 2]], dtype=torch.long)
    generated_scores = torch.tensor(
        [
            [[0.0, 3.0, -2.0], [0.5, -1.0, 2.0]],
        ],
        dtype=torch.float32,
    )

    def fake_generate_outputs(env_obs, generation_kwargs, *, return_scores=False):
        assert return_scores is True
        return (
            ["stop"],
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        )

    def fail_recompute(**kwargs):
        raise AssertionError("post-generation logprob recomputation should not run")

    monkeypatch.setattr(policy, "_generate_batch_outputs", fake_generate_outputs)
    monkeypatch.setattr(policy, "_compute_logits_from_embeds", fail_recompute)

    actions, metadata = policy._predict_train_batch_cached(
        env_obs={},
        generation_kwargs={"max_new_tokens": 4},
    )

    assert actions.shape == (1, policy.num_action_chunks, 1)
    assert set(metadata["forward_inputs"]) == {
        "prompt_inputs_embeds",
        "prompt_attention_mask",
        "response_ids",
        "response_mask",
        "action_token_mask",
        "action_token_slot_ids",
        "action",
        "parsed_action_char_count",
        "response_alpha_char_count",
    }
    assert metadata["forward_inputs"]["prompt_inputs_embeds"].shape == (1, 3, 2)
    assert metadata["forward_inputs"]["prompt_attention_mask"].shape == (1, 3)
    assert metadata["forward_inputs"]["response_ids"].tolist() == [[1, 2, 0, 0]]
    assert metadata["forward_inputs"]["response_mask"].tolist() == [
        [True, True, False, False]
    ]
    assert metadata["forward_inputs"]["action_token_mask"].tolist() == [
        [False, False, False, False]
    ]
    assert metadata["forward_inputs"]["action"].tolist() == [[0, 4, 4, 4]]
    assert metadata["forward_inputs"]["parsed_action_char_count"].tolist() == [4]
    assert metadata["forward_inputs"]["response_alpha_char_count"].tolist() == [4]

    expected = torch.log_softmax(generated_scores, dim=-1).gather(
        -1,
        response_ids.unsqueeze(-1),
    )
    expected = torch.cat(
        [expected, torch.zeros((1, 2, 1), dtype=torch.float32)],
        dim=1,
    )
    torch.testing.assert_close(metadata["prev_logprobs"], expected)


def test_uninavid_train_metadata_rejects_overlong_prompt_by_default(monkeypatch):
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    policy.cfg = SimpleNamespace(model_max_length=5)
    prompt_inputs_embeds = torch.ones((1, 4, 2), dtype=torch.float32)
    prompt_attention_mask = torch.ones((1, 4), dtype=torch.long)
    response_ids = torch.tensor([[1, 2]], dtype=torch.long)
    generated_scores = torch.zeros((1, 2, 3), dtype=torch.float32)

    def fake_generate_outputs(env_obs, generation_kwargs, *, return_scores=False):
        assert return_scores is True
        return (
            ["stop"],
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        )

    monkeypatch.setattr(policy, "_generate_batch_outputs", fake_generate_outputs)

    with pytest.raises(
        ValueError,
        match="prompt length exceeds model_max_length - max_new_tokens",
    ):
        policy._predict_train_batch(
            env_obs={},
            generation_kwargs={"max_new_tokens": 2},
            num_action_chunks=1,
        )


def test_uninavid_train_metadata_can_mask_overlong_prompt_rows(monkeypatch):
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    policy.cfg = SimpleNamespace(
        model_max_length=5,
        drop_overlong_train_metadata=True,
    )
    prompt_inputs_embeds = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    prompt_attention_mask = torch.tensor(
        [
            [0, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    response_ids = torch.tensor([[6375, 5040], [6375, 5040]], dtype=torch.long)
    generated_scores = torch.zeros((2, 2, 7000), dtype=torch.float32)

    def fake_generate_outputs(env_obs, generation_kwargs, *, return_scores=False):
        assert return_scores is True
        return (
            ["forward stop", "forward stop"],
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        )

    monkeypatch.setattr(policy, "_generate_batch_outputs", fake_generate_outputs)

    actions, metadata = policy._predict_train_batch(
        env_obs={},
        generation_kwargs={"max_new_tokens": 2},
        num_action_chunks=1,
    )

    forward_inputs = metadata["forward_inputs"]
    assert actions.shape == (2, 1, 1)
    assert forward_inputs["prompt_inputs_embeds"].shape == (2, 3, 2)
    assert forward_inputs["prompt_attention_mask"].tolist() == [
        [1, 1, 1],
        [1, 1, 1],
    ]
    torch.testing.assert_close(
        forward_inputs["prompt_inputs_embeds"][0],
        torch.tensor(
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        forward_inputs["prompt_inputs_embeds"][1],
        torch.tensor(
            [[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]],
            dtype=torch.float32,
        ),
    )
    assert forward_inputs["response_mask"].tolist() == [
        [True, True],
        [True, True],
    ]
    assert forward_inputs["action_token_mask"].tolist() == [
        [True, False],
        [False, False],
    ]
    assert metadata["prev_logprobs"][1].eq(0).all()


def test_uninavid_train_metadata_records_response_text_diagnostic_counts(
    monkeypatch,
):
    class DiagnosticTokenizer(_Tokenizer):
        def batch_decode(self, response_ids, skip_special_tokens=True):
            return ["forward left ignored", "123 !!!"]

    policy = UniNaVidForActionPrediction(
        tokenizer=DiagnosticTokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    prompt_inputs_embeds = torch.ones((2, 3, 2), dtype=torch.float32)
    prompt_attention_mask = torch.ones((2, 3), dtype=torch.long)
    response_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    generated_scores = torch.zeros((2, 2, 5), dtype=torch.float32)

    def fake_generate_outputs(env_obs, generation_kwargs, *, return_scores=False):
        assert return_scores is True
        return (
            ["forward left ignored", "123 !!!"],
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        )

    monkeypatch.setattr(policy, "_generate_batch_outputs", fake_generate_outputs)

    _actions, metadata = policy._predict_train_batch(
        env_obs={},
        generation_kwargs={"max_new_tokens": 2},
        num_action_chunks=1,
    )

    forward_inputs = metadata["forward_inputs"]
    assert forward_inputs["parsed_action_char_count"].dtype == torch.long
    assert forward_inputs["response_alpha_char_count"].dtype == torch.long
    assert forward_inputs["parsed_action_char_count"].tolist() == [7, 0]
    assert forward_inputs["response_alpha_char_count"].tolist() == [18, 0]
