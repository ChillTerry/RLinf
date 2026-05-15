from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.uninavid.nav_rollout import select_slot_rgb_frames
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


def test_uninavid_generate_batched_navigation_outputs_can_return_scores(monkeypatch):
    score_step_1 = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    score_step_2 = torch.tensor([[3.0, 4.0, 5.0]], dtype=torch.float32)
    model = _GenerateModel(
        outputs=SimpleNamespace(
            sequences=torch.tensor([[9, 1, 2]], dtype=torch.long),
            scores=(score_step_1, score_step_2),
        )
    )
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
        lambda cache, visual_features, new_frames: torch.ones((1, 2), dtype=torch.float32),
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
        policy._generate_batched_navigation_outputs(
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
        policy._pad_response_forward_inputs_with_logprobs(
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

    monkeypatch.setattr(policy, "_generate_batched_navigation_outputs", fake_generate_outputs)
    monkeypatch.setattr(policy, "_compute_response_logprobs_from_embeds", fail_recompute)

    actions, metadata = policy._predict_action_batch_with_batched_feature_cache_train(
        env_obs={},
        generation_kwargs={"max_new_tokens": 4},
    )

    assert actions.shape == (1, policy.num_action_chunks, 1)
    assert set(metadata["forward_inputs"]) == {
        "prompt_inputs_embeds",
        "prompt_attention_mask",
        "response_ids",
        "response_mask",
    }
    assert metadata["forward_inputs"]["prompt_inputs_embeds"].shape == (1, 3, 2)
    assert metadata["forward_inputs"]["prompt_attention_mask"].shape == (1, 3)
    assert metadata["forward_inputs"]["response_ids"].tolist() == [[1, 2, 0, 0]]
    assert metadata["forward_inputs"]["response_mask"].tolist() == [
        [True, True, False, False]
    ]

    expected = torch.log_softmax(generated_scores, dim=-1).gather(
        -1,
        response_ids.unsqueeze(-1),
    )
    expected = torch.cat(
        [expected, torch.zeros((1, 2, 1), dtype=torch.float32)],
        dim=1,
    )
    torch.testing.assert_close(metadata["prev_logprobs"], expected)
