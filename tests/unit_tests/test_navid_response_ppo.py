from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import preprocess_loss_inputs
from rlinf.data.embodied_io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models.embodiment.navid.constants import IGNORE_INDEX
from rlinf.models.embodiment.navid.navid_action_model import NaVidForRLActionPrediction
from rlinf.utils.utils import compute_logprobs_from_logits


class _DummyTokenizer:
    pad_token_id = 0


class _DummyImageProcessor:
    pass


class _DummyNaVidModel(nn.Module):
    def __init__(
        self, *, max_seq_len: int = 32, vocab_size: int = 11, hidden_size: int = 8
    ):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.logits_param = nn.Parameter(torch.randn(max_seq_len, vocab_size))
        self.hidden_param = nn.Parameter(torch.randn(max_seq_len, hidden_size))
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = SimpleNamespace(vision_tower=None)
        self.dtype = torch.float32
        self.prompts = None

    def update_prompt(self, prompts=None):
        self.prompts = prompts


def _make_policy(
    *, add_value_head: bool = False, max_history_len: int | None = None
) -> NaVidForRLActionPrediction:
    hidden_size = 8
    return NaVidForRLActionPrediction(
        tokenizer=_DummyTokenizer(),
        model=_DummyNaVidModel(hidden_size=hidden_size),
        image_processor=_DummyImageProcessor(),
        action_dim=1,
        num_action_chunks=3,
        add_value_head=add_value_head,
        hidden_size=hidden_size,
        max_prompt_length=8,
        max_history_len=max_history_len,
    )


def _patch_teacher_forcing_forward(policy: NaVidForRLActionPrediction) -> None:
    def _stub_forward_teacher_forcing_multimodal(
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: torch.Tensor,
        output_hidden_states: bool,
    ):
        del input_ids, attention_mask, pixel_values
        batch_size, seq_len = labels.shape
        logits = (
            policy.model.logits_param[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        )
        hidden = (
            policy.model.hidden_param[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        )
        outputs = SimpleNamespace(
            logits=logits,
            hidden_states=(hidden,) if output_hidden_states else None,
            attentions=None,
            past_key_values=None,
        )
        return outputs, labels

    policy._forward_teacher_forcing_multimodal = (
        _stub_forward_teacher_forcing_multimodal
    )


def test_predict_action_batch_saves_response_fields(monkeypatch):
    policy = _make_policy(add_value_head=False)
    device = next(policy.model.parameters()).device

    monkeypatch.setattr(policy, "preprocess_env_obs", lambda env_obs: env_obs)
    monkeypatch.setattr(
        policy,
        "_build_prompts_and_convs",
        lambda **kwargs: (
            ["prompt_a", "prompt_b"],
            ["question_a", "question_b"],
            [object(), object()],
        ),
    )
    monkeypatch.setattr(policy, "_build_special_token_tensors", lambda **kwargs: None)
    monkeypatch.setattr(
        policy,
        "_tokenize_and_pad_prompts",
        lambda **kwargs: (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device),
            torch.ones((2, 4), dtype=torch.long, device=device),
        ),
    )
    monkeypatch.setattr(
        policy, "_build_stopping_criteria", lambda **kwargs: ("stop", object())
    )
    monkeypatch.setattr(
        policy,
        "_preprocess_new_frames",
        lambda **kwargs: torch.zeros((2, 3, 2, 2), dtype=torch.float32, device=device),
    )
    monkeypatch.setattr(
        policy,
        "_accumulate_history_frames",
        lambda **kwargs: [torch.zeros((1, 3, 2, 2), device=device) for _ in range(2)],
    )
    monkeypatch.setattr(
        policy,
        "_generate",
        lambda **kwargs: SimpleNamespace(
            scores=(
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 1.0], [0.3, 0.1, 0.2, 0.9, 0.4]],
                    device=device,
                ),
                torch.tensor(
                    [[0.3, 0.5, 0.1, 1.2, 0.4], [0.1, 0.2, 1.1, 0.4, 0.5]],
                    device=device,
                ),
                torch.tensor(
                    [[0.1, 0.2, 1.5, 0.3, 0.4], [2.0, 0.1, 0.2, 0.3, 0.4]],
                    device=device,
                ),
            ),
            sequences=torch.tensor(
                [
                    [1, 2, 3, 4, 4, 3, 2],
                    [5, 6, 7, 8, 3, 0, 0],
                ],
                device=device,
            ),
            hidden_states=None,
        ),
    )
    monkeypatch.setattr(
        policy, "_decode_generated_texts", lambda **kwargs: ["forward 25", "stop"]
    )
    monkeypatch.setattr(
        policy,
        "_parse_actions_from_texts",
        lambda **kwargs: np.full((2, 3, 1), "move_forward", dtype="<U12"),
    )

    env_obs = {
        "main_images": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        "task_descriptions": ["task_a", "task_b"],
        "states": torch.tensor([11, 22]),
    }

    _, result = policy.predict_action_batch(env_obs=env_obs, max_new_tokens=5)

    assert result["prev_logprobs"].shape == (2, 5)
    assert result["prev_values"] is None
    assert torch.equal(
        result["forward_inputs"]["response_token_ids"],
        torch.tensor([[4, 3, 2, 0, 0], [3, 0, 0, 0, 0]], device=device),
    )
    assert torch.equal(
        result["forward_inputs"]["response_mask"],
        torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], device=device),
    )
    assert torch.equal(
        result["forward_inputs"]["prompt_lengths"],
        torch.tensor([4, 4], device=device),
    )
    assert torch.equal(
        result["forward_inputs"]["response_lengths"],
        torch.tensor([3, 1], device=device),
    )
    assert torch.allclose(result["prev_logprobs"][0, 3:], torch.zeros(2, device=device))
    assert torch.allclose(result["prev_logprobs"][1, 1:], torch.zeros(4, device=device))


def test_rollout_result_stacks_padded_variable_response_steps(monkeypatch):
    policy = _make_policy(add_value_head=False, max_history_len=4)
    device = next(policy.model.parameters()).device

    monkeypatch.setattr(policy, "preprocess_env_obs", lambda env_obs: env_obs)
    monkeypatch.setattr(
        policy,
        "_build_prompts_and_convs",
        lambda **kwargs: (
            ["prompt_a", "prompt_b"],
            ["question_a", "question_b"],
            [object(), object()],
        ),
    )
    monkeypatch.setattr(policy, "_build_special_token_tensors", lambda **kwargs: None)
    monkeypatch.setattr(
        policy,
        "_tokenize_and_pad_prompts",
        lambda **kwargs: (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device),
            torch.ones((2, 4), dtype=torch.long, device=device),
        ),
    )
    monkeypatch.setattr(
        policy, "_build_stopping_criteria", lambda **kwargs: ("stop", object())
    )
    monkeypatch.setattr(
        policy,
        "_preprocess_new_frames",
        lambda **kwargs: torch.zeros((2, 3, 2, 2), dtype=torch.float32, device=device),
    )

    history_lengths = iter([1, 2])

    def _fake_history(**kwargs):
        history_len = next(history_lengths)
        return [
            torch.full((history_len, 3, 2, 2), fill_value=i + 1, device=device)
            for i in range(2)
        ]

    monkeypatch.setattr(policy, "_accumulate_history_frames", _fake_history)

    generate_outputs = iter(
        [
            SimpleNamespace(
                scores=(
                    torch.tensor(
                        [[0.1, 0.2, 0.3, 0.4, 1.0], [0.3, 0.1, 0.2, 0.9, 0.4]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.3, 0.5, 0.1, 1.2, 0.4], [0.1, 0.2, 1.1, 0.4, 0.5]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.1, 0.2, 1.5, 0.3, 0.4], [2.0, 0.1, 0.2, 0.3, 0.4]],
                        device=device,
                    ),
                ),
                sequences=torch.tensor(
                    [
                        [1, 2, 3, 4, 4, 3, 2],
                        [5, 6, 7, 8, 3, 0, 0],
                    ],
                    device=device,
                ),
                hidden_states=None,
            ),
            SimpleNamespace(
                scores=(
                    torch.tensor(
                        [[0.2, 0.8, 0.3, 0.4, 0.1], [0.4, 0.2, 0.6, 0.1, 0.3]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.9, 0.1, 0.2, 0.3, 0.4], [0.1, 0.7, 0.2, 0.3, 0.4]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.2, 0.4, 0.7, 0.1, 0.3], [0.5, 0.1, 0.2, 0.9, 0.4]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.3, 0.4, 0.2, 1.1, 0.1], [0.2, 0.6, 0.1, 0.3, 0.4]],
                        device=device,
                    ),
                    torch.tensor(
                        [[0.5, 0.1, 1.2, 0.2, 0.3], [0.4, 0.3, 0.8, 0.2, 0.1]],
                        device=device,
                    ),
                ),
                sequences=torch.tensor(
                    [
                        [1, 2, 3, 4, 1, 2, 3, 4, 2],
                        [5, 6, 7, 8, 2, 1, 3, 4, 2],
                    ],
                    device=device,
                ),
                hidden_states=None,
            ),
        ]
    )
    monkeypatch.setattr(policy, "_generate", lambda **kwargs: next(generate_outputs))
    monkeypatch.setattr(
        policy,
        "_decode_generated_texts",
        lambda **kwargs: ["forward 25", "stop"],
    )
    monkeypatch.setattr(
        policy,
        "_parse_actions_from_texts",
        lambda **kwargs: np.full((2, 3, 1), "move_forward", dtype="<U12"),
    )

    env_obs = {
        "main_images": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        "task_descriptions": ["task_a", "task_b"],
        "states": torch.tensor([11, 22]),
    }

    _, result_step1 = policy.predict_action_batch(env_obs=env_obs, max_new_tokens=5)
    _, result_step2 = policy.predict_action_batch(env_obs=env_obs, max_new_tokens=5)

    rollout_result = EmbodiedRolloutResult()
    rollout_result.append_step_result(
        ChunkStepResult(
            prev_logprobs=result_step1["prev_logprobs"],
            forward_inputs=result_step1["forward_inputs"],
        )
    )
    rollout_result.append_step_result(
        ChunkStepResult(
            prev_logprobs=result_step2["prev_logprobs"],
            forward_inputs=result_step2["forward_inputs"],
        )
    )

    trajectory = rollout_result.to_trajectory()

    assert trajectory.prev_logprobs.shape == (2, 2, 5)
    assert trajectory.forward_inputs["response_token_ids"].shape == (2, 2, 5)
    assert trajectory.forward_inputs["response_mask"].shape == (2, 2, 5)
    assert trajectory.forward_inputs["pixel_values"].shape == (2, 2, 4, 3, 2, 2)


def test_predict_action_batch_truncates_to_max_new_tokens(monkeypatch):
    policy = _make_policy(add_value_head=False)
    device = next(policy.model.parameters()).device

    monkeypatch.setattr(policy, "preprocess_env_obs", lambda env_obs: env_obs)
    monkeypatch.setattr(
        policy,
        "_build_prompts_and_convs",
        lambda **kwargs: (
            ["prompt_a", "prompt_b"],
            ["question_a", "question_b"],
            [object(), object()],
        ),
    )
    monkeypatch.setattr(policy, "_build_special_token_tensors", lambda **kwargs: None)
    monkeypatch.setattr(
        policy,
        "_tokenize_and_pad_prompts",
        lambda **kwargs: (
            torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device=device),
            torch.ones((2, 4), dtype=torch.long, device=device),
        ),
    )
    monkeypatch.setattr(
        policy, "_build_stopping_criteria", lambda **kwargs: ("stop", object())
    )
    monkeypatch.setattr(
        policy,
        "_preprocess_new_frames",
        lambda **kwargs: torch.zeros((2, 3, 2, 2), dtype=torch.float32, device=device),
    )
    monkeypatch.setattr(
        policy,
        "_accumulate_history_frames",
        lambda **kwargs: [torch.zeros((1, 3, 2, 2), device=device) for _ in range(2)],
    )
    monkeypatch.setattr(
        policy,
        "_generate",
        lambda **kwargs: SimpleNamespace(
            scores=(
                torch.tensor(
                    [[0.1, 0.2, 0.3, 0.4, 1.0], [0.3, 0.1, 0.2, 0.9, 0.4]],
                    device=device,
                ),
                torch.tensor(
                    [[0.3, 0.5, 0.1, 1.2, 0.4], [0.1, 0.2, 1.1, 0.4, 0.5]],
                    device=device,
                ),
                torch.tensor(
                    [[0.1, 0.2, 1.5, 0.3, 0.4], [2.0, 0.1, 0.2, 0.3, 0.4]],
                    device=device,
                ),
            ),
            sequences=torch.tensor(
                [
                    [1, 2, 3, 4, 4, 3, 2],
                    [5, 6, 7, 8, 3, 2, 1],
                ],
                device=device,
            ),
            hidden_states=None,
        ),
    )
    monkeypatch.setattr(
        policy, "_decode_generated_texts", lambda **kwargs: ["forward 25", "left 60"]
    )
    monkeypatch.setattr(
        policy,
        "_parse_actions_from_texts",
        lambda **kwargs: np.full((2, 3, 1), "move_forward", dtype="<U12"),
    )

    env_obs = {
        "main_images": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)],
        "task_descriptions": ["task_a", "task_b"],
        "states": torch.tensor([11, 22]),
    }

    _, result = policy.predict_action_batch(env_obs=env_obs, max_new_tokens=2)

    assert result["prev_logprobs"].shape == (2, 2)
    assert torch.equal(
        result["forward_inputs"]["response_token_ids"],
        torch.tensor([[4, 3], [3, 2]], device=device),
    )
    assert torch.equal(
        result["forward_inputs"]["response_mask"],
        torch.ones((2, 2), dtype=torch.long, device=device),
    )


def test_default_forward_recomputes_response_token_logprobs_and_entropy():
    policy = _make_policy(add_value_head=True)
    _patch_teacher_forcing_forward(policy)

    batch_size = 2
    prompt_len = 4
    response_len = 3
    forward_inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]),
        "attention_mask": torch.ones((batch_size, prompt_len), dtype=torch.long),
        "pixel_values": torch.zeros((batch_size, 1, 3, 2, 2), dtype=torch.float32),
        "response_token_ids": torch.tensor([[4, 5, 6], [6, 5, 4]], dtype=torch.long),
        "response_mask": torch.ones((batch_size, response_len), dtype=torch.long),
    }

    output = policy.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
    )

    full_input_ids, _, labels = policy._build_teacher_forcing_batch(
        prompt_input_ids=forward_inputs["input_ids"],
        prompt_attention_mask=forward_inputs["attention_mask"],
        response_token_ids=forward_inputs["response_token_ids"],
        response_mask=forward_inputs["response_mask"],
    )
    seq_len = full_input_ids.shape[1]
    logits = policy.model.logits_param[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels.ne(IGNORE_INDEX)
    expected_logprobs = compute_logprobs_from_logits(shift_logits, shift_labels)

    compact_expected = torch.zeros((batch_size, response_len), dtype=torch.float32)
    for batch_idx in range(batch_size):
        valid_values = expected_logprobs[batch_idx][valid_mask[batch_idx]]
        compact_expected[batch_idx, : valid_values.shape[0]] = valid_values

    assert output["logprobs"].shape == (batch_size, response_len)
    assert output["entropy"].shape == (batch_size, response_len)
    assert output["values"].shape == (batch_size, response_len)
    assert torch.allclose(output["logprobs"], compact_expected)


def test_whole_response_chunk_level_smoke():
    policy = _make_policy(add_value_head=False)
    _patch_teacher_forcing_forward(policy)

    batch_size = 16
    response_len = 4
    forward_inputs = {
        "input_ids": torch.arange(batch_size * 5, dtype=torch.long).reshape(
            batch_size, 5
        )
        % 7
        + 1,
        "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
        "pixel_values": torch.zeros((batch_size, 1, 3, 2, 2), dtype=torch.float32),
        "response_token_ids": torch.tensor(
            [[4, 5, 6, 7]] * batch_size, dtype=torch.long
        ),
        "response_mask": torch.ones((batch_size, response_len), dtype=torch.long),
    }

    output = policy.default_forward(
        forward_inputs=forward_inputs, compute_logprobs=True
    )
    old_logprobs = output["logprobs"].detach() - 0.05
    advantages = torch.linspace(
        0.1, 1.6, steps=batch_size, dtype=torch.float32
    ).unsqueeze(-1)
    loss_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    loss_mask_sum = torch.ones((batch_size, 1), dtype=torch.float32)

    processed = preprocess_loss_inputs(
        logprobs=output["logprobs"],
        old_logprobs=old_logprobs,
        advantages=advantages,
        logprob_type="chunk_level",
        single_action_dim=1,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        reward_type="chunk_level",
        values=None,
        prev_values=None,
        returns=None,
    )

    assert processed["logprobs"].shape == (batch_size,)
    assert torch.allclose(processed["logprobs"], output["logprobs"].sum(dim=-1))

    loss, _ = policy_loss(
        loss_type="actor",
        task_type="embodied",
        reward_type="chunk_level",
        logprob_type="chunk_level",
        single_action_dim=1,
        logprobs=output["logprobs"],
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        values=None,
        prev_values=None,
        returns=None,
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        max_episode_steps=3,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.model.logits_param.grad is not None
    assert policy.model.logits_param.grad.abs().sum() > 0


def test_response_token_token_level_smoke():
    policy = _make_policy(add_value_head=False)
    _patch_teacher_forcing_forward(policy)

    batch_size = 16
    response_len = 4
    forward_inputs = {
        "input_ids": torch.arange(batch_size * 5, dtype=torch.long).reshape(
            batch_size, 5
        )
        % 7
        + 1,
        "attention_mask": torch.ones((batch_size, 5), dtype=torch.long),
        "pixel_values": torch.zeros((batch_size, 1, 3, 2, 2), dtype=torch.float32),
        "response_token_ids": torch.tensor(
            [[4, 5, 6, 7]] * batch_size, dtype=torch.long
        ),
        "response_mask": torch.ones((batch_size, response_len), dtype=torch.long),
    }

    output = policy.default_forward(
        forward_inputs=forward_inputs, compute_logprobs=True
    )
    old_logprobs = output["logprobs"].detach() - 0.05
    advantages = torch.linspace(
        0.1, 1.6, steps=batch_size, dtype=torch.float32
    ).unsqueeze(-1)
    loss_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    loss_mask_sum = torch.ones((batch_size, 1), dtype=torch.float32)

    processed = preprocess_loss_inputs(
        logprobs=output["logprobs"],
        old_logprobs=old_logprobs,
        advantages=advantages,
        logprob_type="token_level",
        single_action_dim=1,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        reward_type="chunk_level",
        values=None,
        prev_values=None,
        returns=None,
    )

    assert processed["logprobs"].shape == (batch_size, response_len, 1)
    assert torch.allclose(processed["logprobs"][..., 0], output["logprobs"])
    assert processed["advantages"].shape == (batch_size, 1, 1)
    assert torch.allclose(
        processed["advantages"],
        advantages.view(batch_size, 1, 1),
    )

    loss, _ = policy_loss(
        loss_type="actor",
        task_type="embodied",
        reward_type="chunk_level",
        logprob_type="token_level",
        single_action_dim=1,
        logprobs=output["logprobs"],
        old_logprobs=old_logprobs,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        values=None,
        prev_values=None,
        returns=None,
        clip_ratio_high=0.28,
        clip_ratio_low=0.2,
        max_episode_steps=3,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.model.logits_param.grad is not None
    assert policy.model.logits_param.grad.abs().sum() > 0
