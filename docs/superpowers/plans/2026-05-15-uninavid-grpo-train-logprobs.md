# UniNaVid GRPO Train Logprobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change UniNaVid GRPO train rollout `prev_logprobs` to come from `generate().scores` instead of a post-generation response-logprob forward pass.

**Architecture:** Keep UniNaVid train metadata unchanged. Add train-only handling around batched navigation generation so the train path requests generation scores, computes token logprobs from those processed scores, pads response tensors together, and still stores detached `prompt_inputs_embeds` / `prompt_attention_mask` in `forward_inputs`.

**Tech Stack:** Python, PyTorch, Hugging Face `generate`, RLinf UniNaVid embodied policy, pytest.

---

## File Structure

- Modify `rlinf/models/embodiment/uninavid/uninavid_action_model.py`
  - Import `compute_logprobs_from_logits`.
  - Make `_generate_batched_navigation_outputs` optionally return generation scores for train.
  - Add a small helper to compute `prev_logprobs` from generation scores.
  - Add a small helper to pad `response_ids`, `response_mask`, and `prev_logprobs` together.
  - Update `_predict_action_batch_with_batched_feature_cache_train` to use generation scores.
- Modify `tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py`
  - Add focused unit tests for generation-score logprobs and padding.
  - Add a focused train-path test that proves `_compute_response_logprobs_from_embeds` is not called and metadata stays unchanged.

## Task 1: Add Failing Unit Tests

**Files:**
- Modify: `tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py`

- [ ] **Step 1: Add imports and helper fixtures**

Add these imports below the existing imports:

```python
from types import SimpleNamespace

from rlinf.models.embodiment.uninavid.uninavid_action_model import (
    UniNaVidForActionPrediction,
)
```

Add these helper classes after the existing tests:

```python
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
```

- [ ] **Step 2: Add helper-level test for generation-score logprobs**

Append this test:

```python
def test_uninavid_generation_scores_compute_prev_logprobs_and_response_mask():
    policy = UniNaVidForActionPrediction(
        tokenizer=_Tokenizer(),
        model=_GenerateModel(outputs=None),
        image_processor=None,
    )
    scores = torch.tensor(
        [
            [[0.0, 2.0, -1.0], [1.0, 0.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    response_ids = torch.tensor([[1, 2]], dtype=torch.long)

    prev_logprobs, response_mask = policy._compute_generation_score_logprobs(
        generated_scores=scores,
        response_ids=response_ids,
    )

    expected = torch.log_softmax(scores, dim=-1).gather(
        -1,
        response_ids.unsqueeze(-1),
    )
    torch.testing.assert_close(prev_logprobs, expected)
    assert response_mask.tolist() == [[True, True]]
```

- [ ] **Step 3: Add helper-level test for joint padding**

Append this test:

```python
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
```

- [ ] **Step 4: Add train-path test for metadata and no recompute**

Append this test:

```python
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
```

- [ ] **Step 5: Run tests and verify they fail**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py -q
```

Expected: failure because `_compute_generation_score_logprobs`,
`_pad_response_forward_inputs_with_logprobs`, and the `return_scores` parameter
do not exist yet.

## Task 2: Implement Generation-Score Logprob Helpers

**Files:**
- Modify: `rlinf/models/embodiment/uninavid/uninavid_action_model.py`

- [ ] **Step 1: Import `compute_logprobs_from_logits`**

Add this import with the other RLinf imports:

```python
from rlinf.utils.utils import compute_logprobs_from_logits
```

- [ ] **Step 2: Add generation-score logprob helper**

Add this method near `_build_response_mask`:

```python
    def _compute_generation_score_logprobs(
        self,
        *,
        generated_scores: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if generated_scores.dim() != 3:
            raise ValueError(
                "UniNaVid generation scores must have shape [batch, response_len, vocab]."
            )
        if response_ids.shape != generated_scores.shape[:2]:
            raise ValueError(
                "UniNaVid response ids must match generation score batch and length."
            )

        response_mask = self._build_response_mask(response_ids)
        prev_logprobs = compute_logprobs_from_logits(
            logits=generated_scores.float(),
            target=response_ids,
        ).unsqueeze(-1)
        prev_logprobs = prev_logprobs * response_mask.unsqueeze(-1).to(
            prev_logprobs.dtype
        )
        return prev_logprobs, response_mask
```

- [ ] **Step 3: Add joint padding helper**

Add this method after `_pad_response_forward_inputs`:

```python
    def _pad_response_forward_inputs_with_logprobs(
        self,
        *,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        prev_logprobs: torch.Tensor,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_len = int(response_ids.shape[1])
        if response_mask.shape != response_ids.shape:
            raise ValueError("UniNaVid response mask must match response ids shape.")
        if prev_logprobs.shape != (*response_ids.shape, 1):
            raise ValueError(
                "UniNaVid generation prev_logprobs must have shape [batch, response_len, 1]."
            )
        if current_len > target_len:
            raise ValueError(
                "UniNaVid train metadata response length exceeds max_new_tokens."
            )
        if current_len == target_len:
            return response_ids, response_mask, prev_logprobs

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        pad_len = target_len - current_len
        id_pad = torch.full(
            (response_ids.shape[0], pad_len),
            int(pad_token_id),
            dtype=response_ids.dtype,
            device=response_ids.device,
        )
        mask_pad = torch.zeros(
            (response_ids.shape[0], pad_len),
            dtype=torch.bool,
            device=response_ids.device,
        )
        logprob_pad = prev_logprobs.new_zeros(
            (prev_logprobs.shape[0], pad_len, prev_logprobs.shape[2])
        )
        return (
            torch.cat([response_ids, id_pad], dim=1),
            torch.cat([response_mask.to(torch.bool), mask_pad], dim=1),
            torch.cat([prev_logprobs, logprob_pad], dim=1),
        )
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py::test_uninavid_generation_scores_compute_prev_logprobs_and_response_mask tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py::test_uninavid_pads_response_ids_masks_and_generation_logprobs_together -q
```

Expected: both tests pass.

## Task 3: Wire Train Generation Scores Into UniNaVid Rollout

**Files:**
- Modify: `rlinf/models/embodiment/uninavid/uninavid_action_model.py`

- [ ] **Step 1: Extend `_generate_batched_navigation_outputs` signature**

Change the method signature to:

```python
    def _generate_batched_navigation_outputs(
        self,
        env_obs: dict[str, Any],
        generation_kwargs: dict[str, Any],
        *,
        return_scores: bool = False,
    ):
```

- [ ] **Step 2: Request generation scores only when asked**

Replace the current `self.model.generate(...)` block in
`_generate_batched_navigation_outputs` with:

```python
            generate_kwargs = dict(generation_kwargs)
            if return_scores:
                generate_kwargs["return_dict_in_generate"] = True
                generate_kwargs["output_scores"] = True
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generate_kwargs,
            )
            if return_scores:
                generated_scores = torch.stack(tuple(outputs.scores), dim=1).float()
                response_ids = outputs.sequences[
                    :, 1 : 1 + generated_scores.shape[1]
                ]
            else:
                generated_scores = None
                response_ids = self._response_ids_from_inputs_embeds_generation(
                    outputs
                )
            output_texts = self.tokenizer.batch_decode(
                response_ids,
                skip_special_tokens=True,
            )
            return (
                output_texts,
                inputs_embeds,
                attention_mask,
                response_ids,
                generated_scores,
            )
```

- [ ] **Step 3: Keep `_generate_batched_navigation_texts` compatible**

Change `_generate_batched_navigation_texts` to unpack the extra return value:

```python
        output_texts, _, _, _, _ = self._generate_batched_navigation_outputs(
            env_obs,
            generation_kwargs,
        )
```

- [ ] **Step 4: Use generation scores in train rollout**

Change the start of `_predict_action_batch_with_batched_feature_cache_train` to:

```python
        (
            output_texts,
            prompt_inputs_embeds,
            prompt_attention_mask,
            response_ids,
            generated_scores,
        ) = self._generate_batched_navigation_outputs(
            env_obs,
            generation_kwargs,
            return_scores=True,
        )
```

Replace the current response padding and post-generation recomputation block:

```python
        response_ids, response_mask = self._pad_response_forward_inputs(
            response_ids,
            target_len=response_len,
        )
        with torch.no_grad():
            prev_logprobs = self._compute_response_logprobs_from_embeds(
                prompt_inputs_embeds=prompt_inputs_embeds,
                prompt_attention_mask=prompt_attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
            )
```

with:

```python
        if generated_scores is None:
            raise ValueError("UniNaVid train generation expected output scores.")
        prev_logprobs, response_mask = self._compute_generation_score_logprobs(
            generated_scores=generated_scores,
            response_ids=response_ids,
        )
        response_ids, response_mask, prev_logprobs = (
            self._pad_response_forward_inputs_with_logprobs(
                response_ids=response_ids,
                response_mask=response_mask,
                prev_logprobs=prev_logprobs,
                target_len=response_len,
            )
        )
```

- [ ] **Step 5: Run train-path test**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py::test_uninavid_train_rollout_uses_generation_scores_without_recompute -q
```

Expected: the test passes.

## Task 4: Regression Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run UniNaVid rollout metadata unit tests**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run UniNaVid loss tests**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_rl_loss.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run UniNaVid toy overfit test**

Run:

```bash
pytest tests/unit_tests/models/embodiment/uninavid/test_toy_overfit.py -q
```

Expected: test passes. If this test is slow in the local environment, report the
runtime and any failure output instead of changing training logic.

- [ ] **Step 4: Check focused diff**

Run:

```bash
git diff -- rlinf/models/embodiment/uninavid/uninavid_action_model.py tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py
```

Expected: diff only changes UniNaVid train rollout logprob computation and the
new focused tests. Metadata keys remain unchanged.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add rlinf/models/embodiment/uninavid/uninavid_action_model.py tests/unit_tests/models/embodiment/uninavid/test_rollout_metadata.py
git commit -m "fix: use generation scores for uninavid train logprobs"
```

Expected: commit succeeds and does not include unrelated files.

