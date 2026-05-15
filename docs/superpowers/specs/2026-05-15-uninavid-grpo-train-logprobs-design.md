# UniNaVid GRPO Train Logprobs Design

## Goal

Change only UniNaVid GRPO train rollout `prev_logprobs` computation to use the
NavID-style generation score path.

The train rollout should compute `prev_logprobs` from
`generate(..., return_dict_in_generate=True, output_scores=True).scores` instead
of running a second model forward pass after generation.

## Scope

This design changes only UniNaVid GRPO train rollout logprob computation.

It does not change `forward_inputs` metadata. Train metadata continues to store
detached `prompt_inputs_embeds` and `prompt_attention_mask`.

It does not change actor update prompt reconstruction.

It does not change eval or non-train rollout behavior.

Changes should stay under `rlinf/models/embodiment/uninavid/` whenever feasible.
No RLinf core framework changes are part of this design.

## Current Behavior

The UniNaVid GRPO train rollout path generates response tokens, pads the prompt
and response tensors, and then recomputes rollout `prev_logprobs` by running a
second model forward over the generated response:

```python
with torch.no_grad():
    prev_logprobs = self._compute_response_logprobs_from_embeds(
        prompt_inputs_embeds=prompt_inputs_embeds,
        prompt_attention_mask=prompt_attention_mask,
        response_ids=response_ids,
        response_mask=response_mask,
    )
```

The generated response ids and masks are stored in metadata together with the
detached prompt embeddings:

```python
metadata = {
    "prev_logprobs": prev_logprobs.detach(),
    "prev_values": None,
    "forward_inputs": {
        "prompt_inputs_embeds": prompt_inputs_embeds.detach(),
        "prompt_attention_mask": prompt_attention_mask.detach(),
        "response_ids": response_ids.detach(),
        "response_mask": response_mask.detach(),
    },
}
```

This design keeps the metadata structure unchanged and removes only the
post-generation logprob recomputation.

## New Logprob Path

UniNaVid GRPO train generation should request generation scores:

```python
outputs = self.model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True,
    **generation_kwargs,
)
```

The train path should decode response ids from `outputs.sequences`. Because
UniNaVid generation uses `inputs_embeds` without `input_ids`, Hugging Face
generation creates a synthetic seed token. Response ids start after that seed:

```python
generated_scores = torch.stack(tuple(outputs.scores), dim=1).float()
response_ids = outputs.sequences[:, 1 : 1 + generated_scores.shape[1]]
```

Rollout `prev_logprobs` should be computed directly from `generated_scores`:

```python
response_mask = self._build_response_mask(response_ids)
prev_logprobs = compute_logprobs_from_logits(
    logits=generated_scores,
    target=response_ids,
).unsqueeze(-1)
prev_logprobs = prev_logprobs * response_mask.unsqueeze(-1).to(prev_logprobs.dtype)
```

After that, pad `response_ids`, `response_mask`, and `prev_logprobs` together to
the configured response target length.

## Metadata

Metadata remains unchanged:

```python
metadata = {
    "prev_logprobs": prev_logprobs.detach(),
    "prev_values": None,
    "forward_inputs": {
        "prompt_inputs_embeds": prompt_inputs_embeds.detach(),
        "prompt_attention_mask": prompt_attention_mask.detach(),
        "response_ids": response_ids.detach(),
        "response_mask": response_mask.detach(),
    },
}
```

This means actor update continues to use detached prompt embeddings. This design
does not attempt to train `mm_projector` through prompt reconstruction.

## Logprob Semantics

For the target local dependency `transformers==4.31.0`, generation `outputs.scores`
are processed generation scores, not guaranteed raw model logits.

In sample mode:

```text
raw logits
-> logits_processor
-> logits_warper
-> outputs.scores
-> softmax sampling distribution
```

Therefore the approved training semantics are:

```text
old_logprobs: processed generation-score logprobs from rollout
new_logprobs: raw actor-forward logprobs from current model
```

This semantic difference is accepted to align UniNaVid rollout `prev_logprobs`
with the NavID-style generation-score path and avoid the extra rollout-time
forward pass.

## Tests

Focused tests should verify:

1. UniNaVid GRPO train generation requests `return_dict_in_generate=True` and
   `output_scores=True`.
2. UniNaVid GRPO train rollout computes `prev_logprobs` from `outputs.scores`.
3. UniNaVid GRPO train rollout no longer calls
   `_compute_response_logprobs_from_embeds`.
4. `response_ids`, `response_mask`, and `prev_logprobs` have matching padded
   response lengths.
5. `forward_inputs` metadata still contains `prompt_inputs_embeds` and
   `prompt_attention_mask`.
6. Eval and non-train rollout paths keep their existing behavior.

