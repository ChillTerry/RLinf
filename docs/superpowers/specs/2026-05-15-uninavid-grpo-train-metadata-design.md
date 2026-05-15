# UniNaVid GRPO Train Metadata Design

## Goal

Replace UniNaVid GRPO train metadata that stores detached prompt embeddings with
tensor-only prompt source data. Actor updates must rebuild the multimodal prompt
through the current `mm_projector`, current text embeddings, current LLM
backbone, and current `lm_head`.

## Scope

This design changes only the UniNaVid GRPO train path.

The eval path keeps the existing UniNaVid online navigation behavior, including
similarity-based long-history merge.

Changes should stay under `rlinf/models/embodiment/uninavid/` whenever feasible.
No RLinf core framework changes are part of this design.

## Current Behavior

The UniNaVid GRPO train rollout path currently builds prompt embeddings during
rollout and stores them in train metadata:

```python
metadata["forward_inputs"] = {
    "prompt_inputs_embeds": prompt_inputs_embeds.detach(),
    "prompt_attention_mask": prompt_attention_mask.detach(),
    "response_ids": response_ids.detach(),
    "response_mask": response_mask.detach(),
}
```

Actor training then reuses `prompt_inputs_embeds` directly. This detaches prompt
construction from the actor update graph, so GRPO loss does not train the
`mm_projector` or prompt text embeddings through the sampled multimodal prompt.

## Train Chain Assumptions

The target launch path is:

```text
bash examples/embodiment/run_embodiment.sh habitat_r2r_grpo_uninavid
-> examples/embodiment/train_embodied_agent.py
-> EmbodiedRunner
-> EnvWorker + MultiStepRolloutWorker
-> HabitatEnv.chunk_step
```

The train env inherits these settings from `examples/embodiment/config/env/habitat_r2r.yaml`:

```yaml
auto_reset: False
ignore_terminations: False
```

For this path, `EnvWorker.bootstrap_step()` resets all train envs at the start
of each rollout epoch. `HabitatEnv.chunk_step()` then returns a uniform new-frame
window for all env slots:

```text
bootstrap observation: 1 frame
after each action chunk: num_action_chunks frames
```

The env observation's `rgb_frame_history` is only the new-frame window for the
current rollout step. Full navigation history is maintained incrementally inside
the UniNaVid rollout policy's per-slot navigation cache.

In the approved design, UniNaVid GRPO train assumes slot history lengths are
uniform within a rollout batch. The implementation should assert this invariant
when building train metadata.

## New Metadata

UniNaVid GRPO train metadata should store tensor-only prompt source data:

```python
forward_inputs = {
    "prompt_input_ids": LongTensor[B, prompt_input_len],
    "history_visual_features": Tensor[B, history_frames, nav_size, vision_dim],
    "current_visual_features": Tensor[B, current_visual_tokens, vision_dim],
    "response_ids": LongTensor[B, response_len],
    "response_mask": BoolTensor[B, response_len],
}
```

The metadata must not store:

```python
"prompt_inputs_embeds"
"prompt_attention_mask"
"history_lengths"
"history_frame_mask"
```

`history_lengths` and `history_frame_mask` are intentionally omitted because the
approved train path has uniform history length. `history_frames` is represented
by `history_visual_features.shape[1]`.

## Train Rollout Prompt Construction

The UniNaVid GRPO train rollout path should build prompts from pre-projector
visual features.

Per slot:

1. Read the current new-frame window from env observation.
2. Run the vision tower to obtain pre-projector frame features.
3. Build history features with `process_grid(nav_size)`.
4. Append those pre-projector history features to the per-slot train navigation
   cache.
5. Build current-frame features with `process_grid(8)` from the current frame.
6. Rebuild the generation prompt by applying the current rollout model's
   `mm_projector` to the pre-projector history and current features.
7. Insert projected history features by direct concatenation, with separator text
   embeddings between frames.
8. Insert projected current-frame features in the current observation position.

The GRPO train path must not use `similarity > similarity_threshold` long-history
merge. Direct concatenation applies to both train rollout generation and actor
update prompt reconstruction, so sampled actions and train logprobs use the same
prompt structure.

## Actor Update Prompt Reconstruction

Actor update should rebuild prompt embeddings from `forward_inputs`:

1. Embed `prompt_input_ids` using the current model's text embedding layer.
2. Project `history_visual_features` using the current `mm_projector`.
3. Project `current_visual_features` using the current `mm_projector`.
4. Direct-concatenate all history frame visual tokens.
5. Rebuild per-sample navigation prompt embeddings with the same prompt layout
   used during train rollout generation.
6. Left-pad rebuilt prompts to the train prompt target length or batch maximum.
7. Embed `response_ids` with the current text embedding layer.
8. Concatenate prompt and response embeddings.
9. Run the current LLM backbone and `lm_head`.
10. Compute response token logprobs and GRPO actor loss.

The expected gradient path is:

```text
GRPO loss
  -> lm_head
  -> LLM backbone
  -> response token embeddings
  -> prompt text embeddings
  -> mm_projector
```

The vision tower is not trained by this metadata change because train metadata
stores pre-projector visual features, not RGB images.

## Invariants

When building GRPO train metadata, the implementation should reject non-uniform
slot history lengths:

```python
if len({features.shape[0] for features in per_slot_history_features}) != 1:
    raise ValueError("UniNaVid GRPO train expects uniform history length.")
```

This prevents future train configurations with independent auto-reset or
asynchronous episode age from silently treating padding as real visual history.

## Tests

Focused tests should verify:

1. UniNaVid GRPO train metadata no longer contains `prompt_inputs_embeds`.
2. Actor forward for UniNaVid GRPO train accepts only tensor-only prompt source
   data plus response tensors.
3. `mm_projector` parameters receive nonzero gradients after GRPO loss backward.
4. Rebuilt train prompt lengths and attention masks match the rollout train
   prompt structure.
5. Eval and non-train rollout paths still use the existing behavior and are not
   affected.
6. Non-uniform train history lengths raise a clear `ValueError`.

