# UniNaVid Habitat GRPO Design

Date: 2026-05-07

## Goal

Enable RL post-training for UniNaVid on Habitat R2R through RLinf embodied GRPO while preserving the current UniNaVid Habitat eval adaptation as the reference behavior.

The first implementation should be the smallest training-only increment required to connect UniNaVid to RLinf's rollout, advantage, and actor-update contracts. Existing UniNaVid eval logic, Habitat chunk execution semantics, parser behavior, and navigation cache behavior should remain unchanged unless a required RL training integration point cannot be implemented otherwise.

## Non-Goals

- Do not redesign UniNaVid navigation rollout or parser behavior.
- Do not replace the current Habitat stop/no-op chunk execution semantics.
- Do not introduce progress, nDTW, format, or action-level shaping rewards in the first version.
- Do not add action-token span alignment.
- Do not rebuild the UniNaVid vision tower, multimodal projector, or online navigation cache path during actor training.
- Do not make generic RLinf behavior changes unless isolated behind `model_type == "uninavid"`.

## Reference Behavior To Preserve

The existing UniNaVid prompt asks for the next four actions. The training config should therefore use:

```yaml
actor:
  model:
    num_action_chunks: 4
    action_dim: 1
```

The existing parser behavior is preserved:

- If fewer than four primitive actions are parsed, pad with `no-op`.
- If `stop` is parsed, stop parsing and pad the remaining chunk slots with `no-op`.
- If more than four primitive actions are parsed, keep only the first four.
- Unknown text is ignored by the keyword parser. If no valid action is parsed, all four slots are padded as `no-op`.

This means the environment always receives a fixed action tensor shaped as `[B, 4, 1]`.

## Training Semantics

UniNaVid remains a text-generating navigation policy. The actor loss is computed over generated response tokens, not over parsed Habitat action ids.

Configured RLinf algorithm semantics:

```yaml
algorithm:
  adv_type: grpo
  loss_type: actor
  reward_type: chunk_level
  logprob_type: token_level
  entropy_type: token_level
  loss_agg_func: token-mean
```

One training sample corresponds to:

```text
current Habitat observation and UniNaVid nav cache
  -> UniNaVid generates one response string
  -> response is parsed into four primitive Habitat actions
  -> Habitat executes the four-action chunk
  -> sparse success reward is accumulated at chunk level
  -> chunk GRPO advantage is broadcast to valid response tokens
```

## Reward Design

The first version uses the existing sparse Habitat success reward only:

```text
reward = reward_coef * success
```

With `use_rel_reward: True`, a successful stop produces one positive reward increment and avoids repeated success reward on following no-op chunk slots.

No progress shaping, nDTW reward, format reward, action-level reward, or custom reward worker is introduced in the first version.

Expected behavior:

- If a group has no successful samples, group rewards may all be zero and GRPO advantage may be zero.
- This sparse signal is accepted for the first implementation because the goal is to validate the training data path without changing task semantics.

## Rollout Output Contract

`UniNaVidForActionPrediction.predict_action_batch(...)` should keep eval behavior unchanged. In train mode, or when logprob collection is requested, it additionally returns training metadata.

Rollout actions:

```python
actions  # [B, 4, 1], parsed Habitat primitive action ids
```

Rollout metadata:

```python
{
    "prev_logprobs": old_token_logprobs,  # [B, response_len, 1]
    "prev_values": None,
    "forward_inputs": {
        "inputs_embeds": detached_navigation_prompt_embeds,
        "attention_mask": prompt_attention_mask,
        "response_ids": response_token_ids,
        "response_mask": response_token_mask,
    },
}
```

`prev_logprobs` should be computed by teacher-forcing the generated response with the rollout model after generation, matching the ActiveVLN/verl pattern of recomputing old log probabilities rather than trusting generation scores.

## Actor Update Contract

`UniNaVidForActionPrediction.default_forward(...)` should support actor-side logprob recomputation from `forward_inputs`.

It should return:

```python
{
    "logprobs": current_token_logprobs,  # [B, response_len, 1]
    "values": None,
    "entropy": token_entropy_or_none,
}
```

The saved navigation prompt embeddings are treated as detached inputs. The first version updates the LLM/LoRA response-token likelihood path and does not backpropagate into vision tower, multimodal projector, or navigation-cache construction.

LoRA-first is the default training target. Full LLM FSDP fine-tuning may remain configurable if supported by the existing framework, but it is not the primary validation path.

## Masking And Boundary Cases

The policy loss uses response-token masks, not action-chunk masks.

Boundary cases:

- Fewer than four actions: parser pads no-op for environment execution. Only generated response tokens receive logprob loss.
- `stop` before four actions: parser pads no-op for the remaining chunk slots. Only generated response tokens receive logprob loss.
- More than four actions: environment executes the first four parsed actions. The whole generated response remains in the token logprob loss by default.
- Empty response: if EOS was generated, EOS is a valid response token. If no token exists, the response mask is all zero and the sample contributes no actor loss.
- Padded no-op actions are deterministic parser outputs. They affect the chunk reward but do not create synthetic no-op response labels.

Because RLinf embodied loss masks are action-chunk oriented, UniNaVid needs a model-specific bridge that uses `forward_inputs["response_mask"]` as the actor-loss mask. Any shared-path change must be isolated behind `model_type == "uninavid"` and documented as required because UniNaVid's training unit is a variable-length text response rather than a fixed action chunk.

## ActiveVLN Alignment

This design follows ActiveVLN/verl in the following ways:

- Generate a text response.
- Parse the text response into environment actions.
- Recompute old log probabilities by teacher-forcing the generated response.
- Train on response-token log probabilities with a response mask.
- Use group sampling for GRPO.

It intentionally differs from ActiveVLN in the following ways to preserve UniNaVid eval semantics:

- It does not enforce strict action-string formatting.
- It does not treat fewer than max actions as an error.
- It does not add format reward.
- It does not use GT prefix or expert action warmup.
- It does not rebuild multimodal inputs during actor training in the first version.

## Required Tests

Focused tests should cover:

- `num_action_chunks: 4` config for UniNaVid Habitat GRPO.
- Train-mode rollout returns actions, response ids, response mask, old token logprobs, and no values.
- Existing eval-mode UniNaVid rollout output remains unchanged.
- Parser padding behavior is unchanged for short responses and `stop`.
- Actor default forward recomputes response-token logprobs with the expected shape.
- UniNaVid-specific response mask is applied to actor loss without affecting other model types.
- Sparse success reward path remains the Habitat env default for this training config.

## Open Implementation Notes

- The rollout helper should avoid using generation scores for old logprobs; recompute with teacher forcing.
- Response token padding length should be local to the rollout batch and represented by `response_mask`.
- If generated responses include long explanations, the first implementation trains all generated response tokens. Action-span masking is explicitly out of scope.
- Metrics should include parsed action count before padding, padded no-op count, stop position, response length, and chunk reward to monitor short-output behavior.
