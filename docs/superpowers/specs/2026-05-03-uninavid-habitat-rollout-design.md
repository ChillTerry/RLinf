# UniNaVid Habitat Rollout Design

Date: 2026-05-03

## Goal

Add first-class RLinf rollout and evaluation support for UniNaVid in the Habitat
VLN environment. The first implementation stage targets evaluation and rollout,
not PPO post-training. The default fast path should support batched Habitat
environments while preserving UniNaVid navigation memory semantics and meeting
the success/SPL tolerance defined in the success criteria.

## Assumptions

- UniNaVid remains registered as an embodied model through the existing model
  registry.
- Habitat observations provide RGB images through `wrist_images`, instructions
  through `task_descriptions`, and episode ids through `states`.
- The first version uses Habitat discrete navigation actions: stop, forward,
  left, right, and no-op.
- The authoritative original baseline for parity is
  `NaVid-VLN-CE/run.py` with `agent_uninavid.py`, not
  `Uni-NaVid/offline_eval_uninavid.py`.
- The original UniNaVid prompt template is preserved for evaluation parity.
- The original early-stop heuristic is intentionally excluded from RLinf parity
  acceptance for this task and must be documented in the final comparison
  summary.
- The first version does not train the actor, so PPO logprob replay, value
  prediction, and `default_forward` for RL losses are out of scope.

## Success Criteria

- `sequential_cache` mode runs UniNaVid inside RLinf Habitat with one environment
  and reproduces the original UniNaVid eval inputs for each evaluated step:
  identical navigation prompt text, identical current RGB frame after the same
  image preprocessing path, and identical historical frame token sequence before
  generation.
- `sequential_cache` mode runs with multiple Habitat env slots without navigation
  history leaking between slots or episodes.
- `batched_feature_cache` mode runs as the default multi-env evaluation path.
- `sequential_cache` and `batched_feature_cache` are both evaluated against the
  authoritative original UniNaVid eval script on the same checkpoint, split,
  seed policy, and sampling settings. Each comparison must use at least 100
  episodes from the same filtered episode set; single-episode and 20-episode
  runs are allowed only as smoke tests and do not satisfy acceptance.
- `sequential_cache` and `batched_feature_cache` success and SPL must each be no
  lower than the original UniNaVid eval script by more than 5 absolute
  percentage points on the evaluated episode set. This lower-bound acceptance
  applies independently to success and SPL for each rollout mode. Exceeding the
  original result is acceptable and should be recorded as an improvement.
- Before every manual evaluation run, record all local GPU free memory and GPU
  utilization. Select devices by highest idle GPU capacity first
  (`100 - utilization`), then by highest free memory. If the selected devices do
  not have enough memory, add more devices in the same priority order until the
  run fits or all GPUs are selected. If all GPUs still OOM, reduce config-driven
  memory pressure and record the final GPU ids and config overrides with the
  results.
- Record a parity baseline before any non-parity optimization experiment. The
  final summary document must distinguish parity runs from improvement
  experiments and note that early-stop was intentionally excluded.
- Unit tests cover cache reset, slot isolation, action parsing, and action tensor
  shape.

## Non-Goals

- PPO, GRPO, or other actor training.
- Value head integration.
- Training-time logprob recomputation from rollout caches.
- Reworking UniNaVid SFT data loading or SFT worker behavior.
- A broad refactor of UniNaVid internals beyond the minimum needed to expose
  navigation cache operations.

## Existing Constraint

The original UniNaVid evaluation loop assumes a single agent. It stores online
navigation history on the model object:

```python
self.model.get_model().initialize_online_inference_nav_feat_cache()
self.rgb_list.append(rgb)
self.model.get_model().new_frames = len(self.rgb_list)
output_ids = self.model.generate(input_ids, images=[video], use_cache=True)
```

The model-side state is global:

```python
self.feat_cache = None
self.long_feat_cache = None
self.weight = 1
self.new_frames = 0
```

This is correct for a single sequential environment, but unsafe for RLinf
batched Habitat rollout. In a batched rollout, one model instance serves several
environment slots. Each slot can have a different instruction, scene, episode id,
and history length. A single global `feat_cache` would mix the histories of
different slots and would also carry old history into a newly reset episode.

The design therefore treats navigation memory as per-environment-slot state.

## Public Model Interface

Extend `UniNaVidForActionPrediction` with rollout/eval support:

```python
def predict_action_batch(
    self,
    env_obs,
    calculate_logprobs=False,
    calculate_values=False,
    mode="eval",
    **kwargs,
):
    return actions, rollout_metadata
```

The first version returns a rollout-compatible result without training terms:

```python
return actions, {
    "prev_logprobs": None,
    "prev_values": None,
    "forward_inputs": {},
}
```

The output action tensor is:

```python
actions: torch.Tensor  # [batch_size, num_action_chunks, 1]
```

For UniNaVid Habitat parity, `num_action_chunks` should default to 2. The
original prompt asks for the next four actions, but the authoritative original
agent executes only the first two parsed actions from each generation result.

## Cache State

Add a small cache record for navigation rollout:

```python
@dataclass
class UniNaVidNavCache:
    episode_id: int | None = None
    feat_cache: torch.Tensor | None = None
    long_feat_cache: torch.Tensor | None = None
    weight: int = 1
    new_frames: int = 0
```

The wrapper owns:

```python
self._nav_caches: dict[int, UniNaVidNavCache]
```

The dictionary key is the batch slot id. Before every prediction, compare the
stored `episode_id` with `int(env_obs["states"][slot_id])`. If the id changes,
reset only that slot's cache.

## Action Parsing

Map generated UniNaVid text to Habitat action ids:

```text
stop    -> 0
forward -> 1
left    -> 2
right   -> 3
unknown -> 4
padding -> 4
```

Parsing should be forgiving about whitespace and punctuation. It should keep the
first `num_action_chunks` valid actions. If fewer than `num_action_chunks`
actions are generated, pad with no-op. If `stop` appears, keep the stop action
and pad the remaining chunk with no-op. This matches Habitat's chunk stepping
behavior, which truncates after stop and pads the rest of the chunk. For parity,
`num_action_chunks=2` because the original agent only consumes two actions per
generation even though the prompt asks for four.

## Rollout Modes

Expose a config switch:

```yaml
actor:
  model:
    rollout_mode: batched_feature_cache
```

Supported values:

- `sequential_cache`: parity and debugging path.
- `batched_feature_cache`: default fast path.

### sequential_cache

This mode preserves the original UniNaVid online eval path by reusing upstream
generation with an explicit per-slot cache swap. It processes each batch slot
one at a time:

```python
for slot_id in range(batch_size):
    episode_id = int(env_obs["states"][slot_id])
    reset_cache_if_episode_changed(slot_id, episode_id)

    load_slot_cache_into_model(slot_id)
    output_text = generate_one_env(slot_id)
    save_model_cache_back_to_slot(slot_id)

    actions[slot_id] = parse_output(output_text)
```

`load_slot_cache_into_model` writes the slot cache into the model's global
UniNaVid fields before generation. `save_model_cache_back_to_slot` copies the
updated fields back after generation. This prevents one slot from seeing another
slot's history while still using the upstream online memory path.

This mode is intentionally slower because batch size `B` performs `B` generation
calls. It is the baseline for verifying prompt construction, image processing,
action parsing, Habitat wiring, and reset boundaries.

### batched_feature_cache

This is the target default path for RLinf multi-env evaluation. It avoids global
model-side cache semantics and makes cache ownership explicit:

```python
episode_ids = env_obs["states"]
reset_changed_episode_caches(episode_ids)

pixel_batch = preprocess_rgb_batch(env_obs["wrist_images"])
current_features = vision_tower(pixel_batch)

for slot_id in range(batch_size):
    update_slot_feature_cache(slot_id, current_features[slot_id])
    visual_tokens[slot_id] = build_navigation_visual_tokens(slot_id)

input_ids = build_batched_navigation_input_ids(prompts)
outputs = generate_with_batched_visual_tokens(input_ids, visual_tokens)
```

The implementation should extract only the navigation-cache-specific UniNaVid
logic into helpers rather than refactoring the full model. The useful helpers
are:

```python
process_grid(visual_features, grid_size)
update_online_nav_cache(cache, current_visual_tokens)
build_navigation_visual_tokens(cache)
build_multimodal_embeddings(input_ids, attention_mask, visual_tokens)
```

The fast path encodes only the current RGB frame for each slot, updates that
slot's feature cache, compresses the per-slot history, and then batches the LLM
generation. This keeps the memory semantics slot-local while avoiding repeated
vision encoding of the full raw RGB history.

## Prompt and Sampling

The default prompt should match the original UniNaVid eval template:

```text
Imagine you are a robot programmed for navigation tasks. You have been given a
video of historical observations and an image of the current observation
<image>. Your assigned task is: '{}'. Analyze this series of images to determine
your next four actions. The predicted action should be one of the following:
forward, left, right, or stop.
```

Default eval sampling should match the original script unless overridden:

```yaml
do_sample: true
temperature_eval: 0.2
top_k: 50
top_p: 0.6
max_new_token: 1024
```

The authoritative original script explicitly sets `do_sample`, `temperature`,
and `max_new_tokens`. The effective `top_k` and `top_p` values come from the
checkpoint `generation_config.json` under `transformers==4.31.0`, so parity
runs must preserve `top_k: 50` and `top_p: 0.6`.

The RLinf config should still allow these parameters to be overridden for
clearly labeled optimization experiments, but only after a parity baseline is
recorded.

## GPU Resource Selection and OOM Fallback

Before running the original UniNaVid eval script, `sequential_cache`, or
`batched_feature_cache`, inspect current GPU state with `nvidia-smi` or an
equivalent NVML query. The scheduler or run script should capture at least:

```text
gpu_id
memory.free
memory.total
utilization.gpu
```

Choose GPUs with this priority:

1. Higher idle GPU capacity first, computed as `100 - utilization.gpu`.
2. Higher free memory second.
3. Stable ascending `gpu_id` as the final tie breaker.

The first attempt should use the smallest device set expected to fit the
selected mode. On OOM, expand the selected device set using the same priority
order until either the run succeeds or all local GPUs are selected.

If all GPUs are selected and the run still OOMs, reduce memory pressure through
configuration rather than changing model semantics first. Preferred reductions:

- lower `env.eval.total_num_envs` or `env.train.total_num_envs`;
- lower per-rollout parallelism before changing sampling behavior;
- enable existing offload options when available;
- lower precision only if the checkpoint and model path support it.

Do not change prompt text, image preprocessing, historical frame tokenization,
or eval sampling parameters for acceptance runs unless the result is recorded as
a separate non-comparable ablation.

## Implementation Plan

1. Add cache and action parsing helpers to the UniNaVid wrapper.
2. Implement `sequential_cache` prediction using cache swap around the existing
   UniNaVid generation path.
3. Add a Habitat UniNaVid eval config using `rollout.backend: huggingface`,
   `actor.model.model_type: uninavid`, `actor.model.num_action_chunks: 2`, and
   the authoritative effective sampling config.
4. Validate single-env parity against the original UniNaVid eval script by
   comparing prompt text, current RGB tensors, historical frame token sequence,
   success, and SPL on at least 100 shared episodes from the same filtered
   dataset. Acceptance is that RLinf is not more than 5 percentage points below
   the original on success or SPL. Document that early-stop was intentionally
   excluded from parity scope.
5. Add run-time GPU selection guidance or a helper script that records GPU free
   memory/utilization, selects devices by idle capacity then free memory, and
   documents any OOM-driven config reductions.
6. Implement `batched_feature_cache` by extracting minimal navigation cache
   helpers from UniNaVid internals.
7. Validate multi-env cache isolation and compare `batched_feature_cache`
   success/SPL against the original UniNaVid eval script on the same 100 shared
   episodes. Record improvements above the original separately from parity.
8. After parity is recorded, run explicitly labeled optimization experiments and
   summarize them separately from parity evidence.

## Tests

Add focused unit tests:

```python
def test_uninavid_nav_cache_resets_when_episode_id_changes():
    assert changed_episode_id_clears_only_that_slot_cache


def test_uninavid_nav_cache_is_slot_isolated():
    assert updating_one_slot_does_not_change_another_slot_cache


def test_uninavid_action_parser_maps_text_to_habitat_ids():
    assert generated_navigation_words_map_to_discrete_habitat_ids


def test_uninavid_predict_action_batch_returns_habitat_chunk_shape():
    assert returned_actions_shape_is_batch_by_two_chunks_by_one
```

Add a small smoke test if Habitat assets are available in CI or document it as a
manual GPU/Habitat check if assets are not available.

## Risks and Mitigations

- Batched generation may not exactly match single-env generation because of
  numerical and sampling differences. The accepted metric is success/SPL
  not more than 5 percentage points below the stated original baseline, not
  generated action token identity.
- UniNaVid internal cache logic is currently coupled to model global fields.
  `sequential_cache` provides a parity baseline before introducing the fast path.
- There are multiple upstream UniNaVid evaluation entrypoints in the workspace.
  The comparison must stay pinned to `NaVid-VLN-CE/run.py` with
  `agent_uninavid.py`, or the baseline will drift.
- Reusing `env_obs["states"]` as episode id assumes Habitat keeps it stable within
  an episode and changes it after reset. This should be verified with a small
  cache reset test and a multi-env smoke run.
- Long episodes can grow cache memory. The fast path should reuse UniNaVid's
  existing short/long memory compression semantics instead of storing raw RGB
  history indefinitely.
- The original early-stop heuristic is intentionally omitted for this task. That
  may shift absolute metrics, so every parity and optimization summary must
  state that the comparison excludes early-stop.
- GPU availability can vary across shared servers. Runs should record selected
  GPU ids, free memory, utilization, and any OOM fallback config changes so
  success/SPL comparisons are reproducible.

## Future PPO Work

After rollout/eval is stable, PPO post-training requires:

```python
def default_forward(forward_inputs, compute_logprobs, compute_values, **kwargs):
    return {"logprobs": logprobs, "values": values}
```

The rollout path must return:

```python
{
    "prev_logprobs": prev_logprobs,
    "prev_values": prev_values,
    "forward_inputs": forward_inputs,
}
```

For `actor_critic`, UniNaVid also needs a value head. The value feature should be
chosen consistently with OpenVLA/OpenVLA-OFT, likely from the hidden state near
the generated navigation action tokens. This is intentionally deferred until the
Habitat rollout path is stable.
