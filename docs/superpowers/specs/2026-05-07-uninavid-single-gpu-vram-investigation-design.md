# UniNaVid Single-GPU VRAM Investigation Design

## Context

Running `bash examples/embodiment/eval_embodiment.sh habitat_r2r_eval_uninavid`
shows large single-GPU memory swings over time. The current investigation focuses
on finding the root cause before changing runtime behavior.

The suspected memory domains are:

- Habitat simulator, OpenGL context, scene assets, and episode reset behavior.
- UniNaVid batched visual encoding and LLaMA generation.
- UniNaVid per-slot navigation history cache.
- PyTorch CUDA caching allocator behavior, especially reserved memory retained
  after generation peaks.

This investigation assumes an isolated single GPU is available and Ray group names
or namespaces do not conflict with other jobs.

## Goals

- Reproduce the single-GPU memory swing with the smallest useful eval setup.
- Distinguish PyTorch model memory from non-PyTorch GPU memory.
- Determine whether the dominant variable is env batch size, decode length,
  rollout mode, episode step count, or Habitat reset/scene behavior.
- Produce enough evidence to justify either no code change, a config adjustment,
  or a narrow UniNaVid/Habitat-only instrumentation patch.

## Non-Goals

- Do not modify generic RLinf scheduler, worker, placement, or rollout flow.
- Do not change UniNaVid action semantics, prompt text, cache semantics, or
  Habitat metrics behavior.
- Do not add fallback behavior, heuristic memory cleanup, or post-processing
  bandages.
- Do not install dependencies or alter the runtime environment.

## Baseline Configuration

Run a single-GPU RLinf Habitat UniNaVid eval with:

```text
cluster.component_placement.actor,env,rollout = GPUx-GPUx
env.eval.total_num_envs = 1
env.eval.max_episode_steps = 512
env.eval.max_steps_per_rollout_epoch = 512
env.eval.video_cfg.save_video = false
actor.model.rollout_mode = batched_feature_cache
algorithm.length_params.max_new_token = 1024
```

The exact GPU id is chosen at execution time. The selected GPU should be idle
before the run starts.

## Measurement

Collect external GPU samples at one-second cadence for the selected GPU:

```text
timestamp
gpu_index
memory.used
memory.free
memory.total
utilization.gpu
```

Also collect process-level GPU memory during the same window:

```text
gpu_uuid
pid
process_name
used_memory
```

Record the RLinf log path, full resolved config, and worker role-to-PID mapping.
The sampling window must include model load, Habitat env initialization, eval
reset, rollout generation, and shutdown.

## Experiment Matrix

### Phase 1: Minimal Baseline

Run the baseline with `env.eval.total_num_envs=1`.

Interpretation:

- Large swings at `total_num_envs=1` point toward Habitat reset/scene behavior,
  single-sample generation peaks, or allocator behavior.
- A stable run at `total_num_envs=1` means the main issue likely needs batch
  amplification to reproduce.

### Phase 2: Batch Scaling

Run the same config with only this variable changed:

```text
env.eval.total_num_envs = 1, 2, 4, 8, 16
```

Interpretation:

- Peak memory and swing amplitude increasing with env count points toward
  batched UniNaVid visual encoding, batched `inputs_embeds`, or generation KV
  cache.
- Weak dependence on env count points toward Habitat fixed costs, allocator
  behavior, or scene/reset effects.
- A discontinuity at one env count suggests padding length, generated length,
  or scene allocation changed at that scale.

### Phase 3: Rollout Mode Contrast

Fix a safe env count, preferably `4` or `8`, and compare:

```text
actor.model.rollout_mode = batched_feature_cache
actor.model.rollout_mode = sequential_cache
```

Interpretation:

- A much smoother `sequential_cache` run identifies batched generation as the
  dominant source.
- Similar swings in both modes point toward Habitat/OpenGL/scene reset or
  allocator behavior.
- Lower total memory in `sequential_cache` with continued gradual growth points
  toward navigation history cache or allocator retention.

### Phase 4: Decode Length Contrast

Fix env count and rollout mode, then compare:

```text
algorithm.length_params.max_new_token = 32, 128, 1024
```

Interpretation:

- Peak memory increasing with `max_new_token` identifies LLaMA generation KV
  cache and generated length as a major driver.
- Similar peaks across lengths point toward visual encoding, Habitat, or
  allocator behavior.

### Phase 5: Episode Step and History Contrast

Fix env count, rollout mode, and decode length, then compare:

```text
env.eval.max_steps_per_rollout_epoch = 512
env.eval.max_steps_per_rollout_epoch = 1024
```

Interpretation:

- Memory growing within an episode and dropping after reset points toward
  navigation history cache.
- Memory growing across episodes without a corresponding drop points toward
  cache reset behavior or allocator retention.
- Memory jumping mostly around reset or scene changes points toward Habitat.

## Evidence Classification

Use this table to classify the observed behavior:

| Observation | Likely Root Cause |
| --- | --- |
| Peak grows with `total_num_envs` | Batched visual encoding, `inputs_embeds`, or generation KV cache |
| `sequential_cache` is much smoother | `batched_feature_cache` generation path |
| Peak grows with `max_new_token` | LLaMA generation KV cache or long generation |
| `total_num_envs=1` still swings heavily | Habitat reset/scene behavior or allocator |
| `nvidia-smi` swings but PyTorch allocated memory does not | Habitat/OpenGL or non-PyTorch CUDA memory |
| PyTorch allocated drops but reserved stays high | CUDA caching allocator retention |
| Memory grows gradually inside each episode | UniNaVid navigation history cache |
| Memory jumps at reset/scene boundaries | Habitat simulator, OpenGL context, or scene assets |

## Instrumentation Gate

Only add instrumentation if the external measurements cannot distinguish the
dominant source. Any instrumentation must be opt-in and scoped to:

- `rlinf/models/embodiment/uninavid/`
- `rlinf/envs/habitat/`

Acceptable model-side trace points:

- before and after navigation image preprocessing;
- before and after vision tower encoding;
- before and after slot cache update;
- before and after navigation `inputs_embeds` padding;
- before and after `model.generate`.

Acceptable Habitat-side trace points:

- before and after `reset`;
- before and after `chunk_step`;
- before and after `_wrap_obs`;
- around auto-reset handling.

Each trace should record:

```text
component
rank
pid
stage
torch.cuda.memory_allocated
torch.cuda.memory_reserved
torch.cuda.max_memory_allocated
```

If possible, pair those readings with external `nvidia-smi` samples so PyTorch
and non-PyTorch GPU memory can be separated.

## Success Criteria

The investigation is complete when the evidence supports one of these outcomes:

- The swing is explained by expected batched generation/KV-cache peaks.
- The swing is explained by Habitat/OpenGL/scene reset memory.
- The swing is explained by PyTorch caching allocator retention rather than an
  active leak.
- The swing is traced to UniNaVid navigation history cache growth.
- The evidence is insufficient, and a narrow opt-in instrumentation patch is
  justified with specific trace points.

## Execution Order

Run in this order unless an earlier phase already identifies the dominant cause:

1. Minimal baseline with `total_num_envs=1`.
2. Batch scaling with `total_num_envs=1,4,8`.
3. Rollout mode contrast at a safe env count.
4. Decode length contrast.
5. Episode step/history contrast.
6. Add opt-in instrumentation only if the above remains ambiguous.
