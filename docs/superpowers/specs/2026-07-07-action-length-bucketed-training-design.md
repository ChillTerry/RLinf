# Action-Length-Bucketed Training for Habitat VLN (B1′ Gradient-Accumulation)

**Date:** 2026-07-07
**Config target:** `examples/embodiment/config/habitat_rxr_grpo_uninavid.yaml` (and r2r sibling)
**Status:** Design (approved pre-implementation)

## 1. Problem

All training episodes are rolled out under a single uniform `max_episode_steps` / `max_steps_per_rollout_epoch` (rxr: 200). Short episodes (gt action length ~30) therefore waste collection compute and produce long masked tails. We want to bucket the dataset by GT action length, give each bucket its own (shorter) `max_steps_per_rollout_epoch`, and still let one actor update see a distribution of lengths for stability.

## 2. Key Architectural Constraints (from code investigation)

1. **One training step = one rollout collection + one actor update** (`rlinf/runners/embodied_runner.py`). Inside the rollout, `algorithm.rollout_epoch` (rxr: 4) inner-epochs are accumulated.
2. **`EmbodiedRolloutResult` stacks inner-epochs into a single fixed-length tensor** with `trajectory_length = n_chunk_steps * rollout_epoch + 1`; the actor reshapes via `view(rollout_epoch, -1, ...)` (`rlinf/workers/actor/fsdp_actor_worker.py:1187-1195`, `process_nested_dict_for_adv`). Hence all inner-epochs in a step **must share the same `n_chunk_steps`** under the current mechanism.
3. **`max_steps_per_rollout_epoch` → `n_train_chunk_steps = max_steps_per_rollout_epoch // num_action_chunks`** (`rlinf/workers/env/env_worker.py:107-110`) determines the tensor time dimension. `max_episode_steps` is only the truncation horizon and affects `dones`/`loss_mask`, not tensor shape.
4. **Episode allocation happens once at env init** (`build_habitat_global_plan` → `global_plan["episode_sequences"]`); each env subprocess's dataset is sliced to a fixed `episode_ids` list for the whole run (`rlinf/envs/habitat/habitat_env.py:796-810`). `update_reset_state_ids` (`habitat_env.py:450-451`) is a `pass` stub.
5. **Runtime episode-subset switching is feasible without scene reload.** Habitat exposes `env.habitat_env.episodes = new_subset` (rebuilds the `EpisodeIterator` only, no sim rebuild; scene meshes cached) and `episode_iterator.set_next_episode_by_id/index`. `update_reset_state_ids` is the reserved hook (libero env implements the same pattern).

## 3. Chosen Approach: B1′ (Gradient-Accumulation Across Buckets)

Per-inner-epoch variable `max_steps_per_rollout_epoch` is incompatible with the single-stacked-tensor mechanism (Constraint 2). Rather than pad (wastes memory/forward) or make each inner-epoch an independent optimizer step (loses per-step distribution), we **decouple inner-epochs from the single tensor**: each inner-epoch (one bucket) is collected as an independent fixed-length batch; the actor feeds all of them through its existing micro-batch gradient-accumulation loop and performs **one `optimizer.step()` per training step**. This satisfies both goals:

- **Save env-sim compute:** each bucket collects only its own `n_chunk_steps`.
- **One update sees the full length distribution:** gradients from all buckets are accumulated before the step.
- **No ragged tensors, no padding:** each bucket batch is a clean fixed-length tensor.
- **Reuses existing grad-accum infra** (`micro_batch_size` → `global_batch_size` already accumulates).

Rejected alternatives:
- **B1 (independent optimizer step per bucket):** loses per-step distribution; needs the same control-flow refactor plus per-bucket stability work.
- **B3′ (pad-to-max within step):** preserves single-tensor stacking but pads short buckets to the longest bucket's `n_chunk_steps`, inflating memory and actor-forward compute while saving only env-sim. Strictly dominated by B1′ on compute and memory.
- **Per training-step bucketing (option A):** meets compute savings and cross-step distribution with far less change, but the user explicitly wants a single training step to see multiple buckets; rejected on that basis.

## 4. Architecture

Train-side only; eval is untouched (single global `max_steps_per_rollout_epoch` / `max_episode_steps`, `auto_reset=True` path unchanged).

A **bucket** is a GT-action-length interval `[lo, hi)` with bound
`max_steps_per_rollout_epoch = ceil_to_multiple(hi * max_steps_ratio, num_action_chunks)`
and `max_episode_steps = max_steps_per_rollout_epoch` (so an episode of length ≤ hi finishes within the collection window with slack; see §8 Edge Case 8 for partial-episode handling).

Within one training step, the `rollout_epoch` inner-epochs cycle through buckets (round-robin over the **curriculum-unlocked** set). Each inner-epoch: all envs run the current bucket's episodes with that bucket's `n_chunk_steps`; the result is kept as a separate fixed-length sub-result. After all inner-epochs, the env worker sends a **list** of sub-results to the actor. The actor zero_grads, computes advantages per sub-result (per-group normalization), filters failed groups for long buckets, then for each sub-result runs `update_epoch` forward+backward passes (accumulating into the same grads), and performs one `optimizer.step()`.

Cross training step, a `BucketScheduler` unlocks longer buckets by curriculum rule.

**Invariant:** per-inner-epoch tensors remain fixed-length; variable length occurs only *between* inner-epochs; a single optimizer step sees all buckets.

## 5. Components and Changes

### 5.1 `rlinf/envs/habitat/extensions/allocator.py`

- Reuse `subgoal_pipeline/bin_by_action_length.py` binning logic: bin episodes by `len(gt_data[ep]["actions"])` into intervals of `action_length_bin_size` (default 50), upper-bounded by `cfg.max_gt_action_length` (rxr 150 → `[[0,50),[50,100),[100,150]]`).
- For each bucket, call the existing `random_episode_sequences` (train mode, `auto_reset=False`) to produce `episode_sequences[process][group][slot]`, and attach `max_steps_per_rollout_epoch`, `max_episode_steps`.
- Add `build_habitat_bucketed_plan(cfg, *, num_group, total_num_processes) -> dict` returning:
  ```
  {
    "config_path", "overrides",
    "bucket_ids": [b0, b1, ...]            # sorted ascending by length
    "bucket_plans": {
       b_id: {"max_steps_per_rollout_epoch": int, "max_episode_steps": int,
              "episode_sequences": [[per process][per group][episode_id list]]}
    }
  }
  ```
- Refactor `build_habitat_global_plan` to call `build_habitat_bucketed_plan` and, for backward compatibility, default to a single bucket spanning `[0, max_gt_action_length]` (i.e., the current behavior) when bucketing is disabled.

### 5.2 `rlinf/envs/habitat/habitat_env.py` + `rlinf/envs/habitat/venv.py`

- Implement `update_reset_state_ids(self, bucket_plan)` (replacing `pass`): for each group, set each env's active episode list to `bucket_plan["episode_sequences"][self.seed_offset][group_id]` via a new worker RPC that calls `env.habitat_env.episodes = subset` (habitat's `episodes` setter; no sim rebuild), then reset. The sequential non-shuffled iterator (`habitat_env.py:137-138`, `shuffle=False`) starts at `list[0]` after the swap.
- Add worker RPC commands (mirroring `get_current_episode_metadata` at `venv.py:194-200`):
  - `set_active_episodes(subset_per_env)`: swaps the active episode subset per env.
  - `set_next_episode_by_id(env_idx, episode_id)`: forces the next reset to a specific episode (optional, for exact group alignment if needed).
- **Preserve GRPO group alignment (Constraint, §8 Edge Case 7):** all `group_size` envs in a group must land on the same episode. The current mechanism already achieves this; bucket switching must not break it. Add an assertion during implementation.

### 5.3 `rlinf/workers/env/env_worker.py`

- Replace the init-time scalar `self.n_train_chunk_steps` with a per-bucket lookup: `n_chunk_steps_for(bucket) = bucket_plans[bucket]["max_steps_per_rollout_epoch"] // num_action_chunks`.
- In `_run_interact_once`, the `for epoch in range(self.rollout_epoch)` loop becomes:
  1. `bucket = bucket_schedule[epoch]`
  2. `self.env_list[i].update_reset_state_ids(bucket_plans[bucket])` (switch active episodes + reset)
  3. collect `n_chunk_steps_for(bucket)` chunk-steps into a **per-epoch sub-result** (not stacked into one accumulated result)
  4. `finish_rollout()` at epoch end
- After the loop, send the **list** of per-epoch sub-results to the actor.
- `_inject_habitat_global_plan` now stores the bucketed plan; existing `episode_sequences` single-bucket consumers continue to work via §5.1 backward-compat path.

### 5.4 `rlinf/workers/rollout/hf/huggingface_worker.py`

- `generate_one_epoch` receives the current epoch's bucket and uses `n_chunk_steps_for(bucket)` as its loop bound (same source of truth as the env worker, driven by the broadcast schedule).

### 5.5 `rlinf/runners/embodied_runner.py`

- Add `BucketScheduler`:
  - State: unlocked bucket set, current training step, per-bucket recent success rate (from `MetricLogger`).
  - `step() -> bucket_schedule`: returns `[bucket_id] * rollout_epoch` via round-robin over the unlocked set (when `rollout_epoch > len(unlocked)`, round-robin repeats).
  - Curriculum unlock rule: every `curriculum_interval` training steps **or** when the success rate of the currently-longest unlocked bucket ≥ `curriculum_success_threshold`, unlock the next bucket. Optional warmup: first `warmup_steps` steps use only `b0`.
  - Resumable: state saved in checkpoint (§8 Edge Case 6).
- Broadcast `bucket_schedule` (and the bucket→`max_steps_per_rollout_epoch`/`max_episode_steps` table) to env worker, rollout worker, and actor at the start of each training step.

### 5.6 `rlinf/data/embodied_io_struct.py` + `rlinf/workers/actor/fsdp_actor_worker.py`

- `EmbodiedRolloutResult`: add the ability to hold per-inner-epoch sub-results; `to_trajectory` returns `list[Trajectory]` (one fixed-length trajectory per bucket-epoch).
- `recv_rollout_trajectories`: accept the list.
- `_process_received_rollout_batch`: for each trajectory, compute advantages independently with **per-group** normalization (confirm `normalize_advantages` is group-local, not global; if global, change to per-group — §7 Stabilization 1).
- Training loop: `zero_grad`; for each trajectory: filter failed groups for long buckets (§7.3), then for `ue in range(update_epoch)`: forward + backward (accumulate into the same grads); finally one `optimizer.step()`.
- Failed-group filter: extend `filter_rewards` (`fsdp_actor_worker.py:1216-1269`) to `filter_groups(bucket, success_mask)`, dropping whole groups with `success=False` for buckets above a configurable length threshold.

## 6. Per-Training-Step Data Flow

```
runner: BucketScheduler.step() -> bucket_schedule = [b0, b1, b2, b0]   (curriculum-gated)
runner broadcasts schedule + bucket table -> env/rollout/actor

for epoch in range(rollout_epoch):
    bucket = bucket_schedule[epoch]
    env_worker: update_reset_state_ids(bucket_plans[bucket])          # swap active episodes, reset
    env_worker + rollout_worker: collect n_chunk_steps[bucket] steps  # bucket-specific length
    finish_rollout()
env_worker -> actor: send [sub_result_b0, sub_result_b1, sub_result_b2, sub_result_b0]

actor: zero_grad
  for each sub_result:
      compute_adv (per-group norm)
      filter_groups(bucket, success_mask) for long buckets
      for ue in range(update_epoch): forward + backward (grad accumulate)
actor: optimizer.step()                                               # one step sees full distribution
```

## 7. Stabilization Levers

1. **Per-group advantage normalization (confirm).** GRPO normalizes advantages within each `group_size` group (same episode, different sampling). A long episode where all group members fail has ~zero advantage variance → ~zero gradient → **auto-downweighted**. Action: verify in `fsdp_actor_worker` that normalization is group-local; change to per-group if currently global.
2. **Curriculum bucket unlocking.** Start with only `b0` (shortest); unlock longer buckets over training (§5.5). Directly counteracts "long = hard = early failure".
3. **Failed-group filtering for long buckets.** Drop `success=False` groups for long buckets before the gradient contribution (§5.6), reducing noise from failed long rollouts. Reuses the `filter_rewards` mechanism.

## 8. Edge Cases

1. **Bucket episode count insufficient to fill `total_num_envs / group_size` distinct episodes per epoch:** at plan-build time (`build_habitat_bucketed_plan`), merge the bucket with an adjacent bucket (adaptive bins) until the merged episode count is sufficient; if the last remaining merged bucket is still insufficient, drop it from `bucket_ids` (the curriculum never unlocks it) and log via `logger.warning`. Merging is static, not per-step.
2. **`max_steps_per_rollout_epoch` must be a multiple of `num_action_chunks`:** `hi * max_steps_ratio` is `ceil`-ed up to the nearest multiple.
3. **Train-mode allocator divisibility (`episode_count // (total_num_processes * num_group)`):** reuse `get_trimmed_episode_count` trim; insufficient counts fall into Edge Case 1.
4. **`rollout_epoch` not a multiple of the unlocked-bucket count:** round-robin permits repeats; no deadlock.
5. **Eval unaffected:** eval keeps single global `max_steps_per_rollout_epoch`/`max_episode_steps`, `auto_reset=True` path untouched.
6. **Resume:** checkpoint must save `BucketScheduler` state (unlocked set, current step, success-rate window); `runner.resume_dir` restores it.
7. **GRPO group alignment drift:** after a bucket subset swap, all `group_size` envs in a group must still reset to the same episode. Add an assertion during implementation; use `set_next_episode_by_id` if list-position-based alignment is insufficient.
8. **Episode not finished within the collection window:** handled by `dones`/`loss_mask` (existing `compute_loss_mask` at `fsdp_actor_worker.py:1200-1213`); partial-episode reward uses `subgoal_progress` (no special-casing).
9. **`update_epoch` over per-bucket batches:** each sub-result is independently re-iterated `update_epoch` times (PPO-style); all gradients accumulate; one `optimizer.step()`.

## 9. Configuration (new keys under `env.train`)

```yaml
env:
  train:
    action_length_bin_size: 50          # or `action_length_bins: [[0,50],[50,100],[100,150]]` explicit
    max_steps_ratio: 1.3                # max_steps_per_rollout_epoch = ceil(hi * ratio, num_action_chunks)
    bucket_curriculum_enabled: true
    curriculum_interval: 50             # unlock next bucket every N steps (alternative to success-rate)
    curriculum_success_threshold: 0.3   # unlock next bucket when longest-unlocked SR >= threshold
    warmup_steps: 0                     # first N steps: only b0
    bucket_schedule_seed: 42
    # max_gt_action_length (existing) bounds the last bucket's hi
```

`max_steps_per_rollout_epoch` and `max_episode_steps` become derived per-bucket and are no longer single static values on `env.train` (kept on `env.eval` as before).

## 10. Testing

- **Unit: `allocator`** — bucket plan construction; bin boundary (gt == `hi` belongs to next bin); Edge Case 1 (merge), Edge Case 2 (ceil to multiple), Edge Case 3 (trim); backward-compat single-bucket path.
- **Unit: `BucketScheduler`** — round-robin schedule; curriculum unlock by interval and by success rate; resume state save/restore.
- **Unit: actor grad-accum** — feed a list of unequal-length trajectories; assert one `optimizer.step()`, correct accumulated gradient, per-group normalization, failed-group filtering.
- **Integration/e2e (small dataset, 1–2 training steps):** assert per-epoch `n_chunk_steps` varies by bucket, actor receives multiple bucket batches, `loss_mask` correctness, GRPO group alignment assertion holds.
- **Extend** `tests/unit_tests/envs/test_habitat_allocator.py` to cover the bucketed plan.

## 11. Out of Scope

- Eval bucketing (eval stays uniform).
- Variable length *within* an inner-epoch (only between inner-epochs).
- Per-bucket reward clamping / KL-clip tightening (lever 4, deferred; revisit if long-bucket instability persists).
- Bucket sampling weights within a step (lever 5, deferred; round-robin is the default).
