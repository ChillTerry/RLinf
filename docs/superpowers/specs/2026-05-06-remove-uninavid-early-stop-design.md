# Remove UniNaVid Habitat Early Stop

## Context

The current UniNaVid Habitat evaluation path contains model-specific EARLY_STOP behavior in `rlinf/envs/habitat/habitat_env.py`. That behavior can rewrite actions to `stop` when a rotation-count or step-count heuristic fires, then participates in chunk-level reset handling. This makes UniNaVid eval diverge from the CMA Habitat behavior where `auto_reset=True` and `ignore_terminations=True` let environments continue until `max_episode_steps`, while first-done episode metrics are cached and overlaid onto live metrics.

The desired behavior is to remove UniNaVid EARLY_STOP entirely, including its config surface, while preserving the corrected `initial_distance_to_goal` handling used for SPL.

## Goals

- Delete UniNaVid EARLY_STOP runtime behavior and related config-facing switches.
- Align Habitat eval reset and metrics behavior with `remotes/origin/feature/vln-train-cma`.
- Preserve UniNaVid-specific observation/action compatibility that is not EARLY_STOP related.
- Preserve the `initial_distance_to_goal` fix:
  - initialize with `np.nan`;
  - reset per-env values on reset;
  - seed from current Habitat metrics when available;
  - fall back to first step metrics only when still unset.
- Keep changes scoped to `rlinf/envs/habitat/` and Habitat/UniNaVid config files if config cleanup is needed.

## Non-Goals

- Do not modify RLinf generic runner, worker, scheduler, or rollout flow unless a direct bug prevents Habitat eval from using the desired semantics.
- Do not introduce alternate heuristics, fallbacks, or post-processing to replace EARLY_STOP.
- Do not change UniNaVid model inference or prompt/action parsing.
- Do not attempt to preserve old EARLY_STOP-influenced metric values.

## Proposed Design

Use the CMA Habitat step/reset model as the behavioral target:

```python
truncations = self.elapsed_steps >= self.max_episode_steps
dones_for_metric_save = terminations | truncations
metric_save_masks = dones_for_metric_save & (~self.dones_once)
if metric_save_masks.any():
    self._save_metrics(infos, metric_save_masks)

self._overlay_first_done_episode_metrics(infos)

if self.ignore_terminations:
    terminations[:] = False
dones = terminations | truncations

if dones.any() and self.auto_reset:
    obs, infos = self._handle_auto_reset(dones, obs, infos)
```

The model should not rewrite policy actions based on distance stagnation, rotation count, or elapsed step thresholds. In eval, an early model-generated `STOP` may still produce the first-done metrics, but `ignore_terminations=True` prevents that termination from resetting the env. The env continues to the shared `max_episode_steps` truncation, and reset occurs there. This keeps episode count controlled by:

```text
max_steps_per_rollout_epoch / max_episode_steps * total_num_envs
```

Metrics remain accurate because `_save_metrics()` records each env only on its first done, and `_overlay_first_done_episode_metrics()` keeps those first-done values visible in later live metrics.

## Code Changes

In `rlinf/envs/habitat/habitat_env.py`, remove:

- `_UNINAVID_ORIGINAL_EARLY_STOP_ROTATION`;
- `_UNINAVID_ORIGINAL_EARLY_STOP_STEPS`;
- `_uninavid_early_stop_last_distance_to_goal`;
- `_uninavid_early_stop_rotation_counts`;
- `_uninavid_original_early_stop_enabled()`;
- `_ensure_uninavid_early_stop_state()`;
- `_reset_uninavid_early_stop_state()`;
- `_apply_uninavid_original_early_stop()`;
- `_update_uninavid_original_early_stop_state()`;
- `_habitat_step_trace_dir()`;
- `_append_habitat_step_trace()`;
- `step()` logic that rewrites proposed actions or emits early-stop trace records;
- `step()` emission of `_reset_dones_mask`;
- `chunk_step()` logic that temporarily disables `auto_reset`, accumulates deferred resets, or no-ops envs after mid-chunk done.

Keep:

- `_squeeze_singleton_action_dim()` for UniNaVid action shape compatibility;
- `_format_habitat_actions()` if the current UniNaVid Habitat vector env requires dict-style actions;
- `_uninavid_use_raw_rgb_enabled()` and raw RGB observation handling;
- `_attach_uninavid_chunk_history()` for UniNaVid chunked visual history;
- `action_map` support for UniNaVid `no_op` action id if the model/action-prep path may emit it;
- first-done metrics caching and overlay;
- `initial_distance_to_goal` reset/seed/fallback logic.

In config files, remove EARLY_STOP-facing keys if present:

- `uninavid_original_early_stop`;
- `uninavid_early_stop_rotation`;
- `uninavid_early_stop_steps`;
- `habitat_step_trace_dir` when used only for EARLY_STOP tracing.

No generic RLinf config validation change is expected unless these fields are declared in a schema elsewhere.

## Metrics Impact

This intentionally changes UniNaVid eval metrics when old EARLY_STOP behavior was active. Metrics will no longer represent an artificial stop inserted by rotation/step heuristics. They will represent the model's real first `STOP` or the `max_episode_steps` truncation point.

Expected effects:

- `num_trajectories` should stay governed by rollout configuration.
- `success`, `spl`, `distance_to_goal`, and `trajectory_Length` may change for runs where EARLY_STOP previously fired.
- `spl` should keep the corrected initial-distance denominator due to the preserved `initial_distance_to_goal` handling.

## Verification

Run targeted static checks first:

```bash
rg -n "EARLY_STOP|early_stop|uninavid_early_stop|habitat_step_trace|_reset_dones_mask" rlinf/envs/habitat examples/embodiment/config
python -m compileall rlinf/envs/habitat
```

If the environment dependencies are available, run a small UniNaVid eval smoke test with `habitat_r2r_eval_uninavid` and confirm:

- no early-stop action rewrite appears in code or logs;
- eval reset happens at `max_episode_steps` under `auto_reset=True` and `ignore_terminations=True`;
- saved metrics files are written once per episode id;
- aggregate `num_trajectories` matches the configured episode count formula.

If Habitat/UniNaVid assets or dependencies are unavailable, report that the runtime smoke test could not be executed and include the completed static verification.

## Acceptance Loop

After implementation, run the UniNaVid Habitat eval using the same acceptance setup represented by:

```text
results/uninavid_habitat/20260505T160803Z-acceptance/batched_candidate_greedy_v2
```

Use that directory as the reference for the run mode and metric floor. Its recorded RLinf baseline is:

```json
{
  "episodes_valid": 100,
  "success_rate": 0.49,
  "spl": 0.45595802783966066
}
```

The implementation is not complete if `success_rate` or `spl` drops by more than 2 absolute percentage points from that baseline. In decimal metric form, this means the new run must satisfy:

```text
success_rate >= 0.47
spl >= 0.43595802783966064
```

If either metric fails this floor, continue the implementation/debugging loop until both metrics are within the 2-point tolerance. Fixes in that loop must preserve the design goals above: do not reintroduce EARLY_STOP, action-rewrite heuristics, or post-processing bandages to recover the metric.
