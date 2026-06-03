# Habitat Subgoal Progress Reward Design

## Goal

Design a Habitat R2R reward that uses the generated subgoal dataset to provide dense, interpretable learning signal for RL. The reward should replace the current terminal `success + nDTW` reward on the Habitat R2R UniNaVid/CMA path and should not affect other environments or generic RLinf model flows.

The target dataset is `VLN-CE/datasets/r2r/train/r2r_train_with_subgoals.json`. It keeps the same episode schema as the original R2R train split, but replaces each episode's single final `goals` entry with multiple waypoint-like goals. There is no separate `subgoals` field.

## Success Criteria

- A successful episode receives comparable total success-related reward regardless of how many subgoals it contains.
- Intermediate progress reward is normalized so each progress step has bounded scale.
- The agent can advance past a subgoal without exact positioning, but only receives subgoal success bonus for high-precision arrival.
- Premature `stop` is penalized until all subgoals have been completed.
- The reward remains local to the Habitat R2R path and does not change other VLA model or environment reward behavior.

## Reward Mode

Use a stateful active-subgoal reward. This mode completely replaces terminal `success + nDTW` for this experiment.

Out of scope for this design:

- retaining nDTW as an auxiliary reward
- changing GRPO advantage computation
- changing rollout grouping or sampling
- adding fallback trajectories or dynamic re-sampling
- changing the policy action space

## Per-Environment State

Each vectorized env instance maintains:

```text
active_subgoal_index
previous_distance_to_active_subgoal
initial_distance_to_active_subgoal
completed_subgoal_count
subgoal_success_given
non_positive_progress_steps
```

`subgoal_success_given` is a per-episode boolean sequence with length `num_subgoals`.

State resets when an env resets to a new episode.

## Distance Semantics

The active target is:

```text
episode.goals[active_subgoal_index]
```

Distance should be computed explicitly as geodesic distance from the current agent position to the active subgoal position:

```text
D_geo(agent_position, episode.goals[active_subgoal_index].position)
```

Do not rely on Habitat's default `distance_to_goal` semantics for multi-goal episodes, because it may not represent the currently active subgoal.

## Subgoal Thresholds

Use two separate thresholds:

```text
subgoal_switch_distance = 1.0
subgoal_success_distance = 0.5
```

`subgoal_switch_distance` controls state-machine advancement. When the agent reaches this threshold for the active subgoal, the active subgoal advances to the next one.

`subgoal_success_distance` controls high-precision milestone reward. The agent receives subgoal success bonus only if the active subgoal is within this stricter threshold before it is switched away.

If the agent reaches `subgoal_switch_distance` but not `subgoal_success_distance`, the active subgoal is advanced and that subgoal's success bonus is permanently lost.

## Progress Reward

For the current active subgoal:

```text
progress_delta =
    D_geo(s_{t-1}, active_subgoal)
    - D_geo(s_t, active_subgoal)
```

Normalize by the distance recorded when this subgoal became active:

```text
normalized_progress =
    clip(
        progress_delta / initial_distance_to_active_subgoal,
        -1.0,
        1.0,
    )
```

Then:

```text
r_progress = progress_reward_coef * normalized_progress
```

`initial_distance_to_active_subgoal` is recorded at subgoal activation time, not at episode start for all subgoals.

If `initial_distance_to_active_subgoal` is already less than or equal to zero due to reset or numerical edge cases, clamp the denominator to a small positive epsilon before division.

## Subgoal Success Reward

When the active subgoal satisfies:

```text
distance_to_active_subgoal <= subgoal_success_distance
```

and that subgoal has not already received success bonus:

```text
r_subgoal_success =
    subgoal_success_reward_coef / num_subgoals
```

The total subgoal success bonus for an episode is bounded by:

```text
sum(r_subgoal_success) <= subgoal_success_reward_coef
```

This makes episodes with different subgoal counts comparable when they complete all subgoals precisely.

## Subgoal Switching

After reward terms for the current step are computed, switch the active subgoal when:

```text
distance_to_active_subgoal <= subgoal_switch_distance
```

Switching should:

- increment `active_subgoal_index`
- increment `completed_subgoal_count`
- record `initial_distance_to_active_subgoal` for the next active subgoal
- reset `previous_distance_to_active_subgoal` for the next active subgoal
- reset `non_positive_progress_steps`

If the active subgoal is the final subgoal, switching marks all subgoals finished. The agent still needs to output `stop` to receive final stop success reward.

## Stall Penalty

Use a patience-based penalty:

```text
if normalized_progress <= 0:
    non_positive_progress_steps += 1
else:
    non_positive_progress_steps = 0

if non_positive_progress_steps >= stall_patience:
    r_penalty = -stall_penalty_coeff
else:
    r_penalty = 0
```

The penalty continues every step after the patience threshold until positive normalized progress resumes or the active subgoal changes.

Recommended initial defaults:

```text
stall_patience = 3
stall_penalty_coeff = 1.0
```

## Stop Reward

For non-stop actions:

```text
r_stop = 0
```

For `stop` actions before all subgoals are finished:

```text
r_stop = -premature_stop_coeff
```

For `stop` actions after all subgoals are finished:

```text
success_scale =
    1.0 - min(distance_to_final_goal / final_success_distance, 1.0)

r_stop =
    stop_success_reward_coef * success_scale
```

`distance_to_final_goal` should use the final subgoal position, which is the final entry in `episode.goals`.

`final_success_distance` can default to Habitat's existing success distance, typically `3.0`, unless the experiment explicitly chooses a different terminal stop threshold.

## Total Reward

Each step reward is:

```text
R_t =
    r_progress
    + r_subgoal_success
    + r_penalty
    + r_stop
```

Reward scale bounds:

```text
r_progress              in [-progress_reward_coef, progress_reward_coef]
sum(r_subgoal_success)  in [0, subgoal_success_reward_coef]
r_stop_success          in [0, stop_success_reward_coef]
r_premature_stop        = -premature_stop_coeff
r_penalty_per_step      = -stall_penalty_coeff
```

## Single-Goal Episodes

If an episode has only one goal, treat it as a one-subgoal episode:

```text
num_subgoals = 1
active_subgoal_index = 0
```

The reward becomes final-goal progress plus one possible subgoal success bonus plus final stop reward.

## Implementation Scope

Keep implementation local to Habitat R2R reward handling:

- `rlinf/envs/habitat/`
- `rlinf/envs/habitat/extensions/`
- Habitat R2R config files under `examples/embodiment/config/`

If a shared-path change becomes unavoidable, isolate it behind an explicit Habitat R2R and model-specific condition so other VLA models keep their existing reward behavior.

## Configuration

Add explicit config fields for the Habitat R2R path:

```text
reward_mode: subgoal_progress
progress_reward_coef
subgoal_success_reward_coef
subgoal_switch_distance
subgoal_success_distance
stop_success_reward_coef
final_success_distance
premature_stop_coeff
stall_patience
stall_penalty_coeff
```

The old Habitat terminal reward coefficients should not control this mode:

```text
success_reward_coef
ndtw_reward_coef
```

These can remain for configs that still use terminal reward, but `subgoal_progress` should use the new coefficient names to avoid semantic ambiguity.

## Verification Plan

### Dataset Verification

Confirm that:

- episodes loaded from the subgoal dataset have one or more entries in `goals`
- `num_subgoals` matches `len(episode.goals)`
- final goal is the final entry in `episode.goals`

### Reward Unit Verification

Manually test reward computation for:

- positive progress toward active subgoal
- negative progress away from active subgoal
- progress normalization and clipping
- subgoal success bonus at distance `<= 0.5`
- switch without bonus at `0.5 < distance <= 1.0`
- no bonus recovery after switching
- premature stop before all subgoals are complete
- final stop after all subgoals are complete
- one-goal episode behavior

### Integration Verification

Run a short Habitat R2R rollout and check that:

- reward is nonzero on intermediate steps
- active subgoal index increases during navigation
- subgoal success total does not exceed `subgoal_success_reward_coef`
- premature stop produces negative reward
- final stop reward is scaled by final distance
- no nDTW reward is added in `subgoal_progress` mode

### Training-Level Verification

Compare the first training updates against the terminal-only baseline:

- reward distribution should be denser and less binary
- failed trajectories should receive graded progress signal
- successful episodes with different subgoal counts should have comparable success-related reward scale
- TensorBoard or metrics logs should expose enough components to audit reward attribution

Recommended logged components:

```text
r_progress
r_subgoal_success
r_penalty
r_stop
active_subgoal_index
completed_subgoal_count
distance_to_active_subgoal
normalized_progress
```
