# Geodesic-Distance Sub-goal Dataset Builder — Design

- Date: 2026-07-07
- Status: Draft (pending user review)
- Scope: Add a non-GPT sub-goal selection pipeline that places sub-goals at fixed
  geodesic-distance intervals along the replayed GT trajectory.

## Background

`subgoal_pipeline/build_dataset.py` currently uses a GPT model to pick landmark
keyframes as sub-goals. We want a deterministic, GPT-free alternative that marks
a sub-goal every `D` meters of geodesic distance, drops a trailing sub-goal that
is closer than `D` to the final goal, and always keeps the original episode final
goal unchanged.

The geodesic distance is provided by Habitat's
`sim.geodesic_distance(source, target)` (already used at
`rlinf/envs/habitat/venv.py:73`), which returns the navmesh shortest-path
distance between two 3D positions.

## Goals

1. Select intermediate sub-goals so that consecutive sub-goals are ~`D` meters
   apart in geodesic distance (greedy "accumulate from the previous sub-goal"
   semantics).
2. Iteratively drop trailing intermediate sub-goals whose geodesic distance to
   the **original episode goal** is `< D`, so no sub-goal sits redundantly close
   to the final goal.
3. Keep each episode's final goal position identical to the original
   `train.json.gz` goal (already enforced by `subgoals_to_goals` /
   `force_original_final_goal`).
4. Reuse existing replay / artifact / video / gt infrastructure; leave the GPT
   pipeline untouched.

## Non-goals

- No GPT, no frame sampling, no token usage, no `min_step_gap`.
- No modification to `shift_subgoals.py` or `regularize_spacing.py`. Spacing is
  regular by construction; those GPT-specific post-processors are unnecessary
  here. If ever needed on geodesic output, add an `--artifact_name` flag later.

## Architecture

New file `subgoal_pipeline/build_dataset_geodesic.py`, launched via
`python3 -m subgoal_pipeline.build_dataset_geodesic ...`.

Reused modules (no changes):

- `gt.py`: `load_ground_truth_trajectories`, `build_dataset_indices`,
  `select_trajectory_groups`.
- `replay.py`: `replay_gt_actions_in_memory` (produces `List[MemoryStep]`; the
  last step is the STOP frame).
- `artifacts.py`: `build_episode_meta`, `enrich_subgoals_from_memory`,
  `validate_subgoal_payload`, `write_source_episode_artifact`,
  `subgoals_to_goals`.
- `common.py`: `load_json`, `write_json`, `append_jsonl`,
  `prepare_habitat_data_path`, `instruction_text`, `extract_goal_payload`,
  `detect_dataset_type`, `is_english_instruction`.
- `video.py`: `write_subgoal_video_from_memory`,
  `write_topdown_image_from_memory`.
- `rlinf/envs/habitat/extensions/measures.py`: `euclidean_distance` (fallback).

The main loop mirrors `build_dataset_online` but replaces the OpenAI call +
`sanitize_model_subgoals` with `select_geodesic_subgoal_steps`. Resume logic,
trajectory-group selection, dataset output assembly, and visualization are
shared in shape (copied, not refactored — see *Surgical changes* below).

### Artifact naming

- Per-trajectory artifact: `subgoals_geodesic.json` (not `subgoals_openai.json`,
  to avoid confusing GPT and geodesic outputs).
- Resume glob: `traj*/subgoals_geodesic.json`.
- No `subgoals_openai_usage.jsonl` (no tokens to log).

## Core selection algorithm

A thin wrapper plus a pure selection function. The wrapper handles the
`inf`/`nan` fallback:

```python
def _geodesic(sim, a, b, *, episode_label="", step_idx=None):
    d = float(sim.geodesic_distance(a, b))
    if math.isfinite(d) and d >= 0:
        return d
    fallback = euclidean_distance(a, b)
    print(
        f"  !! geodesic_distance not finite ({d!r}); "
        f"falling back to euclidean={fallback:.3f} "
        f"ep={episode_label} step={step_idx} a={list(a)} b={list(b)}"
    )
    return float(fallback)


def select_geodesic_subgoal_steps(steps, sim, subgoal_distance, final_goal_position):
    # steps[0] = start (not a sub-goal); steps[-1] = STOP frame.
    selected: List[int] = []  # indices into `steps`
    if len(steps) < 2:
        return selected
    anchor = steps[0].agent_position
    for i in range(1, len(steps)):
        d = _geodesic(sim, anchor, steps[i].agent_position)
        if d >= subgoal_distance:
            selected.append(i)
            anchor = steps[i].agent_position
    while selected:
        last_pos = steps[selected[-1]].agent_position
        if _geodesic(sim, last_pos, final_goal_position) < subgoal_distance:
            selected.pop()
        else:
            break
    return selected
```

### Final sub-goal

- `best_step = steps[-1].step` (the STOP frame).
- `subgoal_position` and `agent_position_at_keyframe` overridden to the original
  episode goal position (from `extract_goal_payload(episode)`), matching
  `force_original_final_goal` semantics.
- `is_final_goal = True`.
- `agent_rotation_at_keyframe` from the STOP frame.

### Edge cases

- Total path `< D`: `selected` empty → sub-goals list contains only the final
  goal. `validate_subgoal_payload` accepts this (non-empty, last is final).
- `geodesic_distance` returns `inf`/`nan` (navmesh disconnected): fall back to
  `euclidean_distance` and print a record (episode, step, both endpoints, raw
  value, fallback value).
- Final goal unreachable from a selected sub-goal: same `inf`/`nan` fallback
  applies in the tail-cleanup loop.

## Payload and downstream compatibility

The payload is field-compatible with the GPT version so `subgoals_to_goals`,
`validate_subgoal_payload`, and the video writers work unchanged.

Field differences vs the GPT payload:

| Field | GPT version | Geodesic version |
|---|---|---|
| `model`, `reasoning_effort` | present | removed |
| `token_usage` | present | removed |
| `subgoal_instruction_source` | present | removed (no language source) |
| `sampled_frame_steps` | GPT-sampled frames | selected sub-goals' `best_step` list |
| `landmark` | GPT text | `""` |
| `landmark_source` | GPT text | `"geodesic_distance_milestone"` |
| `reason` | GPT text | `"auto-selected at geodesic interval"` |
| `goal_position_source` | `"final_subgoal_agent_position"` | `"original_train.goals[0].position"` |
| new | — | `subgoal_selection_method: "geodesic_distance"` |
| new | — | `subgoal_distance: <D>` |

Dataset-level `_subgoal_generation`:

- `script` → `"subgoal_pipeline.build_dataset_geodesic"`.
- Remove `model`, `reasoning_effort`.
- Add `subgoal_distance`.
- `failures` retained (records per-trajectory exceptions, same shape minus GPT
  fields).

## CLI parameters

Inherited from `build_dataset.py` (still relevant):

`--config --split --train_json --gt_json --scenes_dir --out_dir --output_json
--max_trajectories --target_episodes --max_gt_actions --min_gt_actions
--subgoal_radius --include_unselected --pretty_output --overwrite
--continue_on_error --dry_run_selection --no_video --video_fps
--video_highlight_frames --no_topdown --topdown_max_size`

Removed (GPT-only):

`--model --reasoning_effort --max_output_tokens --max_frames --frame_stride
--min_step_gap --image_format --jpeg_quality --base_url --api_key --user-agent
--usage_log_name`

Added:

- `--subgoal_distance` (float, **required**, no default) — the geodesic interval
  `D` in meters.

Example:

```bash
python3 -m subgoal_pipeline.build_dataset_geodesic \
  --config rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml \
  --split train \
  --train_json VLN-CE/datasets/rxr/train/train_guide_reachable.json.gz \
  --gt_json VLN-CE/datasets/rxr/train/train_guide_gt_reachable.json.gz \
  --scenes_dir VLN-CE/scene_dataset \
  --target_episodes 2000 --max_gt_actions 200 --min_gt_actions 150 \
  --out_dir results/rxr_subgoal_geodesic_train2000 \
  --output_json VLN-CE/datasets/rxr/train/train_guide_subgoals_geodesic.json.gz \
  --subgoal_distance 3.0 --overwrite
```

## Testing

- New unit test `tests/unit_tests/test_geodesic_subgoal_selection.py`.
- `select_geodesic_subgoal_steps` tested with a fake `sim` whose
  `geodesic_distance` returns a scripted value table, plus a constructed
  `List[MemoryStep]`. Cases:
  1. Evenly spaced selection at multiples of `D`.
  2. Tail sub-goal within `D` of final is dropped (single removal).
  3. Two trailing sub-goals both within `D` of final → both removed (iterative).
  4. Path shorter than `D` → empty intermediates, only final goal remains.
  5. `geodesic_distance` returns `inf` once → euclidean fallback used, selection
     still proceeds.
- No Habitat end-to-end test (requires scene assets); CI skips it. Selection
  correctness is fully covered by the unit test with the fake sim.

## Surgical changes

- One new file (`build_dataset_geodesic.py`) + one new test file.
- No edits to `build_dataset.py` or any existing module.
- ~100 lines of resume/output boilerplate are copied rather than refactored into
  a shared helper, to avoid entangling the GPT and geodesic paths and to keep the
  diff minimal. Acceptable for a pipeline script.
