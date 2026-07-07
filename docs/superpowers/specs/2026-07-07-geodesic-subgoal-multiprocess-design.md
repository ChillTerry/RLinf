# Geodesic Sub-goal Builder: Multi-process Parallel Rendering — Design

- Date: 2026-07-07
- Status: Draft (pending user review)
- Scope: Add multi-process parallel rendering to
  `subgoal_pipeline/build_dataset_geodesic.py` so that "process-count" scenes
  are loaded and rendered simultaneously, with scenes evenly distributed (by
  count) across specified GPUs.

## Background

`build_dataset_geodesic.py` currently runs single-process: one `habitat.Env`
reused across all episodes, switching scenes via `env.current_episode = episode;
env.reset()`. Each cross-scene `reset()` triggers `sim.reconfigure` (mesh
reload) — the dominant cost. GPU is fixed at `gpu_device_id: 0` in the yaml.

Key facts established during exploration:
- Setting `cfg.habitat.dataset.content_scenes = [scene_name]` makes a worker's
  `env.episodes` contain only that scene, so the env never switches scenes
  (intra-scene resets only reposition the agent, no mesh reload).
- GPU isolation per worker = `os.environ["CUDA_VISIBLE_DEVICES"] = <gpu_id>`
  set before habitat imports; habitat `gpu_device_id=0` then picks the visible
  first GPU. No yaml change needed.
- `select_trajectory_groups` returns `(trajectory_id, episode_ids,
  representative_id)` with no scene_id, but the representative's scene is
  `source_episode_by_id[rep]["scene_id"]`.
- Workers write per-trajectory artifacts to unique dirs
  (`out_root/traj<traj_id>.../subgoals_geodesic.json`) — no path conflicts.
- Aggregation can reuse the existing resume glob-scan
  (`out_root.glob("traj*/subgoals_geodesic.json")`) to rebuild
  `subgoals_by_trajectory` in the main process with zero inter-process
  communication.
- Repo precedent for offline parallelism: `rlinf/envs/calvin/utils.py` uses
  `concurrent.futures.ProcessPoolExecutor`.

## Goals

1. Run N worker processes concurrently, each rendering one scene at a time and
   generating sub-goals for all trajectory groups in that scene.
2. Distribute scenes evenly **by count** across the specified GPUs
   (round-robin: `scene[k] -> gpu[k % G]`).
3. Pin each worker process to one GPU for its lifetime via
  `CUDA_VISIBLE_DEVICES`.
4. Preserve existing behavior when `--num_processes 1 --gpus 0` (default).
5. Reuse existing replay/select/enrich/video/topdown logic unchanged; leave the
   GPT `build_dataset.py` untouched.

## Non-goals

- No Ray (overkill for an offline tool; ProcessPoolExecutor is sufficient).
- No change to the GPT pipeline.
- No dynamic load-balancing across GPUs (even-by-count round-robin is the
  requested semantics; per-scene work imbalance is accepted).
- No Habitat end-to-end test in CI (needs scene assets + multiple GPUs).

## Architecture

Only `subgoal_pipeline/build_dataset_geodesic.py` (+ unit tests) changes.

### Main process (no habitat)

1. `select_trajectory_groups` → `groups` (existing).
2. Bucket groups by the representative's scene:
   `bucket_groups_by_scene(groups, source_episode_by_id) -> Dict[scene_id, List[group]]`.
   Sort scene ids deterministically; round-robin assign
   `scene_list[k] -> gpu_ids[k % G]`.
3. For each GPU `g`, build a `ProcessPoolExecutor(max_workers=wp[g],
   mp_context=mp.get_context("spawn"), initializer=init_worker,
   initargs=(g,))`. Submit one `process_scene` task per scene assigned to `g`.
4. `concurrent.futures.as_completed` over all pools' futures → collect
   `summary` dicts.
5. Rebuild `subgoals_by_trajectory` via the existing resume glob-scan; aggregate
   `modified`/`failed`/`skipped` from summaries; write `failures.jsonl` once;
   assemble and write `output_json` once.

### Worker process (`process_scene` task)

Lazy-import habitat (so the main process and tests never trigger habitat). Build
the config from primitives: `get_config(config_path)` + `read_write` to set
`cfg.habitat.dataset.content_scenes = [scene_id]`, `split`, `data_path`,
`scenes_dir`, `ndtw` measure `SPLIT`/`GT_PATH`, and
`top_down_map.draw_shortest_path = False`. Create `env = habitat.Env(cfg)`. For
each group in `scene_groups`: skip if the artifact already exists (resume);
else `replay_gt_actions_in_memory` → `select_geodesic_subgoal_steps` →
`enrich_subgoals_from_memory` → force final subgoal to original goal →
`validate_subgoal_payload` → `write_json(subgoals_geodesic.json)` →
`write_source_episode_artifact` → optional video/topdown (reuse existing
writers). `env.close()`. Return
`{"scene_id", "modified", "skipped", "failed": [...]}`.

### Single-process default

`--num_processes 1 --gpus 0` produces one pool with `max_workers=1` on GPU 0;
the worker processes all scenes sequentially, one env per scene. This is
behaviorally equivalent to the current single-process output (artifacts
identical) and slightly more efficient (no cross-scene reconfigure mid-scene).
One unified code path — no separate single-process branch.

## Worker task interface

`process_scene` receives only picklable primitives plus this scene's subset
(no OmegaConf object, no full `gt_trajectories`/`source_episode_by_id`):

```
process_scene(
    config_path: str, split: str, scenes_dir: str, gt_json: str,
    habitat_data_path: str,
    scene_id: str,
    scene_groups: List[Tuple[int, List[int], int]]],
    gt_subset: Dict[int, GroundTruthTrajectory],
    source_subset: Dict[int, dict],
    out_dir: str,
    subgoal_distance: float, subgoal_radius: float,
    no_video: bool, video_fps: int, video_highlight_frames: int,
    no_topdown: bool, topdown_max_size: int,
) -> Dict[str, Any]
```

- `gt_subset`/`source_subset` are sliced by the main process from the
  already-loaded `gt_trajectories`/`source_episode_by_id` for just this
  scene's representative ids (small; tens to a few hundred episodes).
- The worker reads `max_steps = int(cfg.habitat.environment.max_episode_steps)`
  from the config it builds (the main process does not build a habitat config
  in multiprocess mode, so it cannot supply this value).
- The worker reuses `replay_gt_actions_in_memory`, `select_geodesic_subgoal_steps`,
  `_build_geodesic_subgoal_dicts`, `enrich_subgoals_from_memory`,
  `validate_subgoal_payload`, `write_json`, `write_source_episode_artifact`,
  `write_subgoal_video_from_memory`, `write_topdown_image_from_memory` — all
  unchanged.
- The worker never writes `output_json` or shared `failures.jsonl`; failures
  come back via the return value.

## GPU / process allocation

- `--gpus` (comma-separated, e.g. `0,1,2,3`) → `gpu_ids`. `--num_processes N`.
- `G = len(gpu_ids)`. Require `N >= G` (else error: process count must be >= GPU
  count).
- `workers_per_gpu[g] = N // G + (1 if g < N % G else 0)` — the first `N % G`
  GPUs get one extra worker.
- Scene → GPU: `scene_list` (sorted unique scene ids) round-robin:
  `gpu_ids[k % G]` for `scene_list[k]`. Each GPU gets `floor`/`ceil` scenes.
- GPU `g`'s pool `max_workers = min(workers_per_gpu[g], len(scenes_on_g))` so
  no idle workers are spawned when a GPU has few scenes.
- `init_worker(gpu_id)`: `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`
  before any habitat import. The `ProcessPoolExecutor` `initializer` runs once
  per worker process; since habitat is imported lazily inside `process_scene`,
  the env var is set first.
- `mp_context = mp.get_context("spawn")` — avoids fork inheriting a
  CUDA/habitat-initialized parent (the main process must not import habitat).
- Worker processes are reused across scene tasks; each task creates and closes
  its own `habitat.Env` (`content_scenes=[scene]`).

## Error handling

- A single scene task that raises → its future is marked broken; the exception
  is recorded as a failure for that scene; other scenes continue; the run does
  not abort. This is the scene-level analogue of `--continue_on_error`.
- `--continue_on_error` still governs per-group behavior inside a worker: when
  set, a failing group is logged and the worker proceeds to the next group;
  when unset, a failing group raises, which surfaces as a broken scene task
  (and aborts only that scene).
- Broken-future exceptions are aggregated into `_subgoal_generation.failures`
  and `failures.jsonl` by the main process.

## Aggregation and metadata

- `as_completed` over all futures → list of `summary` dicts.
- `aggregate_summaries(summaries)` → totals: `modified`, `skipped`,
  `failed` (concatenated), `scenes_processed`, `scenes_failed`.
- Rebuild `subgoals_by_trajectory` via the existing resume glob-scan
  (`out_root.glob("traj*/subgoals_geodesic.json")`) — workers already wrote
  every completed trajectory's payload to disk, so the main process only scans.
- Main process iterates `original_episodes` → `subgoals_to_goals` → writes
  `output_json` once (unchanged from current behavior).
- `failures.jsonl` written once by the main process during aggregation (no
  concurrent appends).
- `_subgoal_generation` meta gains: `num_processes`, `gpus` (list),
  `workers_per_gpu` (list), `parallel_mode: true`, `scenes_total`,
  `scenes_per_gpu` (dict), `scenes_processed`, `scenes_failed`.

## CLI

Inherited from the existing `build_arg_parser` (all current flags stay).
Defaults `--num_processes 1` and `--gpus "0"` reproduce the current
single-process behavior.

Added:

- `--num_processes` (int, default 1) — total worker process count.
- `--gpus` (str, default `"0"`) — comma-separated GPU ids.

Validation: `N >= G`; `--gpus` must parse to non-empty list of non-negative
ints.

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
  --subgoal_distance 3.0 --num_processes 8 --gpus 0,1,2,3 --overwrite
```

## Testing

New unit test `tests/unit_tests/test_geodesic_subgoal_multiprocess.py` (no
habitat):

1. `assign_scenes_to_gpus`: given a scene list and `gpu_ids`, assert
   round-robin evenness, `floor`/`ceil` counts, and deterministic ordering.
2. `compute_workers_per_gpu`: `N` and `G` → per-GPU worker counts with correct
   remainder distribution; `max_workers` clamp when a GPU has fewer scenes than
   workers.
3. `bucket_groups_by_scene`: fake `groups` + `source_subset` → correct
   scene-keyed bucketing; a group is never split across scenes.
4. `aggregate_summaries`: multiple worker summaries → correct
   `modified`/`skipped`/`failed`/`scenes_processed`/`scenes_failed` totals.
5. `validate_num_processes_vs_gpus`: `N < G` raises; `N >= G` accepted;
   `--gpus` parsing rejects empty/malformed input.
6. Arg-parser defaults: `--num_processes 1`, `--gpus "0"`.

No Habitat multi-process end-to-end test (CI skips; needs assets + GPUs). The
`process_scene` internals are covered by the existing selector tests and reuse
the audited `build_dataset.py` env-assembly pattern.

## Surgical changes

- Modify `subgoal_pipeline/build_dataset_geodesic.py` (refactor the online loop
  into `process_scene` + a dispatching main; add pool/alloc helpers).
- Add `tests/unit_tests/test_geodesic_subgoal_multiprocess.py`.
- No edits to `replay.py`/`artifacts.py`/`gt.py`/`common.py`/`video.py` or the
  GPT `build_dataset.py`.
- README: append a multiprocess example to the existing geodesic section.
