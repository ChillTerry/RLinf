# Geodesic Sub-goal Multiprocess Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-process parallel rendering to `subgoal_pipeline/build_dataset_geodesic.py`: N worker processes render scenes concurrently, scenes evenly distributed (by count, round-robin) across specified GPUs, each worker pinned to one GPU via `CUDA_VISIBLE_DEVICES` and loading one scene at a time with `content_scenes=[scene]`.

**Architecture:** Main process (no habitat) selects trajectory groups, buckets them by scene, round-robin assigns scenes to GPUs, and dispatches one `process_scene` task per scene to a per-GPU `ProcessPoolExecutor` (spawn context, `init_worker` pins the GPU). Each worker lazy-imports habitat, builds an env with `content_scenes=[scene]`, loops its scene's groups via the extracted `process_group`, writes per-trajectory artifacts, and returns a summary. Main collects summaries, writes `failures.jsonl` once, rebuilds `subgoals_by_trajectory` via the existing resume glob-scan, and writes `output_json` once.

**Tech Stack:** Python 3.10, `concurrent.futures.ProcessPoolExecutor` with `multiprocessing.get_context("spawn")`, Habitat sim, pytest, ruff (`/opt/venv/habitat/bin/ruff`). Verified: `python -m pkg.mod` + spawn + module-level worker function + initializer works (CUDA_VISIBLE_DEVICES visible to worker).

## Global Constraints

- Only `subgoal_pipeline/build_dataset_geodesic.py` (+ new test file) changes. No edits to `replay.py`/`artifacts.py`/`gt.py`/`common.py`/`video.py` or the GPT `build_dataset.py`.
- Module-level imports must stay habitat-free; `habitat` imported lazily inside `process_scene`/`_build_scene_env_config` only. The main process and tests never import habitat.
- `process_scene`/`init_worker` are module-level (picklable for spawn). Verified `python -m` invocation works with spawn.
- Defaults `--num_processes 1 --gpus "0"` reproduce single-process behavior (one pool, one worker, GPU 0).
- Scene→GPU: round-robin by scene count (`scene_list[k] -> gpu_ids[k % G]`).
- `N >= G` required (else error). `workers_per_gpu[g] = N//G + (1 if g < N%G else 0)`, clamped to `len(scenes_on_g)`.
- Workers never write `output_json` or shared `failures.jsonl`; failures come back via return value; main writes `failures.jsonl` once and `output_json` once.
- Reuse existing `replay_gt_actions_in_memory`, `select_geodesic_subgoal_steps`, `_build_geodesic_subgoal_dicts`, `enrich_subgoals_from_memory`, `validate_subgoal_payload`, `write_json`, `write_source_episode_artifact`, `write_subgoal_video_from_memory`, `write_topdown_image_from_memory` unchanged.
- Ruff lint passes. Conventional Commits, `git commit -s`.
- Run tests from repo root `/data/RLinf` with `python -m pytest`. ruff at `/opt/venv/habitat/bin/ruff`.

---

### Task 1: Pure allocation/aggregation helpers (TDD)

**Files:**
- Modify: `subgoal_pipeline/build_dataset_geodesic.py` (add 6 module-level pure functions after `aggregate`... actually insert before `build_dataset_geodesic`, after `_build_geodesic_subgoal_dicts` ~line 105)
- Test: `tests/unit_tests/test_geodesic_subgoal_multiprocess.py` (create)

**Interfaces:**
- Produces (all pure, habitat-free):
  - `parse_gpus(spec: str) -> list[int]` — parse `"0,1,2,3"` → `[0,1,2,3]`; raise on empty/malformed.
  - `validate_num_processes_vs_gpus(num_processes: int, num_gpus: int) -> None` — raise if `num_processes < num_gpus`.
  - `bucket_groups_by_scene(groups: list[tuple[int, list[int], int]], source_episode_by_id: dict[int, dict]) -> dict[str, list[tuple[int, list[int], int]]]` — bucket groups by `source_episode_by_id[rep]["scene_id"]`.
  - `assign_scenes_to_gpus(scene_list: list[str], gpu_ids: list[int]) -> dict[int, list[str]]` — round-robin `scene_list[k] -> gpu_ids[k % len(gpu_ids)]`.
  - `compute_workers_per_gpu(num_processes: int, gpu_ids: list[int], scenes_per_gpu: dict[int, int]) -> dict[int, int]` — `N//G` + remainder, clamped to `scenes_per_gpu[g]`; skip gpus with 0 scenes.
  - `aggregate_summaries(summaries: list[dict]) -> dict` — totals `{scenes_processed, scenes_failed, processed, skipped, failed}`.

- [ ] **Step 1: Write the failing test file**

Create `tests/unit_tests/test_geodesic_subgoal_multiprocess.py`:

```python
import pytest

from subgoal_pipeline import build_dataset_geodesic as bdg


def test_parse_gpus_basic():
    assert bdg.parse_gpus("0,1,2,3") == [0, 1, 2, 3]
    assert bdg.parse_gpus("0") == [0]
    assert bdg.parse_gpus(" 1 , 2 ") == [1, 2]


def test_parse_gpus_rejects_bad():
    for bad in ["", " ", "0,a", "0,-1", "0,1.5"]:
        with pytest.raises((ValueError, RuntimeError)):
            bdg.parse_gpus(bad)


def test_validate_num_processes_vs_gpus():
    bdg.validate_num_processes_vs_gpus(4, 4)  # ok
    bdg.validate_num_processes_vs_gpus(8, 4)  # ok
    with pytest.raises((ValueError, RuntimeError)):
        bdg.validate_num_processes_vs_gpus(2, 4)


def test_bucket_groups_by_scene():
    # group = (trajectory_id, episode_ids, representative_id)
    groups = [(10, [1, 2], 1), (20, [3], 3), (30, [5, 6], 5)]
    source = {
        1: {"scene_id": "sceneA"},
        3: {"scene_id": "sceneB"},
        5: {"scene_id": "sceneA"},
    }
    buckets = bdg.bucket_groups_by_scene(groups, source)
    assert set(buckets.keys()) == {"sceneA", "sceneB"}
    assert [g[0] for g in buckets["sceneA"]] == [10, 30]
    assert [g[0] for g in buckets["sceneB"]] == [20]


def test_assign_scenes_to_gpus_round_robin():
    scenes = ["s0", "s1", "s2", "s3", "s4"]
    gpus = [0, 1, 2]
    assignment = bdg.assign_scenes_to_gpus(scenes, gpus)
    assert assignment[0] == ["s0", "s3"]
    assert assignment[1] == ["s1", "s4"]
    assert assignment[2] == ["s2"]


def test_compute_workers_per_gpu_even():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 5, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(8, gpus, scenes_per_gpu)
    assert out == {0: 2, 1: 2, 2: 2, 3: 2}


def test_compute_workers_per_gpu_remainder():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 5, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(9, gpus, scenes_per_gpu)
    assert out == {0: 3, 1: 2, 2: 2, 3: 2}


def test_compute_workers_per_gpu_clamp_to_scene_count():
    gpus = [0, 1, 2, 3]
    scenes_per_gpu = {0: 1, 1: 5, 2: 5, 3: 5}
    out = bdg.compute_workers_per_gpu(8, gpus, scenes_per_gpu)
    assert out[0] == 1  # clamped: only 1 scene on GPU 0
    assert out[1] == 2


def test_compute_workers_per_gpu_skips_empty_gpu():
    gpus = [0, 1]
    scenes_per_gpu = {0: 4, 1: 0}
    out = bdg.compute_workers_per_gpu(4, gpus, scenes_per_gpu)
    assert out == {0: 4}  # GPU 1 has no scenes -> skipped


def test_aggregate_summaries():
    summaries = [
        {"scene_id": "A", "processed": 3, "skipped": 1, "failed": [{"e": "x"}]},
        {"scene_id": "B", "processed": 2, "skipped": 0, "failed": [{"e": "y"}, {"e": "z"}]},
        {"scene_id": "C", "processed": 0, "skipped": 0, "failed": [{"e": "w"}]},
    ]
    agg = bdg.aggregate_summaries(summaries)
    assert agg["scenes_processed"] == 2  # A and B processed >=1
    assert agg["scenes_failed"] == 1  # C processed 0 and had failures
    assert agg["processed"] == 5
    assert agg["skipped"] == 1
    assert len(agg["failed"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_multiprocess.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'parse_gpus'`.

- [ ] **Step 3: Write the helper implementations**

Insert these functions into `subgoal_pipeline/build_dataset_geodesic.py` after `_build_geodesic_subgoal_dicts` (before `build_dataset_geodesic`):

```python
def parse_gpus(spec: str) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        raise RuntimeError("--gpus must be a non-empty comma-separated list, e.g. '0,1,2,3'.")
    gpu_ids: list[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise RuntimeError(f"--gpus has an empty entry in '{spec}'.")
        try:
            value = int(token)
        except ValueError as exc:
            raise RuntimeError(f"--gpus entry '{token}' is not an integer.") from exc
        if value < 0:
            raise RuntimeError(f"--gpus entry '{token}' must be a non-negative integer.")
        gpu_ids.append(value)
    if not gpu_ids:
        raise RuntimeError("--gpus parsed to an empty list.")
    return gpu_ids


def validate_num_processes_vs_gpus(num_processes: int, num_gpus: int) -> None:
    if num_processes < 1:
        raise RuntimeError(f"--num_processes must be >= 1, got {num_processes}.")
    if num_gpus < 1:
        raise RuntimeError(f"--gpus count must be >= 1, got {num_gpus}.")
    if num_processes < num_gpus:
        raise RuntimeError(
            f"--num_processes ({num_processes}) must be >= GPU count ({num_gpus})."
        )


def bucket_groups_by_scene(
    groups: list[tuple[int, list[int], int]],
    source_episode_by_id: dict[int, dict],
) -> dict[str, list[tuple[int, list[int], int]]]:
    buckets: dict[str, list[tuple[int, list[int], int]]] = {}
    for trajectory_id, episode_ids, representative_id in groups:
        source = source_episode_by_id.get(int(representative_id)) or {}
        scene_id = str(source.get("scene_id") or "")
        buckets.setdefault(scene_id, []).append(
            (int(trajectory_id), [int(v) for v in episode_ids], int(representative_id))
        )
    return buckets


def assign_scenes_to_gpus(scene_list: list[str], gpu_ids: list[int]) -> dict[int, list[str]]:
    if not gpu_ids:
        raise RuntimeError("gpu_ids must be non-empty.")
    assignment: dict[int, list[str]] = {int(g): [] for g in gpu_ids}
    for k, scene_id in enumerate(scene_list):
        assignment[int(gpu_ids[k % len(gpu_ids)])].append(scene_id)
    return assignment


def compute_workers_per_gpu(
    num_processes: int,
    gpu_ids: list[int],
    scenes_per_gpu: dict[int, int],
) -> dict[int, int]:
    if not gpu_ids:
        raise RuntimeError("gpu_ids must be non-empty.")
    n_gpus = len(gpu_ids)
    base = num_processes // n_gpus
    remainder = num_processes % n_gpus
    out: dict[int, int] = {}
    for idx, gpu_id in enumerate(gpu_ids):
        scene_count = int(scenes_per_gpu.get(int(gpu_id), 0))
        if scene_count == 0:
            continue
        workers = base + (1 if idx < remainder else 0)
        out[int(gpu_id)] = max(1, min(workers, scene_count))
    return out


def aggregate_summaries(summaries: list[dict]) -> dict:
    processed = 0
    skipped = 0
    scenes_processed = 0
    scenes_failed = 0
    failed: list[dict] = []
    for s in summaries:
        s_processed = int(s.get("processed", 0))
        s_failed = s.get("failed") or []
        processed += s_processed
        skipped += int(s.get("skipped", 0))
        if s_processed > 0:
            scenes_processed += 1
        elif s_failed:
            scenes_failed += 1
        failed.extend(s_failed)
    return {
        "scenes_processed": scenes_processed,
        "scenes_failed": scenes_failed,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_multiprocess.py -v`
Expected: 10 passed.

- [ ] **Step 5: Lint + import-without-habitat**

Run: `cd /data/RLinf && /opt/venv/habitat/bin/ruff check subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py`
Expected: no errors.

Run: `cd /data/RLinf && python -c "from subgoal_pipeline import build_dataset_geodesic; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py
git commit -s -m "feat: add geodesic multiprocess allocation helpers"
```

---

### Task 2: Extract worker-side functions (process_group, process_scene, init_worker)

**Files:**
- Modify: `subgoal_pipeline/build_dataset_geodesic.py` (add `process_group`, `_build_scene_env_config`, `init_worker`, `process_scene` after the Task 1 helpers; do NOT yet wire into `build_dataset_geodesic`)

**Interfaces:**
- Consumes: `select_geodesic_subgoal_steps`, `_build_geodesic_subgoal_dicts` (already in module); `replay_gt_actions_in_memory`, `MemoryStep` (replay.py); `enrich_subgoals_from_memory`, `validate_subgoal_payload`, `write_json`, `write_source_episode_artifact` (artifacts/common); `extract_goal_payload`, `instruction_text`, `prepare_habitat_data_path` (common); `write_subgoal_video_from_memory`, `write_topdown_image_from_memory` (video.py).
- Produces:
  - `process_group(env, env_episode, source_episode, gt_trajectory, trajectory_id, episode_ids, representative_id, out_root, habitat_data_path, train_json_path, subgoal_distance, subgoal_radius, max_steps, no_video, video_fps, video_highlight_frames, no_topdown, topdown_max_size, overwrite, continue_on_error, group_idx, total_groups) -> dict` — returns `{"trajectory_id": int, "status": "ok"|"skipped"|"failed", "failure": dict|None}`. Verbatim extraction of the current inline loop body (build_dataset_geodesic.py:257-398), with two changes: (a) the skip-existing branch returns `status="skipped"` instead of mutating `subgoals_by_trajectory`; (b) the except branch returns `status="failed"` with the failure dict instead of mutating `failures`/appending `failures.jsonl`.
  - `init_worker(gpu_id: int) -> None` — sets `os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)`.
  - `_build_scene_env_config(config_path, split, scenes_dir, gt_json, habitat_data_path, scene_id) -> cfg` — lazy-imports habitat, builds cfg with `content_scenes=[scene_id]`.
  - `process_scene(config_path, split, scenes_dir, gt_json, habitat_data_path, scene_id, scene_groups, gt_subset, source_subset, out_dir, subgoal_distance, subgoal_radius, no_video, video_fps, video_highlight_frames, no_topdown, topdown_max_size, overwrite, continue_on_error, train_json_path) -> dict` — worker entry; returns `{"scene_id", "processed", "skipped", "failed": [...]}`.

- [ ] **Step 1: Add the four functions (no test yet — habitat-bound; verified by import + review)**

Insert into `subgoal_pipeline/build_dataset_geodesic.py` after the Task 1 helpers, before `build_dataset_geodesic`. Add `import os` to the module's top-level imports (it is needed by `init_worker`; currently not imported).

```python
def process_group(
    env,
    env_episode,
    source_episode: dict,
    gt_trajectory,
    trajectory_id: int,
    episode_ids: list,
    representative_id: int,
    out_root: Path,
    habitat_data_path: str,
    train_json_path: str,
    subgoal_distance: float,
    subgoal_radius: float,
    max_steps: int,
    no_video: bool,
    video_fps: int,
    video_highlight_frames: int,
    no_topdown: bool,
    topdown_max_size: int,
    overwrite: bool,
    continue_on_error: bool,
    group_idx: int,
    total_groups: int,
) -> dict:
    instruction = instruction_text(source_episode).strip()
    artifact_name = f"traj{int(trajectory_id):06d}_rep_episode_id{int(representative_id):06d}"
    artifact_dir = out_root / artifact_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if (artifact_dir / "subgoals_geodesic.json").exists() and not overwrite:
        print(f"[skip] {artifact_name} existing subgoals_geodesic.json")
        return {"trajectory_id": int(trajectory_id), "status": "skipped", "failure": None}

    episode_label = (
        f"{artifact_name} scene={source_episode.get('scene_id')} "
        f"traj={trajectory_id} group_size={len(episode_ids)}"
    )
    print(
        f"\n[{group_idx + 1}/{total_groups}] Processing {episode_label} "
        f"actions={len(gt_trajectory.actions)}"
    )

    try:
        steps, rollout_info = replay_gt_actions_in_memory(
            env=env,
            episode=env_episode,
            gt_trajectory=gt_trajectory,
            max_steps=max_steps,
        )
        if not steps:
            raise RuntimeError("GT replay produced no frames.")

        final_goal_position, _goals_payload = extract_goal_payload(env_episode)
        if final_goal_position is None:
            raise RuntimeError(f"episode {representative_id} has no goal position.")

        selected = select_geodesic_subgoal_steps(
            steps=steps,
            sim=env.sim,
            subgoal_distance=subgoal_distance,
            final_goal_position=list(final_goal_position),
            episode_label=episode_label,
        )
        raw_subgoals = _build_geodesic_subgoal_dicts(steps, selected)
        enriched = enrich_subgoals_from_memory(subgoals=raw_subgoals, steps=steps)
        final = enriched[-1]
        final["subgoal_position"] = list(final_goal_position)
        final["subgoal_position_source"] = "original_train.goals[0].position"
        final["agent_position_at_keyframe"] = list(final_goal_position)
        final["is_final_goal"] = True

        meta = build_episode_meta(
            episode=env_episode,
            source_episode=source_episode,
            gt_trajectory=gt_trajectory,
            steps=steps,
            rollout_info=rollout_info,
            habitat_data_path=habitat_data_path,
            input_data_path=str(train_json_path),
        )
        payload = {
            "instruction": instruction,
            "scene_id": meta.get("scene_id"),
            "episode_id": int(representative_id),
            "trajectory_id": int(trajectory_id),
            "trajectory_episode_ids": [int(v) for v in episode_ids],
            "subgoal_selection_method": "geodesic_distance",
            "subgoal_distance": subgoal_distance,
            "sampled_frame_steps": [int(item["best_step"]) for item in enriched],
            "goal_position": list(final_goal_position),
            "goal_position_source": "original_train.goals[0].position",
            "subgoals": enriched,
            "postprocess_notes": [],
        }
        validate_subgoal_payload(payload)

        write_json(artifact_dir / "subgoals_geodesic.json", payload, pretty=True)
        write_source_episode_artifact(
            out_dir=artifact_dir,
            source_episode=source_episode,
            meta=meta,
            selected_episode_ids=episode_ids,
            representative_id=representative_id,
        )

        topdown_metric = None
        if not no_video or not no_topdown:
            metrics = env.get_metrics()
            td_key = (
                "top_down_map_vlnce"
                if "top_down_map_vlnce" in metrics
                else "top_down_map"
            )
            topdown_metric = metrics.get(td_key)
            if topdown_metric is None:
                print(
                    f"  !! top_down_map metric unavailable; "
                    f"skipping topdown for {artifact_name}"
                )

        if not no_video:
            write_subgoal_video_from_memory(
                output_path=artifact_dir / "subgoals_geodesic.mp4",
                steps=steps,
                subgoals=enriched,
                fps=int(video_fps),
                highlight_frames=int(video_highlight_frames),
                topdown_metric=topdown_metric,
                sim=env.sim,
                instruction=instruction,
                subgoal_radius=float(subgoal_radius),
            )

        if topdown_metric is not None:
            write_topdown_image_from_memory(
                output_path=artifact_dir / "topdown_subgoals.png",
                topdown_metric=topdown_metric,
                subgoals=enriched,
                sim=env.sim,
                radius=float(subgoal_radius),
                max_size=int(topdown_max_size),
            )

        print(
            f"  -> saved {artifact_dir}; subgoals={len(enriched)} "
            f"distance={subgoal_distance}"
        )
        return {"trajectory_id": int(trajectory_id), "status": "ok", "failure": None}
    except Exception as exc:
        failure = {
            "trajectory_id": int(trajectory_id),
            "episode_ids": [int(v) for v in episode_ids],
            "representative_episode_id": int(representative_id),
            "error": repr(exc),
        }
        if not continue_on_error:
            raise
        print(f"  !! failed {artifact_name}: {exc!r}")
        return {"trajectory_id": int(trajectory_id), "status": "failed", "failure": failure}


def init_worker(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def _build_scene_env_config(
    config_path: str,
    split: str,
    scenes_dir: str,
    gt_json: str,
    habitat_data_path: str,
    scene_id: str,
):
    import habitat  # noqa: F401  (lazy: ensures CUDA_VISIBLE_DEVICES is set first)
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config
    from rlinf.envs.habitat.extensions import measures as _rlinf_measures  # noqa: F401

    cfg = get_habitat_config(str(config_path))
    with read_write(cfg):
        cfg.habitat.dataset.split = str(split)
        cfg.habitat.dataset.data_path = habitat_data_path
        cfg.habitat.dataset.scenes_dir = str(scenes_dir)
        cfg.habitat.dataset.content_scenes = [str(scene_id)]
        ndtw_measure = cfg.habitat.task.measurements.ndtw
        ndtw_measure.SPLIT = str(split)
        ndtw_measure.GT_PATH = str(Path(gt_json).resolve())
        cfg.habitat.task.measurements.top_down_map.draw_shortest_path = False
    return cfg


def process_scene(
    config_path: str,
    split: str,
    scenes_dir: str,
    gt_json: str,
    habitat_data_path: str,
    scene_id: str,
    scene_groups: list,
    gt_subset: dict,
    source_subset: dict,
    out_dir: str,
    subgoal_distance: float,
    subgoal_radius: float,
    no_video: bool,
    video_fps: int,
    video_highlight_frames: int,
    no_topdown: bool,
    topdown_max_size: int,
    overwrite: bool,
    continue_on_error: bool,
    train_json_path: str,
) -> dict:
    import habitat  # lazy

    out_root = Path(out_dir)
    cfg = _build_scene_env_config(
        config_path, split, scenes_dir, gt_json, habitat_data_path, scene_id
    )
    env = habitat.Env(config=cfg)
    env_episode_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    max_steps = int(cfg.habitat.environment.max_episode_steps)

    processed = 0
    skipped = 0
    failed: list = []
    total = len(scene_groups)
    try:
        for offset, (trajectory_id, episode_ids, representative_id) in enumerate(scene_groups):
            rep = int(representative_id)
            env_episode = env_episode_by_id.get(rep)
            if env_episode is None:
                failure = {
                    "trajectory_id": int(trajectory_id),
                    "episode_ids": [int(v) for v in episode_ids],
                    "representative_episode_id": rep,
                    "error": f"representative episode {rep} missing from scene {scene_id} env",
                }
                if not continue_on_error:
                    raise RuntimeError(failure["error"])
                failed.append(failure)
                continue
            result = process_group(
                env=env,
                env_episode=env_episode,
                source_episode=source_subset[rep],
                gt_trajectory=gt_subset[rep],
                trajectory_id=int(trajectory_id),
                episode_ids=[int(v) for v in episode_ids],
                representative_id=rep,
                out_root=out_root,
                habitat_data_path=habitat_data_path,
                train_json_path=train_json_path,
                subgoal_distance=subgoal_distance,
                subgoal_radius=subgoal_radius,
                max_steps=max_steps,
                no_video=no_video,
                video_fps=video_fps,
                video_highlight_frames=video_highlight_frames,
                no_topdown=no_topdown,
                topdown_max_size=topdown_max_size,
                overwrite=overwrite,
                continue_on_error=continue_on_error,
                group_idx=offset,
                total_groups=total,
            )
            status = result["status"]
            if status == "ok":
                processed += 1
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed.append(result["failure"])
    finally:
        env.close()
    return {
        "scene_id": scene_id,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
    }
```

- [ ] **Step 2: Add `import os` to the module top-level imports**

The current import block starts with `import argparse`, `import json`, `import math`, `import time`. Add `import os` in alphabetical order (between `import math` and `import time`).

- [ ] **Step 3: Verify import-without-habitat + existing tests still pass**

Run: `cd /data/RLinf && python -c "from subgoal_pipeline import build_dataset_geodesic; print('ok')"`
Expected: prints `ok` (habitat not imported at module level).

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py -v`
Expected: 19 passed (9 selector + 10 multiprocess helpers). The new functions are not yet called by the main loop — that is intentional; they wire in Task 3.

- [ ] **Step 4: Lint**

Run: `cd /data/RLinf && /opt/venv/habitat/bin/ruff check subgoal_pipeline/build_dataset_geodesic.py`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/build_dataset_geodesic.py
git commit -s -m "feat: add geodesic process_group/process_scene worker functions"
```

---

### Task 3: Wire multiprocess dispatch into build_dataset_geodesic + new CLI args

**Files:**
- Modify: `subgoal_pipeline/build_dataset_geodesic.py` — replace the body of `build_dataset_geodesic` from the `import habitat` line (currently line ~221, after the dry-run `return`) through the end of the function (the `output_json` write), keeping lines 107–219 (load/select/dry-run) intact; add `--num_processes` and `--gpus` to `build_arg_parser`.
- Test: `tests/unit_tests/test_geodesic_subgoal_multiprocess.py` (append arg-parser tests)

**Interfaces:**
- Consumes: Task 1 helpers (`parse_gpus`, `validate_num_processes_vs_gpus`, `bucket_groups_by_scene`, `assign_scenes_to_gpus`, `compute_workers_per_gpu`, `aggregate_summaries`); Task 2 worker (`process_scene`, `init_worker`); `prepare_habitat_data_path`, `append_jsonl`, `load_json`, `write_json`, `subgoals_to_goals` (existing).
- Produces: `--num_processes` (int, default 1) and `--gpus` (str, default `"0"`) CLI args; refactored `build_dataset_geodesic` that dispatches per-GPU `ProcessPoolExecutor` pools.

- [ ] **Step 1: Append failing arg-parser tests**

Append to `tests/unit_tests/test_geodesic_subgoal_multiprocess.py`:

```python
def test_multiprocess_arg_defaults():
    parser = bdg.build_arg_parser()
    args = parser.parse_args(["--subgoal_distance", "3.0"])
    assert args.num_processes == 1
    assert args.gpus == "0"


def test_multiprocess_args_parsed():
    parser = bdg.build_arg_parser()
    args = parser.parse_args(
        ["--subgoal_distance", "3.0", "--num_processes", "8", "--gpus", "0,1,2,3"]
    )
    assert args.num_processes == 8
    assert args.gpus == "0,1,2,3"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_multiprocess.py -v`
Expected: 2 new tests FAIL — `AttributeError: 'Namespace' object has no attribute 'num_processes'`.

- [ ] **Step 3: Add the two CLI args to `build_arg_parser`**

In `build_arg_parser`, after the existing `--subgoal_distance` argument block, add:

```python
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Total worker process count for parallel rendering (default 1 = single process).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU ids, e.g. '0,1,2,3'. Scenes are round-robin distributed by count.",
    )
```

- [ ] **Step 4: Replace the habitat/dispatch/output section of `build_dataset_geodesic`**

Keep `build_dataset_geodesic` lines 107–219 (from `out_root = Path(args.out_dir)` through the `dry_run_selection` `return`) UNCHANGED **except** remove the now-dead `representative_ids = [int(representative_id) for _tid, _ids, representative_id in groups]` line (its only consumer, the `missing_env` check, is removed below — that check now lives inside `process_scene` as the `env_episode is None` branch). Replace everything from the `import habitat` line (currently line ~221) through the end of the function with the dispatch + aggregate + glob-scan + output code below. The new section:

```python
    habitat_data_path = prepare_habitat_data_path(
        data_path=str(train_json_path),
        split=str(args.split),
        out_dir=str(out_root),
    )
    gpu_ids = parse_gpus(str(args.gpus))
    num_processes = int(args.num_processes)
    validate_num_processes_vs_gpus(num_processes, len(gpu_ids))

    if remaining_max_traj >= 0:
        groups = groups[:remaining_max_traj]

    scene_to_groups = bucket_groups_by_scene(groups, source_episode_by_id)
    scene_list = sorted(scene_to_groups.keys())
    gpu_to_scenes = assign_scenes_to_gpus(scene_list, gpu_ids)
    scenes_per_gpu = {int(g): len(scenes) for g, scenes in gpu_to_scenes.items()}
    workers_per_gpu = compute_workers_per_gpu(num_processes, gpu_ids, scenes_per_gpu)

    print(
        f"Multiprocess: num_processes={num_processes} gpus={gpu_ids} "
        f"scenes={len(scene_list)} workers_per_gpu={workers_per_gpu}"
    )

    base_kwargs = dict(
        config_path=str(args.config),
        split=str(args.split),
        scenes_dir=str(args.scenes_dir),
        gt_json=str(args.gt_json),
        habitat_data_path=habitat_data_path,
        out_dir=str(out_root),
        subgoal_distance=float(args.subgoal_distance),
        subgoal_radius=float(args.subgoal_radius),
        no_video=bool(args.no_video),
        video_fps=int(args.video_fps),
        video_highlight_frames=int(args.video_highlight_frames),
        no_topdown=bool(args.no_topdown),
        topdown_max_size=int(args.topdown_max_size),
        overwrite=bool(args.overwrite),
        continue_on_error=bool(args.continue_on_error),
        train_json_path=str(train_json_path),
    )

    summaries = _dispatch_workers(
        gpu_ids, gpu_to_scenes, workers_per_gpu, scene_to_groups,
        gt_trajectories, source_episode_by_id, base_kwargs,
    )
    agg = aggregate_summaries(summaries)
    for failure in agg["failed"]:
        append_jsonl(out_root / "failures.jsonl", failure)

    subgoals_by_trajectory: dict[int, dict] = {}
    for payload_path in sorted(out_root.glob("traj*/subgoals_geodesic.json")):
        try:
            payload = load_json(payload_path)
        except Exception:
            continue
        trajectory_id = int(payload.get("trajectory_id", -1))
        if trajectory_id >= 0:
            subgoals_by_trajectory[trajectory_id] = payload

    output_episodes: list[dict] = []
    modified = 0
    skipped_selected_without_payload = 0
    for episode in original_episodes:
        episode_id = int(episode["episode_id"])
        trajectory_id = int(episode.get("trajectory_id", episode_id))
        payload = subgoals_by_trajectory.get(trajectory_id)
        if payload is None:
            if episode_id in selected_episode_ids:
                skipped_selected_without_payload += 1
            if args.include_unselected:
                output_episodes.append(episode)
            continue
        new_episode = deepcopy(episode)
        new_episode["goals"] = subgoals_to_goals(
            subgoals=list(payload["subgoals"]),
            original_episode=episode,
            radius=float(args.subgoal_radius),
        )
        output_episodes.append(new_episode)
        modified += 1

    output_data = deepcopy(train_data)
    output_data["episodes"] = output_episodes
    output_data["_subgoal_generation"] = {
        "script": "subgoal_pipeline.build_dataset_geodesic",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_json": str(train_json_path),
        "gt_json": str(args.gt_json),
        "out_dir": str(out_root),
        "dataset_type": dataset_type,
        "target_episodes": int(args.target_episodes),
        "include_unselected": bool(args.include_unselected),
        "selected_episodes": len(selected_episode_ids) + int(done_episode_count),
        "modified_episodes": int(modified),
        "representative_episodes": len(subgoals_by_trajectory),
        "max_gt_actions": int(args.max_gt_actions),
        "min_gt_actions": int(args.min_gt_actions),
        "subgoal_radius": float(args.subgoal_radius),
        "subgoal_distance": float(args.subgoal_distance),
        "subgoal_selection_method": "geodesic_distance",
        "parallel_mode": True,
        "num_processes": num_processes,
        "gpus": gpu_ids,
        "workers_per_gpu": workers_per_gpu,
        "scenes_total": len(scene_list),
        "scenes_per_gpu": scenes_per_gpu,
        "scenes_processed": int(agg["scenes_processed"]),
        "scenes_failed": int(agg["scenes_failed"]),
        "resumed_trajectories": len(done_trajectory_ids),
        "resumed_episodes": int(done_episode_count),
        "remaining_target_episodes": int(remaining_target),
        "remaining_max_trajectories": int(remaining_max_traj),
        "failures": agg["failed"],
        **group_stats,
    }
    write_json(output_json, output_data, pretty=bool(args.pretty_output))
    print(
        "\nDone: "
        f"wrote={output_json} episodes={len(output_episodes)} "
        f"modified={modified} selected_without_payload={skipped_selected_without_payload} "
        f"scenes_processed={agg['scenes_processed']} scenes_failed={agg['scenes_failed']} "
        f"failures={len(agg['failed'])}"
    )
```

Then add the `_dispatch_workers` helper as a module-level function (place it just before `build_dataset_geodesic`):

```python
def _dispatch_workers(
    gpu_ids: list[int],
    gpu_to_scenes: dict[int, list[str]],
    workers_per_gpu: dict[int, int],
    scene_to_groups: dict[str, list],
    gt_trajectories: dict,
    source_episode_by_id: dict,
    base_kwargs: dict,
) -> list[dict]:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    spawn_ctx = mp.get_context("spawn")
    summaries: list[dict] = []
    pools: list[ProcessPoolExecutor] = []
    futures: list = []
    for gpu_id in gpu_ids:
        scenes = gpu_to_scenes.get(int(gpu_id), [])
        if not scenes:
            continue
        max_workers = int(workers_per_gpu[int(gpu_id)])
        pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=spawn_ctx,
            initializer=init_worker,
            initargs=(int(gpu_id),),
        )
        pools.append(pool)
        for scene_id in scenes:
            scene_groups = scene_to_groups[scene_id]
            rep_ids = [
                int(rid) for _tid, _eids, rid in scene_groups
                if int(rid) in gt_trajectories and int(rid) in source_episode_by_id
            ]
            gt_subset = {rid: gt_trajectories[rid] for rid in rep_ids}
            source_subset = {rid: source_episode_by_id[rid] for rid in rep_ids}
            kw = dict(base_kwargs)
            kw.update(
                scene_id=scene_id,
                scene_groups=[
                    (int(t), [int(v) for v in eids], int(rid))
                    for t, eids, rid in scene_groups
                ],
                gt_subset=gt_subset,
                source_subset=source_subset,
            )
            futures.append(pool.submit(process_scene, **kw))
    for fut in futures:
        try:
            summaries.append(fut.result())
        except Exception as exc:
            summaries.append(
                {"scene_id": "?", "processed": 0, "skipped": 0, "failed": [{"error": repr(exc)}]}
            )
    for pool in pools:
        pool.shutdown(wait=True)
    return summaries
```

Note: the old `import habitat` / `env = habitat.Env(...)` / `try: for ... finally: env.close()` block (the single-process loop) is REMOVED — it is replaced by `_dispatch_workers`. The single-process case (`--num_processes 1 --gpus "0"`) now runs through one pool with one worker calling `process_scene`.

- [ ] **Step 5: Run all tests to verify they pass**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py -v`
Expected: 21 passed (9 selector + 10 helpers + 2 arg-parser).

- [ ] **Step 6: Verify import-without-habitat + ruff**

Run: `cd /data/RLinf && python -c "from subgoal_pipeline import build_dataset_geodesic; print('ok')"`
Expected: prints `ok`.

Run: `cd /data/RLinf && /opt/venv/habitat/bin/ruff check subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_multiprocess.py
git commit -s -m "feat: wire multiprocess GPU dispatch into geodesic sub-goal builder"
```

---

### Task 4: Document the multiprocess mode in the README

**Files:**
- Modify: `subgoal_pipeline/README.md` (the "Geodesic variant" section added earlier)

**Interfaces:**
- Consumes: the multiprocess CLI from Task 3.

- [ ] **Step 1: Append a multiprocess command example**

In `subgoal_pipeline/README.md`, within the "Geodesic variant (no GPT)" section, after the existing single-process bash block, add a short paragraph + command:

```markdown

For multi-process parallel rendering across GPUs, add `--num_processes` and `--gpus`:
scenes are round-robin distributed by count across the listed GPUs, each worker pinned
to one GPU via `CUDA_VISIBLE_DEVICES` and loading one scene at a time
(`content_scenes=[scene]`, no cross-scene reconfigure). `--num_processes` must be >= the
GPU count.

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
```

- [ ] **Step 2: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/README.md
git commit -s -m "docs: document geodesic multiprocess mode"
```

---

## Self-Review notes

- Spec §1 (architecture: main no-habitat, per-GPU pools, worker env-per-scene, glob-scan aggregate) → Task 3 (`_dispatch_workers` + glob-scan) + Task 2 (`process_scene` env-per-scene).
- Spec §2 (worker interface, primitives + subset, lazy habitat, reuse helpers) → Task 2 `process_scene`/`process_group`/`_build_scene_env_config`; `train_json_path`/`overwrite`/`continue_on_error` added as plumbing primitives (spec signature listed the main params; these are implied picklable primitives).
- Spec §3 (GPU allocation, N>=G, workers_per_gpu remainder, clamp, init_worker CUDA_VISIBLE_DEVICES, spawn, scene-task) → Task 1 `compute_workers_per_gpu`/`validate`/`assign_scenes_to_gpus` + Task 2 `init_worker` + Task 3 `_dispatch_workers` (spawn ctx, per-GPU pool, initargs).
- Spec §4 (aggregate summaries, glob-scan rebuild, main writes failures.jsonl once + output_json once, new meta fields) → Task 1 `aggregate_summaries` + Task 3 main body + meta dict.
- Spec §5 (tests: assign/compute_workers/bucket/aggregate/validate/arg-parser) → Task 1 tests (all six) + Task 3 arg-parser tests. No habitat e2e (by design).
- Spec §CLI (`--num_processes`, `--gpus`) → Task 3 `build_arg_parser`.
- Spec "Surgical changes: README append" → Task 4.
- Deviation: `max_steps` is read by the worker from its built cfg (spec §2 note) — Task 2 `process_scene` reads it and passes to `process_group`; not in `process_scene`'s signature. Consistent.
- Verified `python -m` + spawn + module-level worker + initializer works (empirical test in plan preamble), so worker functions stay in `build_dataset_geodesic.py` (no separate module needed).
