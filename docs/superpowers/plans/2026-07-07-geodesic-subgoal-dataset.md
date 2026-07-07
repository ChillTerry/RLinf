# Geodesic-Distance Sub-goal Dataset Builder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPT-free `subgoal_pipeline.build_dataset_geodesic` module that selects sub-goals at fixed geodesic-distance intervals along the replayed GT trajectory and writes a sub-goal dataset whose final goal equals the original episode goal.

**Architecture:** New standalone script reusing `gt`/`replay`/`artifacts`/`common`/`video`. A pure, habitat-free `select_geodesic_subgoal_steps(steps, sim, subgoal_distance, final_goal_position)` walks the replay steps, greedily picks a sub-goal every `D` meters of `sim.geodesic_distance` from the previous pick, then iteratively drops trailing picks within `D` of the final goal. The main loop mirrors `build_dataset_online` but replaces the OpenAI call + sanitize with this selector.

**Tech Stack:** Python 3, Habitat sim (`sim.geodesic_distance`), numpy, cv2/ffmpeg (video), pytest, ruff (`/opt/venv/habitat/bin/ruff`).

## Global Constraints

- New file `subgoal_pipeline/build_dataset_geodesic.py` + new test `tests/unit_tests/test_geodesic_subgoal_selection.py`. No edits to `build_dataset.py` or any existing module.
- Module-level imports must stay habitat-free (habitat imported lazily inside `build_dataset_geodesic`, exactly like `build_dataset.py:165-168`), so the unit test can import the module without habitat.
- Artifact filename: `subgoals_geodesic.json` (not `subgoals_openai.json`). Resume glob: `traj*/subgoals_geodesic.json`.
- `--subgoal_distance` is a required CLI arg (no default).
- Geodesic fallback: when `sim.geodesic_distance` returns non-finite, fall back to `math.dist` (euclidean) AND `print` a record. (Spec said "measures.py已有 euclidean_distance"; using stdlib `math.dist` instead avoids a habitat-dep import in the pure selector — same euclidean result.)
- Final sub-goal position is forced to the original episode goal position (from `extract_goal_payload`); `is_final_goal=True`; `best_step = steps[-1].step` (STOP frame).
- Ruff lint must pass: `ruff check subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_selection.py`.
- Conventional Commits, `git commit -s`.

---

### Task 1: Geodesic sub-goal selector (pure function, TDD)

**Files:**
- Create: `subgoal_pipeline/build_dataset_geodesic.py` (only the selector + `_geodesic` + imports for now; the main loop is added in Task 2)
- Test: `tests/unit_tests/test_geodesic_subgoal_selection.py`

**Interfaces:**
- Produces: `select_geodesic_subgoal_steps(steps: List[MemoryStep], sim, subgoal_distance: float, final_goal_position: List[float], *, episode_label: str = "") -> List[int]` — returns indices into `steps` for the **intermediate** sub-goals (never includes the final/stop frame). Also produces helper `_geodesic(sim, a, b, *, episode_label="", step_idx=None) -> float`.
- Consumes: `subgoal_pipeline.replay.MemoryStep` (dataclass: `step:int, rgb:np.ndarray, agent_position:List[float], agent_rotation:List[float]`).

- [ ] **Step 1: Write the failing test file**

Create `tests/unit_tests/test_geodesic_subgoal_selection.py`:

```python
import math

import numpy as np
import pytest

from subgoal_pipeline import build_dataset_geodesic
from subgoal_pipeline.replay import MemoryStep


class FakeSim:
    """Stand-in for habitat sim; geodesic_distance is scriptable."""

    def __init__(self, geodesic_fn=None):
        self._fn = geodesic_fn or (lambda a, b: math.dist(list(a), list(b)))

    def geodesic_distance(self, a, b):
        return self._fn(a, b)


def _step(i, pos):
    return MemoryStep(
        step=int(i),
        rgb=np.zeros(1, dtype=np.uint8),
        agent_position=[float(v) for v in pos],
        agent_rotation=[0.0, 0.0, 0.0, 1.0],
    )


def test_evenly_spaced_with_single_tail_removal():
    # 11 steps on a line at x=0..10; D=3; final goal at x=10 (stop frame).
    steps = [_step(i, [float(i), 0.0, 0.0]) for i in range(11)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[10.0, 0.0, 0.0]
    )
    # Greedy picks 3, 6, 9; tail 9->10 == 1 < 3 dropped; 6->10 == 4 kept.
    assert selected == [3, 6]


def test_iterative_double_tail_removal():
    # start(-3,0) -> A(0,0) -> B(3,0) -> final(1.5, 2).
    # Greedy picks A (index 3) and B (index 6).
    # B->final == 2.5 < 3 dropped; A->final == 2.5 < 3 dropped.
    positions = [
        [-3.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [1.5, 2.0, 0.0],
    ]
    steps = [_step(i, p) for i, p in enumerate(positions)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[1.5, 2.0, 0.0]
    )
    assert selected == []


def test_path_shorter_than_distance_yields_no_intermediates():
    steps = [_step(0, [0.0, 0.0, 0.0]), _step(1, [0.5, 0.0, 0.0]), _step(2, [1.0, 0.0, 0.0])]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[1.0, 0.0, 0.0]
    )
    assert selected == []


def test_too_few_steps_returns_empty():
    steps = [_step(0, [0.0, 0.0, 0.0])]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(), subgoal_distance=3.0, final_goal_position=[0.0, 0.0, 0.0]
    )
    assert selected == []


def test_inf_geodesic_falls_back_to_euclidean(capsys):
    # Force the (start -> step at x=3) call to return inf; euclidean=3 should still select it.
    def fn(a, b):
        a = list(a)
        b = list(b)
        if math.dist(a, b) >= 3.0 and abs(a[0] - b[0]) >= 3.0 and a[1] == 0.0 and b[1] == 0.0:
            return float("inf")
        return math.dist(a, b)

    steps = [_step(i, [float(i), 0.0, 0.0]) for i in range(5)]
    selected = build_dataset_geodesic.select_geodesic_subgoal_steps(
        steps, FakeSim(fn), subgoal_distance=3.0, final_goal_position=[4.0, 0.0, 0.0]
    )
    # Greedy picks 3 (via euclidean fallback); tail 3->4 == 1 < 3 dropped -> [].
    assert selected == []
    out = capsys.readouterr().out
    assert "geodesic_distance not finite" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'subgoal_pipeline.build_dataset_geodesic'`.

- [ ] **Step 3: Write the selector implementation**

Create `subgoal_pipeline/build_dataset_geodesic.py` with ONLY the imports + `_geodesic` + `select_geodesic_subgoal_steps` for now (the main loop and arg parser come in Task 2). Header:

```python
import math
from typing import List

from .replay import MemoryStep


def _geodesic(sim, a, b, *, episode_label: str = "", step_idx=None) -> float:
    raw = sim.geodesic_distance(a, b)
    try:
        d = float(raw)
    except (TypeError, ValueError):
        d = math.inf
    if math.isfinite(d) and d >= 0:
        return d
    fallback = math.dist(list(a), list(b))
    print(
        f"  !! geodesic_distance not finite (raw={raw!r}); "
        f"falling back to euclidean={fallback:.3f} "
        f"ep={episode_label} step={step_idx} a={list(a)} b={list(b)}"
    )
    return fallback


def select_geodesic_subgoal_steps(
    steps: List[MemoryStep],
    sim,
    subgoal_distance: float,
    final_goal_position: List[float],
    *,
    episode_label: str = "",
) -> List[int]:
    selected: List[int] = []
    if len(steps) < 2:
        return selected
    anchor = steps[0].agent_position
    for i in range(1, len(steps)):
        d = _geodesic(
            sim, anchor, steps[i].agent_position,
            episode_label=episode_label, step_idx=i,
        )
        if d >= float(subgoal_distance):
            selected.append(i)
            anchor = steps[i].agent_position
    while selected:
        last_step = steps[selected[-1]]
        d = _geodesic(
            sim, last_step.agent_position, final_goal_position,
            episode_label=episode_label, step_idx=last_step.step,
        )
        if d < float(subgoal_distance):
            selected.pop()
        else:
            break
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py -v`
Expected: 5 passed.

- [ ] **Step 5: Lint**

Run: `cd /data/RLinf && /opt/venv/habitat/bin/ruff check subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_selection.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_selection.py
git commit -s -m "feat: add geodesic-distance sub-goal selector"
```

---

### Task 2: Full builder script + arg-parser tests

**Files:**
- Modify: `subgoal_pipeline/build_dataset_geodesic.py` (add imports, `_build_geodesic_subgoal_dicts`, `build_dataset_geodesic`, `build_arg_parser`, `main`)
- Test: `tests/unit_tests/test_geodesic_subgoal_selection.py` (append arg-parser tests)

**Interfaces:**
- Consumes: `select_geodesic_subgoal_steps` (Task 1); `replay_gt_actions_in_memory`, `MemoryStep` (replay.py); `load_ground_truth_trajectories`, `build_dataset_indices`, `select_trajectory_groups` (gt.py); `build_episode_meta`, `enrich_subgoals_from_memory`, `subgoals_to_goals`, `validate_subgoal_payload`, `write_source_episode_artifact` (artifacts.py); `append_jsonl`, `detect_dataset_type`, `extract_goal_payload`, `instruction_text`, `is_english_instruction`, `load_json`, `prepare_habitat_data_path`, `write_json` (common.py); `write_subgoal_video_from_memory`, `write_topdown_image_from_memory` (video.py).
- Produces: CLI entry `python3 -m subgoal_pipeline.build_dataset_geodesic`; `build_arg_parser()`; `build_dataset_geodesic(args)`.

- [ ] **Step 1: Append failing arg-parser tests**

Append to `tests/unit_tests/test_geodesic_subgoal_selection.py`:

```python
def test_subgoal_distance_is_required():
    parser = build_dataset_geodesic.build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_gpt_only_args_are_absent():
    parser = build_dataset_geodesic.build_arg_parser()
    dests = {action.dest for action in parser._actions}
    for gpt_arg in [
        "model", "reasoning_effort", "max_output_tokens", "max_frames",
        "frame_stride", "min_step_gap", "image_format", "jpeg_quality",
        "base_url", "api_key", "user_agent", "usage_log_name",
    ]:
        assert gpt_arg not in dests, f"GPT-only arg --{gpt_arg} should not exist"


def test_subgoal_distance_parsed():
    parser = build_dataset_geodesic.build_arg_parser()
    args = parser.parse_args(["--subgoal_distance", "3.5"])
    assert args.subgoal_distance == 3.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py -v`
Expected: 3 new tests FAIL — `AttributeError: module has no attribute 'build_arg_parser'`.

- [ ] **Step 3: Write the full script body**

Replace the import block at the top of `subgoal_pipeline/build_dataset_geodesic.py` and append the rest. The final file content (Task 1 selector + this task's additions) is:

```python
import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from .artifacts import (
    build_episode_meta,
    enrich_subgoals_from_memory,
    subgoals_to_goals,
    validate_subgoal_payload,
    write_source_episode_artifact,
)
from .common import (
    append_jsonl,
    detect_dataset_type,
    extract_goal_payload,
    instruction_text,
    is_english_instruction,
    load_json,
    prepare_habitat_data_path,
    write_json,
)
from .gt import build_dataset_indices, load_ground_truth_trajectories, select_trajectory_groups
from .replay import MemoryStep, replay_gt_actions_in_memory
from .video import write_subgoal_video_from_memory, write_topdown_image_from_memory


def _geodesic(sim, a, b, *, episode_label: str = "", step_idx=None) -> float:
    raw = sim.geodesic_distance(a, b)
    try:
        d = float(raw)
    except (TypeError, ValueError):
        d = math.inf
    if math.isfinite(d) and d >= 0:
        return d
    fallback = math.dist(list(a), list(b))
    print(
        f"  !! geodesic_distance not finite (raw={raw!r}); "
        f"falling back to euclidean={fallback:.3f} "
        f"ep={episode_label} step={step_idx} a={list(a)} b={list(b)}"
    )
    return fallback


def select_geodesic_subgoal_steps(
    steps: List[MemoryStep],
    sim,
    subgoal_distance: float,
    final_goal_position: List[float],
    *,
    episode_label: str = "",
) -> List[int]:
    selected: List[int] = []
    if len(steps) < 2:
        return selected
    anchor = steps[0].agent_position
    for i in range(1, len(steps)):
        d = _geodesic(
            sim, anchor, steps[i].agent_position,
            episode_label=episode_label, step_idx=i,
        )
        if d >= float(subgoal_distance):
            selected.append(i)
            anchor = steps[i].agent_position
    while selected:
        last_step = steps[selected[-1]]
        d = _geodesic(
            sim, last_step.agent_position, final_goal_position,
            episode_label=episode_label, step_idx=last_step.step,
        )
        if d < float(subgoal_distance):
            selected.pop()
        else:
            break
    return selected


def _build_geodesic_subgoal_dicts(steps: List[MemoryStep], selected_indices: List[int]) -> List[dict]:
    subgoals: List[dict] = []
    for idx, i in enumerate(selected_indices):
        subgoals.append({
            "subgoal_id": idx,
            "landmark": "",
            "landmark_source": "geodesic_distance_milestone",
            "is_final_goal": False,
            "best_step": int(steps[i].step),
            "reason": "auto-selected at geodesic interval",
        })
    subgoals.append({
        "subgoal_id": len(selected_indices),
        "landmark": "",
        "landmark_source": "geodesic_distance_milestone",
        "is_final_goal": True,
        "best_step": int(steps[-1].step),
        "reason": "final goal (stop frame)",
    })
    return subgoals


def build_dataset_geodesic(args: argparse.Namespace) -> None:
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json)
    if output_json.exists() and not args.overwrite and not args.dry_run_selection:
        print(f"Output exists; supplementing in place: {output_json}")

    train_json_path = Path(args.train_json)
    if train_json_path.suffix != ".gz":
        raise RuntimeError(f"{train_json_path} must be a gzipped train JSON file ending with .gz.")

    train_data = load_json(train_json_path)
    dataset_type = detect_dataset_type(train_data)
    print(f"Detected dataset type: {dataset_type}")
    if dataset_type == "rxr":
        episodes_before = len(train_data.get("episodes") or [])
        train_data["episodes"] = [
            ep for ep in train_data.get("episodes", []) if is_english_instruction(ep)
        ]
        print(f"rxr English-only filter: {episodes_before} -> {len(train_data['episodes'])} episodes")
    original_episodes = train_data.get("episodes")
    if not isinstance(original_episodes, list):
        raise RuntimeError(f"{args.train_json} does not contain episodes.")

    source_episode_by_id, by_trajectory = build_dataset_indices(train_data)
    gt_trajectories = load_ground_truth_trajectories(Path(args.gt_json))

    subgoals_by_trajectory: Dict[int, dict] = {}
    done_trajectory_ids: set = set()
    done_episode_count = 0
    if not args.overwrite:
        for payload_path in sorted(out_root.glob("traj*/subgoals_geodesic.json")):
            try:
                payload = load_json(payload_path)
            except Exception:
                continue
            trajectory_id = int(payload.get("trajectory_id", -1))
            if trajectory_id >= 0:
                subgoals_by_trajectory[trajectory_id] = payload
        done_trajectory_ids = set(subgoals_by_trajectory.keys())
        done_episode_count = sum(
            len(by_trajectory[int(t)]) for t in done_trajectory_ids if int(t) in by_trajectory
        )
        if done_trajectory_ids:
            print(
                f"Resume: {len(done_trajectory_ids)} trajectories / "
                f"{done_episode_count} episodes already processed"
            )

    if int(args.target_episodes) > 0 and not args.overwrite:
        remaining_target = max(0, int(args.target_episodes) - int(done_episode_count))
    else:
        remaining_target = int(args.target_episodes)
    if int(args.max_trajectories) >= 0 and not args.overwrite:
        remaining_max_traj = max(0, int(args.max_trajectories) - len(done_trajectory_ids))
    else:
        remaining_max_traj = int(args.max_trajectories)
    print(
        f"Budget: target_episodes {args.target_episodes} -> {remaining_target} remaining; "
        f"max_trajectories {args.max_trajectories} -> {remaining_max_traj} remaining"
    )

    if (
        not args.overwrite
        and int(args.target_episodes) > 0
        and int(done_episode_count) >= int(args.target_episodes)
    ):
        groups, group_stats = [], {
            "selection_mode": "target_already_met",
            "eligible_trajectories": 0,
            "eligible_episodes": 0,
            "selected_trajectories": 0,
            "selected_episodes": 0,
            "skipped_groups_with_missing_gt": 0,
            "skipped_groups_with_too_long_episode": 0,
            "skipped_groups_with_too_short_episode": 0,
            "skipped_partial_groups": 0,
            "skipped_excluded_trajectories": len(done_trajectory_ids),
        }
    else:
        groups, group_stats = select_trajectory_groups(
            train_data=train_data,
            gt_trajectories=gt_trajectories,
            max_gt_actions=int(args.max_gt_actions),
            target_episodes=remaining_target,
            min_gt_actions=int(args.min_gt_actions),
            exclude_trajectory_ids=done_trajectory_ids,
        )
    selected_episode_ids = {
        int(episode_id)
        for _trajectory_id, episode_ids, _representative_id in groups
        for episode_id in episode_ids
    }
    representative_ids = [int(representative_id) for _tid, _ids, representative_id in groups]
    print(
        "Selected trajectory groups: "
        f"trajectories={len(groups)} episodes={len(selected_episode_ids)} "
        f"target_episodes={args.target_episodes}"
    )
    print(f"Selection stats: {json.dumps(group_stats, ensure_ascii=False)}")

    if args.dry_run_selection:
        preview = [
            {
                "trajectory_id": int(trajectory_id),
                "episode_ids": [int(v) for v in episode_ids],
                "representative_episode_id": int(representative_id),
                "representative_instruction": instruction_text(source_episode_by_id[int(representative_id)]),
            }
            for trajectory_id, episode_ids, representative_id in groups[:10]
        ]
        print(f"First selected groups: {json.dumps(preview, indent=2, ensure_ascii=False)}")
        return

    import habitat
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config as get_habitat_config
    from rlinf.envs.habitat.extensions import measures as _rlinf_measures  # noqa: F401

    habitat_data_path = prepare_habitat_data_path(
        data_path=str(train_json_path),
        split=str(args.split),
        out_dir=str(out_root),
    )
    cfg = get_habitat_config(str(args.config))
    with read_write(cfg):
        cfg.habitat.dataset.split = str(args.split)
        cfg.habitat.dataset.data_path = habitat_data_path
        cfg.habitat.dataset.scenes_dir = str(args.scenes_dir)
        ndtw_measure = cfg.habitat.task.measurements.ndtw
        ndtw_measure.SPLIT = str(args.split)
        ndtw_measure.GT_PATH = str(Path(args.gt_json).resolve())
        cfg.habitat.task.measurements.top_down_map.draw_shortest_path = False

    env = habitat.Env(config=cfg)
    env_episode_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    missing_env = [eid for eid in representative_ids if eid not in env_episode_by_id]
    if missing_env:
        raise RuntimeError(f"Representative episodes missing from Habitat env, e.g. {missing_env[:5]}")

    max_steps = int(cfg.habitat.environment.max_episode_steps)
    subgoal_distance = float(args.subgoal_distance)
    failures: List[dict] = []

    try:
        for group_idx, (trajectory_id, episode_ids, representative_id) in enumerate(groups):
            if remaining_max_traj >= 0 and group_idx >= remaining_max_traj:
                break

            source_episode = source_episode_by_id[int(representative_id)]
            env_episode = env_episode_by_id[int(representative_id)]
            gt_trajectory = gt_trajectories[int(representative_id)]
            instruction = instruction_text(source_episode).strip()
            artifact_name = f"traj{int(trajectory_id):06d}_rep_episode_id{int(representative_id):06d}"
            artifact_dir = out_root / artifact_name
            artifact_dir.mkdir(parents=True, exist_ok=True)

            if (artifact_dir / "subgoals_geodesic.json").exists() and not args.overwrite:
                print(f"[skip] {artifact_name} existing subgoals_geodesic.json")
                subgoals_by_trajectory[int(trajectory_id)] = load_json(
                    artifact_dir / "subgoals_geodesic.json"
                )
                continue

            episode_label = (
                f"{artifact_name} scene={source_episode.get('scene_id')} "
                f"traj={trajectory_id} group_size={len(episode_ids)}"
            )
            print(
                f"\n[{group_idx + 1}/{len(groups)}] Processing {episode_label} "
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
                if not args.no_video or not args.no_topdown:
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

                if not args.no_video:
                    write_subgoal_video_from_memory(
                        output_path=artifact_dir / "subgoals_geodesic.mp4",
                        steps=steps,
                        subgoals=enriched,
                        fps=int(args.video_fps),
                        highlight_frames=int(args.video_highlight_frames),
                        topdown_metric=topdown_metric,
                        sim=env.sim,
                        instruction=instruction,
                        subgoal_radius=float(args.subgoal_radius),
                    )

                if topdown_metric is not None:
                    write_topdown_image_from_memory(
                        output_path=artifact_dir / "topdown_subgoals.png",
                        topdown_metric=topdown_metric,
                        subgoals=enriched,
                        sim=env.sim,
                        radius=float(args.subgoal_radius),
                        max_size=int(args.topdown_max_size),
                    )

                subgoals_by_trajectory[int(trajectory_id)] = payload
                print(
                    f"  -> saved {artifact_dir}; subgoals={len(enriched)} "
                    f"distance={subgoal_distance}"
                )
            except Exception as exc:
                failure = {
                    "trajectory_id": int(trajectory_id),
                    "episode_ids": [int(v) for v in episode_ids],
                    "representative_episode_id": int(representative_id),
                    "error": repr(exc),
                }
                failures.append(failure)
                append_jsonl(out_root / "failures.jsonl", failure)
                if not args.continue_on_error:
                    raise
                print(f"  !! failed {artifact_name}: {exc!r}")
    finally:
        env.close()

    output_episodes: List[dict] = []
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
        "subgoal_distance": subgoal_distance,
        "subgoal_selection_method": "geodesic_distance",
        "resumed_trajectories": len(done_trajectory_ids),
        "resumed_episodes": int(done_episode_count),
        "remaining_target_episodes": int(remaining_target),
        "remaining_max_trajectories": int(remaining_max_traj),
        "failures": failures,
        **group_stats,
    }
    write_json(output_json, output_data, pretty=bool(args.pretty_output))
    print(
        "\nDone: "
        f"wrote={output_json} episodes={len(output_episodes)} "
        f"modified={modified} selected_without_payload={skipped_selected_without_payload} "
        f"failures={len(failures)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a train split with geodesic-distance sub-goals (no GPT)."
    )
    parser.add_argument("--config", type=str, default="rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train_json", type=str, default="VLN-CE/datasets/rxr/train/train_guide_reachable.json.gz")
    parser.add_argument("--gt_json", type=str, default="VLN-CE/datasets/rxr/train/train_guide_gt_reachable.json.gz")
    parser.add_argument("--scenes_dir", type=str, default="VLN-CE/scene_dataset")
    parser.add_argument("--out_dir", type=str, default="VLN-CE/datasets/rxr/train_subgoal_geodesic")
    parser.add_argument(
        "--output_json",
        type=str,
        default="VLN-CE/datasets/rxr/train_subgoal_geodesic/train_guide_subgoals_geodesic.json.gz",
    )
    parser.add_argument("--max_trajectories", type=int, default=1)
    parser.add_argument("--target_episodes", type=int, default=1)
    parser.add_argument("--max_gt_actions", type=int, default=200)
    parser.add_argument("--min_gt_actions", type=int, default=150)
    parser.add_argument("--subgoal_radius", type=float, default=2.0)
    parser.add_argument(
        "--subgoal_distance",
        type=float,
        required=True,
        help="Geodesic distance interval D (meters) between consecutive sub-goals.",
    )
    parser.add_argument("--include_unselected", action="store_true")
    parser.add_argument("--pretty_output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run_selection", action="store_true", help="Preview selected trajectory groups only.")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=6)
    parser.add_argument("--video_highlight_frames", type=int, default=3)
    parser.add_argument("--no_topdown", action="store_true", help="Skip the final top-down map image.")
    parser.add_argument("--topdown_max_size", type=int, default=2000)
    return parser


def main() -> None:
    build_dataset_geodesic(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
```

Note: the Task 1 file had only `import math`, `from typing import List`, `from .replay import MemoryStep`, plus the two functions. This step replaces those imports with the full import block above and appends `_build_geodesic_subgoal_dicts` / `build_dataset_geodesic` / `build_arg_parser` / `main`. The two Task 1 functions stay verbatim.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /data/RLinf && python -m pytest tests/unit_tests/test_geodesic_subgoal_selection.py -v`
Expected: 8 passed (5 selector + 3 arg-parser).

- [ ] **Step 5: Verify module imports without habitat**

Run: `cd /data/RLinf && python -c "from subgoal_pipeline import build_dataset_geodesic; print('ok')"`
Expected: prints `ok` (no habitat import triggered).

- [ ] **Step 6: Lint**

Run: `cd /data/RLinf && /opt/venv/habitat/bin/ruff check subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_selection.py`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/build_dataset_geodesic.py tests/unit_tests/test_geodesic_subgoal_selection.py
git commit -s -m "feat: add geodesic-distance sub-goal dataset builder"
```

---

### Task 3: Document the geodesic builder in the README

**Files:**
- Modify: `subgoal_pipeline/README.md`

**Interfaces:**
- Consumes: the CLI from Task 2.

- [ ] **Step 1: Add a "Geodesic variant" subsection**

In `subgoal_pipeline/README.md`, after the existing "Build the dataset" command block (around line 60), insert:

```markdown
## Geodesic variant (no GPT)

`build_dataset_geodesic.py` selects sub-goals deterministically every `--subgoal_distance`
meters of `sim.geodesic_distance` along the replayed GT trajectory, drops a trailing
sub-goal that is closer than `D` to the final goal, and keeps the original episode final
goal. No OpenAI call, no frame sampling, no token usage. Spacing is regular by
construction, so `shift_subgoals.py` / `regularize_spacing.py` are not needed.

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
```

And in the `## Files` list, append:

```markdown
- `build_dataset_geodesic.py`: GPT-free geodesic-distance sub-goal pipeline.
```

- [ ] **Step 2: Commit**

```bash
cd /data/RLinf
git add subgoal_pipeline/README.md
git commit -s -m "docs: document geodesic sub-goal builder"
```

---

## Self-Review notes

- Spec §2 (algorithm) → Task 1. `inf`/`nan` fallback + print → Task 1 `_geodesic`. Iterative tail removal → Task 1 loop + `test_iterative_double_tail_removal`.
- Spec §3 (payload) → Task 2 payload dict: `subgoal_selection_method`, `subgoal_distance`, removed `model`/`token_usage`/`subgoal_instruction_source`, `sampled_frame_steps` = best_steps, final forced to original goal.
- Spec §4 (CLI) → Task 2 `build_arg_parser`: `--subgoal_distance` required; GPT args absent (asserted by `test_gpt_only_args_are_absent`).
- Spec §5 (testing) → Task 1 test file covers all 5 cases; no Habitat e2e (by design).
- Spec §1 (file layout, artifact name) → Task 2 uses `subgoals_geodesic.json`, resume glob matches.
- Deviation logged in Global Constraints: `math.dist` instead of importing `euclidean_distance` from measures.py, to keep the selector habitat-free and unit-testable. Same euclidean semantics.
