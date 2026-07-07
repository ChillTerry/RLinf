import argparse
import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

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
from .gt import (
    build_dataset_indices,
    load_ground_truth_trajectories,
    select_trajectory_groups,
)
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
    steps: list[MemoryStep],
    sim,
    subgoal_distance: float,
    final_goal_position: list[float],
    *,
    episode_label: str = "",
) -> list[int]:
    selected: list[int] = []
    if len(steps) < 2:
        return selected
    anchor = steps[0].agent_position
    # The STOP frame (last step) is reserved as the final sub-goal; never select it as an intermediate.
    for i in range(1, len(steps) - 1):
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


def _build_geodesic_subgoal_dicts(steps: list[MemoryStep], selected_indices: list[int]) -> list[dict]:
    subgoals: list[dict] = []
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
    active_gpus = [int(g) for g in gpu_ids if int(scenes_per_gpu.get(int(g), 0)) > 0]
    if not active_gpus:
        return {}
    n_active = len(active_gpus)
    base = num_processes // n_active
    remainder = num_processes % n_active
    out: dict[int, int] = {}
    for idx, gpu_id in enumerate(active_gpus):
        scene_count = int(scenes_per_gpu.get(int(gpu_id), 0))
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

    subgoals_by_trajectory: dict[int, dict] = {}
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
    failures: list[dict] = []

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
