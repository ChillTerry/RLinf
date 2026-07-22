import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from .artifacts import (
    build_episode_meta,
    drop_trailing_subgoals_near_final_goal,
    enrich_subgoals_from_memory,
    subgoals_to_info,
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
from .openai_client import call_openai_for_episode, sanitize_model_subgoals, usage_record, usage_token_counts
from .replay import replay_gt_actions_in_memory, sample_memory_steps_with_final
from .video import write_subgoal_video_from_memory, write_topdown_image_from_memory


DEFAULT_OPENAI_USER_AGENT = "curl/7.81.0"


def build_openai_client_kwargs(args: argparse.Namespace) -> dict:
    client_kwargs = {"api_key": str(args.api_key).strip()}
    if str(args.base_url).strip():
        client_kwargs["base_url"] = str(args.base_url).strip()
    user_agent = str(getattr(args, "user_agent", "")).strip()
    if user_agent:
        client_kwargs["default_headers"] = {"User-Agent": user_agent}
    return client_kwargs


def build_dataset_online(args: argparse.Namespace) -> None:
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
        print(
            f"rxr English-only filter: {episodes_before} -> {len(train_data['episodes'])} episodes"
        )
    original_episodes = train_data.get("episodes")
    if not isinstance(original_episodes, list):
        raise RuntimeError(f"{args.train_json} does not contain episodes.")

    source_episode_by_id, by_trajectory = build_dataset_indices(train_data)
    gt_trajectories = load_ground_truth_trajectories(Path(args.gt_json))

    # Resume: detect already-processed trajectories from existing artifacts and
    # pre-load their payloads. target_episodes / max_trajectories are treated as
    # TOTAL budgets; the already-processed count is subtracted to get the
    # remaining budget for this run. Skipped entirely when --overwrite is set.
    subgoals_by_trajectory: Dict[int, dict] = {}
    done_trajectory_ids: set = set()
    done_episode_count = 0
    if not args.overwrite:
        for payload_path in sorted(out_root.glob("traj*/subgoals_openai.json")):
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

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing openai dependency. Install it with: pip install openai") from exc

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
        # Only draw the replayed GT-action trajectory on the top-down map;
        # drop the pre-baked geodesic shortest path ("reference path").
        cfg.habitat.task.measurements.top_down_map.draw_shortest_path = False

    env = habitat.Env(config=cfg)
    env_episode_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    missing_env = [eid for eid in representative_ids if eid not in env_episode_by_id]
    if missing_env:
        raise RuntimeError(f"Representative episodes missing from Habitat env, e.g. {missing_env[:5]}")

    client = OpenAI(**build_openai_client_kwargs(args))
    max_steps = int(cfg.habitat.environment.max_episode_steps)

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

            if (artifact_dir / "subgoals_openai.json").exists() and not args.overwrite:
                print(f"[skip] {artifact_name} existing subgoals_openai.json")
                payload = load_json(artifact_dir / "subgoals_openai.json")
                subgoals_by_trajectory[int(trajectory_id)] = payload
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

                sampled_steps = sample_memory_steps_with_final(
                    steps=steps,
                    frame_stride=int(args.frame_stride),
                    max_frames=int(args.max_frames),
                )
                model_out, token_usage = call_openai_for_episode(
                    client=client,
                    args=args,
                    instruction=instruction,
                    sampled_steps=sampled_steps,
                    episode_label=episode_label,
                )
                valid_steps = {int(step.step) for step in sampled_steps}
                final_step = int(steps[-1].step)
                sanitized, notes = sanitize_model_subgoals(
                    raw_subgoals=list(model_out.get("subgoals", [])),
                    valid_steps=valid_steps,
                    final_step=final_step,
                    min_step_gap=int(args.min_step_gap),
                    frame_steps=[int(step.step) for step in sampled_steps],
                )
                enriched = enrich_subgoals_from_memory(subgoals=sanitized, steps=steps)
                final_goal_position, _ = extract_goal_payload(env_episode)
                if final_goal_position is None:
                    raise RuntimeError(
                        f"episode {representative_id} has no final goal position."
                    )
                enriched = drop_trailing_subgoals_near_final_goal(
                    subgoals=enriched,
                    sim=env.sim,
                    final_goal_position=final_goal_position,
                    exclusion_distance=float(args.final_goal_exclusion_distance),
                )
                final = enriched[-1]
                final["subgoal_position"] = list(final_goal_position)
                final["subgoal_position_source"] = "original_train.goals[0].position"
                final["agent_position_at_keyframe"] = list(final_goal_position)

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
                    "model": str(args.model),
                    "reasoning_effort": str(args.reasoning_effort),
                    "min_step_gap": int(args.min_step_gap),
                    "final_goal_exclusion_distance": float(
                        args.final_goal_exclusion_distance
                    ),
                    "sampled_frame_steps": [int(step.step) for step in sampled_steps],
                    "goal_position": list(enriched[-1]["subgoal_position"]),
                    "goal_position_source": "final_subgoal_agent_position",
                    "token_usage": token_usage,
                    "subgoal_instruction_source": {
                        "mode": "longest_instruction_in_trajectory",
                        "trajectory_id": int(trajectory_id),
                        "episode_id": int(representative_id),
                        "episode_dir": artifact_name,
                        "instruction_length": len(instruction),
                        "instruction_text": instruction,
                    },
                    "subgoals": enriched,
                    "raw_model_subgoals": model_out.get("subgoals", []),
                    "postprocess_notes": notes,
                }
                validate_subgoal_payload(payload)

                write_json(artifact_dir / "subgoals_openai.json", payload, pretty=True)
                write_source_episode_artifact(
                    out_dir=artifact_dir,
                    source_episode=source_episode,
                    meta=meta,
                    selected_episode_ids=episode_ids,
                    representative_id=representative_id,
                )
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
                else:
                    topdown_metric = None

                if not args.no_video:
                    write_subgoal_video_from_memory(
                        output_path=artifact_dir / "subgoals_openai.mp4",
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

                append_jsonl(
                    out_root / str(args.usage_log_name),
                    usage_record(
                        args=args,
                        out_dir=artifact_dir,
                        meta=meta,
                        token_usage=token_usage,
                        num_frames=len(sampled_steps),
                    ),
                )
                subgoals_by_trajectory[int(trajectory_id)] = payload
                usage_counts = usage_token_counts(token_usage)
                print(
                    f"  -> saved {artifact_dir}; subgoals={len(enriched)} "
                    f"tokens={usage_counts['total_tokens']}"
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
            # Newly selected in this run but no payload (e.g. processing failed).
            if episode_id in selected_episode_ids:
                skipped_selected_without_payload += 1
            if args.include_unselected:
                output_episodes.append(episode)
            continue

        new_episode = deepcopy(episode)
        new_episode["info"] = subgoals_to_info(
            subgoals=list(payload["subgoals"]),
            original_episode=episode,
        )
        output_episodes.append(new_episode)
        modified += 1

    output_data = deepcopy(train_data)
    output_data["episodes"] = output_episodes
    output_data["_subgoal_generation"] = {
        "script": "subgoal_pipeline.build_dataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_json": str(train_json_path),
        "gt_json": str(args.gt_json),
        "out_dir": str(args.out_dir),
        "dataset_type": dataset_type,
        "target_episodes": int(args.target_episodes),
        "include_unselected": bool(args.include_unselected),
        "selected_episodes": len(selected_episode_ids) + int(done_episode_count),
        "modified_episodes": int(modified),
        "representative_episodes": len(subgoals_by_trajectory),
        "max_gt_actions": int(args.max_gt_actions),
        "min_gt_actions": int(args.min_gt_actions),
        "subgoal_radius": float(args.subgoal_radius),
        "final_goal_exclusion_distance": float(
            args.final_goal_exclusion_distance
        ),
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
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
    parser = argparse.ArgumentParser(description="Build an RxR train split with GPT-selected landmark sub-goals.")
    parser.add_argument("--config", type=str, default="rlinf/envs/habitat/extensions/config/vlnce_rxr_uninavid.yaml")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--train_json", type=str, default="VLN-CE/datasets/rxr/train/train_guide_reachable.json.gz")
    parser.add_argument("--gt_json", type=str, default="VLN-CE/datasets/rxr/train/train_guide_gt_reachable.json.gz")
    parser.add_argument("--scenes_dir", type=str, default="VLN-CE/scene_dataset")
    parser.add_argument("--out_dir", type=str, default="VLN-CE/datasets/rxr/train_subgoal")
    parser.add_argument(
        "--output_json",
        type=str,
        default="VLN-CE/datasets/rxr/train_subgoal/train_guide_subgoals_reachable.json.gz",
    )
    parser.add_argument("--max_trajectories", type=int, default=1)
    parser.add_argument("--target_episodes", type=int, default=1)
    parser.add_argument("--max_gt_actions", type=int, default=200)
    parser.add_argument("--min_gt_actions", type=int, default=150)
    parser.add_argument("--subgoal_radius", type=float, default=2.0)
    parser.add_argument(
        "--final_goal_exclusion_distance",
        type=float,
        default=3.0,
        help="Drop trailing intermediate sub-goals geodesically closer than this distance to the final goal.",
    )
    parser.add_argument(
        "--include_unselected",
        action="store_true",
        help="Keep unprocessed original episodes. By default only selected episodes are written.",
    )
    parser.add_argument("--pretty_output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run_selection", action="store_true", help="Preview selected trajectory groups only.")

    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--min_step_gap", type=int, default=8)
    parser.add_argument("--image_format", type=str, default="jpeg", choices=["jpeg", "png"])
    parser.add_argument("--jpeg_quality", type=int, default=70)
    parser.add_argument("--base_url", type=str, default="https://gmncode.com/v1")
    parser.add_argument("--api_key", type=str, default="sk-94348489ab3d5c069f1928c8dcf34bfdaa5495268200e103378ba97b193a7906")
    parser.add_argument(
        "--user-agent",
        dest="user_agent",
        type=str,
        default=DEFAULT_OPENAI_USER_AGENT,
        help="User-Agent sent by the OpenAI SDK client.",
    )
    parser.add_argument("--usage_log_name", type=str, default="subgoals_openai_usage.jsonl")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--video_fps", type=int, default=6)
    parser.add_argument("--video_highlight_frames", type=int, default=3)
    parser.add_argument("--no_topdown", action="store_true", help="Skip the final top-down map image.")
    parser.add_argument(
        "--topdown_max_size",
        type=int,
        default=2000,
        help="Long-edge pixel cap for the top-down image; 0 disables resizing.",
    )
    return parser


def main() -> None:
    build_dataset_online(build_arg_parser().parse_args())

 
if __name__ == "__main__":
    main()
