import argparse
import json
import os
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
from .common import append_jsonl, instruction_text, load_json, prepare_habitat_data_path, write_json
from .gt import build_dataset_indices, load_ground_truth_trajectories, select_trajectory_groups
from .openai_client import call_openai_for_episode, sanitize_model_subgoals, usage_record, usage_token_counts
from .replay import replay_gt_actions_in_memory, sample_memory_steps_with_final
from .video import write_subgoal_video_from_memory


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
        raise RuntimeError(f"{output_json} already exists; pass --overwrite to replace it.")

    train_json_path = Path(args.train_json)
    if train_json_path.suffix != ".gz":
        raise RuntimeError(f"{train_json_path} must be a gzipped train JSON file ending with .gz.")

    train_data = load_json(train_json_path)
    original_episodes = train_data.get("episodes")
    if not isinstance(original_episodes, list):
        raise RuntimeError(f"{args.train_json} does not contain episodes.")

    source_episode_by_id, _by_trajectory = build_dataset_indices(train_data)
    gt_trajectories = load_ground_truth_trajectories(Path(args.gt_json))
    groups, group_stats = select_trajectory_groups(
        train_data=train_data,
        gt_trajectories=gt_trajectories,
        max_gt_actions=int(args.max_gt_actions),
        target_episodes=int(args.target_episodes),
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

    env = habitat.Env(config=cfg)
    env_episode_by_id = {int(ep.episode_id): ep for ep in env.episodes}
    missing_env = [eid for eid in representative_ids if eid not in env_episode_by_id]
    if missing_env:
        raise RuntimeError(f"Representative episodes missing from Habitat env, e.g. {missing_env[:5]}")

    client = OpenAI(**build_openai_client_kwargs(args))
    max_steps = int(cfg.habitat.environment.max_episode_steps)

    subgoals_by_trajectory: Dict[int, dict] = {}
    failures: List[dict] = []

    try:
        for group_idx, (trajectory_id, episode_ids, representative_id) in enumerate(groups):
            if int(args.max_trajectories) >= 0 and group_idx >= int(args.max_trajectories):
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
                )
                enriched = enrich_subgoals_from_memory(subgoals=sanitized, steps=steps)

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
                if not args.no_video:
                    write_subgoal_video_from_memory(
                        output_path=artifact_dir / "subgoals_openai.mp4",
                        steps=steps,
                        subgoals=enriched,
                        fps=int(args.video_fps),
                        highlight_frames=int(args.video_highlight_frames),
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
        if episode_id not in selected_episode_ids:
            if args.include_unselected:
                output_episodes.append(episode)
            continue

        trajectory_id = int(episode.get("trajectory_id", episode_id))
        payload = subgoals_by_trajectory.get(trajectory_id)
        if payload is None:
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
        "script": "subgoal_pipeline.build_dataset",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_json": str(train_json_path),
        "gt_json": str(args.gt_json),
        "out_dir": str(args.out_dir),
        "target_episodes": int(args.target_episodes),
        "include_unselected": bool(args.include_unselected),
        "selected_episodes": len(selected_episode_ids),
        "modified_episodes": int(modified),
        "representative_episodes": len(subgoals_by_trajectory),
        "max_gt_actions": int(args.max_gt_actions),
        "subgoal_radius": float(args.subgoal_radius),
        "model": str(args.model),
        "reasoning_effort": str(args.reasoning_effort),
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
    parser.add_argument("--out_dir", type=str, default="VLN-CE/datasets/rxr/train/train_guide_subgoals_reachable")
    parser.add_argument(
        "--output_json",
        type=str,
        default="VLN-CE/datasets/rxr/train/train_guide_subgoals_reachable.json.gz",
    )
    parser.add_argument("--target_episodes", type=int, default=1)
    parser.add_argument("--max_gt_actions", type=int, default=80)
    parser.add_argument("--subgoal_radius", type=float, default=3.0)
    parser.add_argument(
        "--include_unselected",
        action="store_true",
        help="Keep unprocessed original episodes. By default only selected episodes are written.",
    )
    parser.add_argument("--pretty_output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--dry_run_selection", action="store_true", help="Preview selected trajectory groups only.")
    parser.add_argument("--max_trajectories", type=int, default=-1)

    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["none", "low", "medium", "high", "xhigh"],
    )
    parser.add_argument("--max_output_tokens", type=int, default=4096)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--frame_stride", type=int, default=2)
    parser.add_argument("--min_step_gap", type=int, default=8)
    parser.add_argument("--image_format", type=str, default="png", choices=["jpeg", "png"])
    parser.add_argument("--jpeg_quality", type=int, default=70)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
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
    return parser


def main() -> None:
    build_dataset_online(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
