#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run interactive Uni-NaVid episodes with real Go2 images, with "
            "motion disabled by default."
        )
    )
    parser.add_argument("--host", required=True, help="Go2 gateway IP address.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Safety cap per episode. Default: 100",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=2.0,
        help="Saved video playback FPS. Default: 2",
    )
    parser.add_argument(
        "--output-root",
        default=None,
    )
    parser.add_argument(
        "--config-name",
        default="realworld_go2_vln_eval_uninavid_original",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum episodes handled by one loaded model. Default: 1000",
    )
    parser.add_argument(
        "--execute-actions",
        action="store_true",
        help=(
            "Actually send model actions to the robot. Without this option "
            "all predictions are intercepted locally."
        ),
    )
    return parser.parse_args()


def terminate_process_group(process: subprocess.Popen, grace_sec: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_sec)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def stream_output(
    process: subprocess.Popen,
    log_path: Path,
) -> None:
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_file:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()


def save_video(episode_dir: Path, timestamp: str, fps: float) -> Path | None:
    frame_paths = sorted((episode_dir / "frames").glob("frame_*.ppm"))
    if not frame_paths:
        print("No completed RGB frames were available; video was not created.")
        return None

    import imageio.v2 as imageio

    video_path = episode_dir / f"{timestamp}.mp4"
    writer = imageio.get_writer(video_path, fps=fps)
    try:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    finally:
        writer.close()
    print(f"Video saved: {video_path}")
    return video_path


def write_driver_metadata(
    episode_dir: Path,
    *,
    timestamp: str,
    instruction: str,
    stop_reason: str,
    return_code: int,
) -> None:
    metadata_path = episode_dir / "driver_result.json"
    metadata_path.write_text(
        json.dumps(
            {
                "episode_timestamp": timestamp,
                "instruction": instruction,
                "stop_reason": stop_reason,
                "return_code": return_code,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def start_persistent_runtime(
    args: argparse.Namespace,
    repo_root: Path,
    session_dir: Path,
    instruction_queue: Path,
) -> tuple[subprocess.Popen, threading.Thread]:
    # Uni-NaVid executes two selected actions per inference. Round only the
    # rollout loop length up; Go2VLNEnv still enforces the exact user cap.
    rollout_steps = ((args.max_steps + 1) // 2) * 2
    embodiment_dir = repo_root / "examples" / "embodiment"
    config_dir = embodiment_dir / "config"
    entrypoint = embodiment_dir / "eval_embodied_agent.py"
    env = os.environ.copy()
    env["EMBODIED_PATH"] = str(embodiment_dir)
    env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
    env["HYDRA_FULL_ERROR"] = "1"
    command = [
        sys.executable,
        str(entrypoint),
        "--config-path",
        str(config_dir),
        "--config-name",
        args.config_name,
        f"runner.logger.log_path={session_dir / 'rlinf'}",
        f"env.eval.override_cfg.socket_host={args.host}",
        (
            "env.eval.override_cfg.dry_run_no_motion="
            f"{str(not args.execute_actions).lower()}"
        ),
        "env.eval.override_cfg.episode_output_dir=null",
        f"env.eval.override_cfg.instruction_queue_path={instruction_queue}",
        f"algorithm.eval_rollout_epoch={args.max_episodes}",
        f"env.eval.max_steps_per_rollout_epoch={rollout_steps}",
        f"env.eval.max_episode_steps={args.max_steps}",
        f"env.eval.override_cfg.max_num_steps={args.max_steps}",
        "env.eval.video_cfg.save_video=false",
    ]

    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_thread = threading.Thread(
        target=stream_output,
        args=(process, session_dir / "run.log"),
        daemon=True,
    )
    output_thread.start()
    return process, output_thread


def enqueue_episode(
    instruction_queue: Path, *, instruction: str, episode_dir: Path
) -> None:
    record = json.dumps(
        {"instruction": instruction, "output_dir": str(episode_dir)},
        ensure_ascii=False,
    )
    with instruction_queue.open("a", encoding="utf-8") as output:
        output.write(record + "\n")
        output.flush()
        os.fsync(output.fileno())


def wait_for_episode(
    process: subprocess.Popen,
    episode_dir: Path,
) -> tuple[str, int]:
    end_path = episode_dir / "episode_end.json"
    manual_requested = False
    print("Press ENTER to request a controlled STOP; Ctrl+C does the same.\n")

    try:
        while process.poll() is None:
            if end_path.is_file():
                result = json.loads(end_path.read_text(encoding="utf-8"))
                return str(result.get("reason", "episode_finished")), int(
                    result.get("steps", 0)
                )
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable and not manual_requested:
                sys.stdin.readline()
                manual_requested = True
                (episode_dir / "manual_stop.request").touch()
                print("\nControlled STOP requested; waiting for the active primitive to finish.")
        raise RuntimeError(
            f"RLinf persistent process exited unexpectedly with code {process.returncode}."
        )
    except KeyboardInterrupt:
        if not manual_requested:
            (episode_dir / "manual_stop.request").touch()
            print("\nControlled STOP requested; waiting for the active primitive to finish.")
        while process.poll() is None:
            if end_path.is_file():
                result = json.loads(end_path.read_text(encoding="utf-8"))
                return str(result.get("reason", "manual_stop")), int(
                    result.get("steps", 0)
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"RLinf persistent process exited unexpectedly with code {process.returncode}."
        )


def run_episode(
    args: argparse.Namespace,
    instruction: str,
    repo_root: Path,
    process: subprocess.Popen,
    instruction_queue: Path,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    episode_dir = (repo_root / args.output_root / timestamp).resolve()
    episode_dir.mkdir(parents=True, exist_ok=False)
    enqueue_episode(
        instruction_queue, instruction=instruction, episode_dir=episode_dir
    )

    print(f"\nEpisode timestamp: {timestamp}")
    print(f"Instruction: {instruction}")
    if args.execute_actions:
        print("MOTION ENABLED: model actions WILL be sent to the robot.")
    else:
        print("Safety mode: model actions will NOT be sent to the robot.")

    stop_reason, steps = wait_for_episode(process, episode_dir)
    print(f"Episode completed: reason={stop_reason}, steps={steps}")
    write_driver_metadata(
        episode_dir,
        timestamp=timestamp,
        instruction=instruction,
        stop_reason=stop_reason,
        return_code=0,
    )
    save_video(episode_dir, timestamp, args.video_fps)
    print(f"Action log: {episode_dir / 'actions.jsonl'}")


def stop_persistent_runtime(
    process: subprocess.Popen, output_thread: threading.Thread, session_dir: Path
) -> None:
    if process.poll() is None:
        terminate_process_group(process)
    process.wait()
    output_thread.join(timeout=5.0)
    print(f"Session log: {session_dir / 'run.log'}")


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive.")
    if not os.environ.get("GO2_VLN_TOKEN"):
        raise RuntimeError("GO2_VLN_TOKEN is not set.")

    if args.output_root is None:
        args.output_root = (
            "logs/go2_model_motion"
            if args.execute_actions
            else "logs/go2_model_dry_run"
        )

    if args.execute_actions:
        print("WARNING: Uni-NaVid predictions will physically move the Go2.")
        print("Verify the physical estop, clear the area, and supervise the robot.")
        confirmation = input("Type MOVE to enable model-controlled motion: ").strip()
        if confirmation != "MOVE":
            print("Motion was not enabled.")
            return

    repo_root = Path(__file__).resolve().parents[3]
    session_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    session_dir = (
        repo_root / args.output_root / f"_session_{session_timestamp}"
    ).resolve()
    session_dir.mkdir(parents=True, exist_ok=False)
    instruction_queue = session_dir / "instruction_queue.jsonl"
    instruction_queue.touch()
    process, output_thread = start_persistent_runtime(
        args, repo_root, session_dir, instruction_queue
    )

    mode = "MOTION" if args.execute_actions else "DRY-RUN"
    print(f"Interactive Go2 Uni-NaVid {mode} mode")
    print("The model is loaded once and reused across episode resets.")
    print("Enter an empty instruction or 'q' to exit.")
    try:
        for _ in range(args.max_episodes):
            try:
                instruction = input("\nInstruction> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not instruction or instruction.lower() in {"q", "quit", "exit"}:
                break
            run_episode(
                args, instruction, repo_root, process, instruction_queue
            )
    finally:
        stop_persistent_runtime(process, output_thread, session_dir)


if __name__ == "__main__":
    main()
