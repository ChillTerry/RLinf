# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


class NavPrimitive(IntEnum):
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    NO_OP = 4


class PrimitiveStatus(IntEnum):
    SUCCEEDED = 0
    OBSTACLE = 1
    TIMEOUT = 2
    ESTOP = 3
    CONTROL_ERROR = 4
    STALE_STATE = 5
    REJECTED = 6


@dataclass(frozen=True)
class PrimitiveResult:
    status: PrimitiveStatus
    message: str = ""
    distance_m: float = 0.0
    yaw_rad: float = 0.0


class Go2VLNTransport(Protocol):
    def wait_for_image(
        self, *, timeout_sec: float, after_sequence: int | None = None
    ) -> tuple[np.ndarray, int]: ...

    def execute(
        self,
        action: NavPrimitive,
        *,
        episode_id: int,
        sequence_id: int,
        timeout_sec: float,
    ) -> PrimitiveResult: ...

    def close(self) -> None: ...


@dataclass
class Go2VLNConfig:
    task_description: str = "Walk forward and stop at the end of the hallway."
    socket_host: str = "192.168.3.15"
    socket_tcp_port: int = 8765
    socket_udp_heartbeat_port: int = 8766
    socket_connect_timeout_sec: float = 5.0
    socket_heartbeat_interval_sec: float = 0.2
    socket_auth_token_env: str = "GO2_VLN_TOKEN"
    socket_require_auth: bool = True
    image_width: int = 640
    image_height: int = 480
    image_timeout_sec: float = 5.0
    action_result_timeout_sec: float = 15.0
    truncate_on_action_timeout: bool = False
    post_action_settle_sec: float = 0.3
    post_action_image_timeout_sec: float = 5.0
    max_num_steps: int = 300
    is_dummy: bool = False
    dummy_frame_value: int = 0
    dry_run_no_motion: bool = False
    episode_output_dir: str | None = None
    # Optional JSONL queue used by the interactive driver.  Each reset consumes
    # one {"instruction": ..., "output_dir": ...} record, allowing multiple
    # episodes to share one already-loaded inference worker.
    instruction_queue_path: str | None = None
    instruction_queue_poll_sec: float = 0.1
    # None selects a nanosecond timestamp base, preventing sequence replay
    # across persistent-session restarts; tests may inject 0.
    episode_id_base: int | None = None


class DummyGo2VLNTransport:
    """Deterministic transport used by unit tests and config smoke tests."""

    def __init__(self, config: Go2VLNConfig):
        self._config = config
        self._image_sequence = 0
        self.executed_actions: list[NavPrimitive] = []
        self.closed = False

    def wait_for_image(
        self, *, timeout_sec: float, after_sequence: int | None = None
    ) -> tuple[np.ndarray, int]:
        del timeout_sec, after_sequence
        self._image_sequence += 1
        frame = np.full(
            (self._config.image_height, self._config.image_width, 3),
            self._config.dummy_frame_value,
            dtype=np.uint8,
        )
        return frame, self._image_sequence

    def execute(
        self,
        action: NavPrimitive,
        *,
        episode_id: int,
        sequence_id: int,
        timeout_sec: float,
    ) -> PrimitiveResult:
        del episode_id, sequence_id, timeout_sec
        self.executed_actions.append(action)
        return PrimitiveResult(status=PrimitiveStatus.SUCCEEDED)

    def close(self) -> None:
        self.closed = True


class Go2VLNEnv(gym.Env):
    """One-robot VLN environment backed by the Go2 socket gateway."""

    CONFIG_CLS = Go2VLNConfig
    metadata = {"render_modes": []}

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info=None,
        hardware_info=None,
        env_idx: int = 0,
        transport: Go2VLNTransport | None = None,
    ):
        del worker_info, hardware_info
        if env_idx != 0:
            raise ValueError("Go2VLNEnv supports exactly one robot environment.")

        config_fields = Go2VLNConfig.__dataclass_fields__
        unknown_keys = sorted(set(override_cfg) - set(config_fields))
        if unknown_keys:
            raise ValueError(f"Unknown Go2 VLN config keys: {unknown_keys}")
        self.config = Go2VLNConfig(**override_cfg)
        if self.config.post_action_settle_sec < 0:
            raise ValueError("post_action_settle_sec cannot be negative.")
        episode_id_base = self.config.episode_id_base
        if episode_id_base is None:
            episode_id_base = time.time_ns()
        episode_id_base = int(episode_id_base)
        if episode_id_base < 0 or episode_id_base >= (1 << 63) - 1:
            raise ValueError("episode_id_base must be in [0, 2^63 - 2].")
        self.task_description = self.config.task_description
        self._episode_id = episode_id_base
        self._sequence_id = 0
        self._num_steps = 0
        self._last_image_sequence: int | None = None
        self._closed = False
        self._episode_finished = False
        self._last_frame: np.ndarray | None = None
        self._episode_output_dir: Path | None = None
        self._frames_dir: Path | None = None
        self._instruction_queue_index = 0
        self._queued_output_dir: str | None = None

        if transport is not None:
            self._transport = transport
        elif self.config.is_dummy:
            self._transport = DummyGo2VLNTransport(self.config)
        else:
            from .socket_transport import SocketGo2VLNTransport

            self._transport = SocketGo2VLNTransport(self.config)

        self.action_space = gym.spaces.Discrete(len(NavPrimitive))
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "episode_id": gym.spaces.Box(
                            low=0,
                            high=np.iinfo(np.int64).max,
                            shape=(1,),
                            dtype=np.int64,
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "front_rgb": gym.spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                self.config.image_height,
                                self.config.image_width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                    }
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        self._consume_next_instruction()
        self._episode_id += 1
        self._sequence_id = 0
        self._num_steps = 0
        self._episode_finished = False
        frame = self._wait_for_fresh_image(self.config.image_timeout_sec)
        self._last_frame = frame
        self._prepare_episode_output()
        self._save_episode_frame(frame, step=0)
        return self._observation(frame), {}

    def step(self, action):
        if self._episode_finished:
            if self._last_frame is None:
                raise RuntimeError("Finished Go2 episode has no final RGB frame.")
            print(
                f"[GO2 MODEL] episode={self._episode_id} "
                "ignored_action_after_episode_end=true",
                flush=True,
            )
            return (
                self._observation(self._last_frame),
                0.0,
                True,
                False,
                {"action": "ignored", "action_status": "episode_finished"},
            )

        manual_stop_requested = bool(
            self._episode_output_dir is not None
            and (self._episode_output_dir / "manual_stop.request").is_file()
        )
        primitive = (
            NavPrimitive.STOP
            if manual_stop_requested
            else self._normalize_action(action)
        )
        if manual_stop_requested:
            print(
                f"[GO2 EPISODE] episode={self._episode_id} "
                "manual_stop_requested=true",
                flush=True,
            )
        self._sequence_id += 1
        self._num_steps += 1
        self._record_predicted_action(primitive)

        action_result_received = False
        try:
            if self.config.dry_run_no_motion:
                result = PrimitiveResult(
                    status=PrimitiveStatus.SUCCEEDED,
                    message="Dry run: predicted action was not sent to the robot.",
                )
            else:
                result = self._transport.execute(
                    primitive,
                    episode_id=self._episode_id,
                    sequence_id=self._sequence_id,
                    timeout_sec=self.config.action_result_timeout_sec,
                )
            action_result_received = True
            settle_sec = (
                self.config.post_action_settle_sec
                if not self.config.dry_run_no_motion
                and primitive in {
                    NavPrimitive.FORWARD,
                    NavPrimitive.LEFT,
                    NavPrimitive.RIGHT,
                }
                else 0.0
            )
            if settle_sec > 0:
                time.sleep(settle_sec)
            print(
                f"[GO2 OBSERVE] episode={self._episode_id} "
                f"step={self._num_steps} action_result={result.status.name} "
                f"settle_sec={settle_sec:.3f} settle_complete=true "
                "requesting_post_action_rgb=true",
                flush=True,
            )
            frame = self._wait_for_fresh_image(
                self.config.post_action_image_timeout_sec
            )
        except TimeoutError as exc:
            frame = self._fallback_frame()
            result = PrimitiveResult(
                status=PrimitiveStatus.TIMEOUT,
                message=str(exc),
            )
        except Exception as exc:
            frame = self._fallback_frame()
            result = PrimitiveResult(
                status=PrimitiveStatus.CONTROL_ERROR,
                message=str(exc),
            )
        self._record_action_result(primitive, result)
        self._save_episode_frame(frame, step=self._num_steps)
        self._last_frame = frame

        terminated = primitive == NavPrimitive.STOP
        if (
            terminated
            and not manual_stop_requested
            and self._episode_output_dir is not None
        ):
            self._atomic_write_text(
                self._episode_output_dir / "model_stop.marker",
                f"step={self._num_steps}\n",
            )
        recoverable_action_timeout = (
            result.status == PrimitiveStatus.TIMEOUT
            and action_result_received
            and not self.config.truncate_on_action_timeout
        )
        unsafe_failure = result.status in {
            PrimitiveStatus.ESTOP,
            PrimitiveStatus.CONTROL_ERROR,
            PrimitiveStatus.STALE_STATE,
            PrimitiveStatus.REJECTED,
        } or (
            result.status == PrimitiveStatus.TIMEOUT
            and not recoverable_action_timeout
        )
        truncated = unsafe_failure or self._num_steps >= self.config.max_num_steps
        self._episode_finished = terminated or truncated
        if self._episode_finished and self._episode_output_dir is not None:
            self._atomic_write_text(
                self._episode_output_dir / "episode_end.json",
                json.dumps(
                    {
                        "episode_id": self._episode_id,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "reason": (
                            "manual_stop"
                            if manual_stop_requested
                            else "model_stop"
                            if terminated
                            else result.status.name.lower()
                            if unsafe_failure
                            else "max_steps"
                        ),
                        "steps": self._num_steps,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
        info = {
            "action": primitive.name.lower(),
            "action_status": result.status.name.lower(),
            "action_message": result.message,
            "distance_m": result.distance_m,
            "yaw_rad": result.yaw_rad,
            "dry_run_no_motion": self.config.dry_run_no_motion,
            "action_timeout_recoverable": recoverable_action_timeout,
        }
        return self._observation(frame), 0.0, terminated, truncated, info

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._transport.close()

    def _wait_for_fresh_image(self, timeout_sec: float) -> np.ndarray:
        frame, image_sequence = self._transport.wait_for_image(
            timeout_sec=timeout_sec,
            after_sequence=self._last_image_sequence,
        )
        frame = np.asarray(frame)
        expected_shape = (
            self.config.image_height,
            self.config.image_width,
            3,
        )
        if frame.shape != expected_shape:
            raise ValueError(
                f"Expected RGB image shape {expected_shape}, got {frame.shape}."
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._last_image_sequence = int(image_sequence)
        return frame

    def _fallback_frame(self) -> np.ndarray:
        return np.zeros(
            (self.config.image_height, self.config.image_width, 3),
            dtype=np.uint8,
        )

    def _observation(self, frame: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
        return {
            "state": {
                "episode_id": np.asarray([self._episode_id], dtype=np.int64),
            },
            "frames": {"front_rgb": frame},
        }

    def _prepare_episode_output(self) -> None:
        configured_output_dir = self._queued_output_dir or self.config.episode_output_dir
        if not configured_output_dir:
            self._episode_output_dir = None
            self._frames_dir = None
            return
        output_dir = Path(configured_output_dir).expanduser().resolve()
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        self._episode_output_dir = output_dir
        self._frames_dir = frames_dir

        metadata = {
            "episode_id": self._episode_id,
            "instruction": self.task_description,
            "dry_run_no_motion": self.config.dry_run_no_motion,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "image_width": self.config.image_width,
            "image_height": self.config.image_height,
        }
        self._atomic_write_text(
            output_dir / "episode.json",
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        )

    def _consume_next_instruction(self) -> None:
        """Wait for and consume one instruction without recreating the model."""
        queue_path_value = self.config.instruction_queue_path
        if not queue_path_value:
            self._queued_output_dir = None
            return

        queue_path = Path(queue_path_value).expanduser().resolve()
        poll_sec = max(float(self.config.instruction_queue_poll_sec), 0.01)
        announced = False
        while not self._closed:
            if queue_path.is_file():
                records = queue_path.read_text(encoding="utf-8").splitlines()
                if self._instruction_queue_index < len(records):
                    raw_record = records[self._instruction_queue_index]
                    self._instruction_queue_index += 1
                    request = json.loads(raw_record)
                    instruction = str(request.get("instruction", "")).strip()
                    if not instruction:
                        raise ValueError("Queued Go2 instruction cannot be empty.")
                    output_dir = request.get("output_dir")
                    self.task_description = instruction
                    self._queued_output_dir = (
                        str(output_dir) if output_dir is not None else None
                    )
                    print(
                        "[GO2 EPISODE] instruction_received=true "
                        f"queue_index={self._instruction_queue_index} "
                        f"instruction={instruction!r}",
                        flush=True,
                    )
                    return
            if not announced:
                print(
                    f"[GO2 EPISODE] waiting_for_instruction={queue_path}",
                    flush=True,
                )
                announced = True
            time.sleep(poll_sec)

        raise RuntimeError("Go2 environment closed while waiting for an instruction.")

    def _record_predicted_action(self, primitive: NavPrimitive) -> None:
        execute = not self.config.dry_run_no_motion
        message = (
            f"[GO2 MODEL] episode={self._episode_id} "
            f"step={self._num_steps} "
            f"predicted_action={primitive.name}({int(primitive)}) "
            f"execute={str(execute).lower()}"
        )
        print(message, flush=True)

        if self._episode_output_dir is None:
            return
        record = {
            "episode_id": self._episode_id,
            "step": self._num_steps,
            "predicted_action": primitive.name.lower(),
            "action_id": int(primitive),
            "executed": execute,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (self._episode_output_dir / "actions.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
    def _save_episode_frame(self, frame: np.ndarray, *, step: int) -> None:
        if self._frames_dir is None:
            return
        output_path = self._frames_dir / f"frame_{step:06d}.ppm"
        temporary_path = output_path.with_suffix(".ppm.tmp")
        with temporary_path.open("wb") as output:
            output.write(
                f"P6\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii")
            )
            output.write(np.ascontiguousarray(frame).tobytes(order="C"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)

    def _record_action_result(
        self,
        primitive: NavPrimitive,
        result: PrimitiveResult,
    ) -> None:
        message = (
            f"[GO2 RESULT] episode={self._episode_id} "
            f"step={self._num_steps} "
            f"action={primitive.name} "
            f"status={result.status.name} "
            f"distance_m={result.distance_m:.3f} "
            f"yaw_rad={result.yaw_rad:.3f} "
            f"message={result.message!r}"
        )
        print(message, flush=True)

        if self._episode_output_dir is None:
            return
        record = {
            "episode_id": self._episode_id,
            "step": self._num_steps,
            "action": primitive.name.lower(),
            "action_id": int(primitive),
            "status": result.status.name.lower(),
            "message": result.message,
            "distance_m": result.distance_m,
            "yaw_rad": result.yaw_rad,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        with (self._episode_output_dir / "action_results.jsonl").open(
            "a", encoding="utf-8"
        ) as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    @staticmethod
    def _atomic_write_text(path: Path, content: str) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)

    @staticmethod
    def _normalize_action(action) -> NavPrimitive:
        action_array = np.asarray(action)
        if action_array.size != 1:
            raise ValueError(
                f"Go2 VLN expects one discrete action, got shape {action_array.shape}."
            )
        try:
            return NavPrimitive(int(action_array.reshape(-1)[0]))
        except ValueError as exc:
            raise ValueError(f"Unsupported Go2 VLN action: {action!r}") from exc
