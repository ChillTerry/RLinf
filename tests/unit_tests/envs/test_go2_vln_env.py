from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from rlinf.envs.realworld.go2.go2_vln_env import (
    Go2VLNEnv,
    NavPrimitive,
    PrimitiveResult,
    PrimitiveStatus,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models.embodiment.uninavid.nav_rollout import select_slot_rgb_frames


class ScriptedTransport:
    def __init__(self, results=()):
        self.results = deque(results)
        self.image_sequence = 0
        self.actions = []
        self.closed = False

    def wait_for_image(self, *, timeout_sec, after_sequence=None):
        del timeout_sec
        assert after_sequence is None or self.image_sequence <= after_sequence
        self.image_sequence += 1
        frame = np.full((3, 4, 3), self.image_sequence, dtype=np.uint8)
        return frame, self.image_sequence

    def execute(
        self,
        action,
        *,
        episode_id,
        sequence_id,
        timeout_sec,
    ):
        del timeout_sec
        self.actions.append((action, episode_id, sequence_id))
        if self.results:
            return self.results.popleft()
        return PrimitiveResult(PrimitiveStatus.SUCCEEDED)

    def close(self):
        self.closed = True


def make_env(transport, **overrides):
    override_cfg = {
        "image_width": 4,
        "image_height": 3,
        "max_num_steps": 5,
        "episode_id_base": 0,
    }
    override_cfg.update(overrides)
    return Go2VLNEnv(
        override_cfg=override_cfg,
        transport=transport,
    )


def test_reset_increments_episode_and_actions_are_sequenced():
    transport = ScriptedTransport()
    env = make_env(transport)

    first_obs, _ = env.reset()
    second_obs, _ = env.reset()
    obs, _, terminated, truncated, info = env.step(NavPrimitive.FORWARD)

    assert first_obs["state"]["episode_id"].item() == 1
    assert second_obs["state"]["episode_id"].item() == 2
    assert obs["frames"]["front_rgb"][0, 0, 0] == 3
    assert transport.actions == [(NavPrimitive.FORWARD, 2, 1)]
    assert not terminated
    assert not truncated
    assert info["action_status"] == "succeeded"


def test_instruction_queue_switches_episode_without_recreating_transport(tmp_path):
    first_output = tmp_path / "episode_one"
    second_output = tmp_path / "episode_two"
    queue_path = tmp_path / "instructions.jsonl"
    queue_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"instruction": "Go to the bag.", "output_dir": str(first_output)}
                ),
                json.dumps(
                    {"instruction": "Go to the box.", "output_dir": str(second_output)}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    transport = ScriptedTransport()
    env = make_env(transport, instruction_queue_path=str(queue_path))

    first_obs, _ = env.reset()
    env.step(NavPrimitive.STOP)
    second_obs, _ = env.reset()

    assert first_obs["state"]["episode_id"].item() == 1
    assert second_obs["state"]["episode_id"].item() == 2
    assert env.task_description == "Go to the box."
    assert (first_output / "episode_end.json").is_file()
    assert (second_output / "episode.json").is_file()
    assert not transport.closed


def test_manual_stop_request_overrides_next_model_action(tmp_path):
    transport = ScriptedTransport()
    env = make_env(transport, episode_output_dir=str(tmp_path))
    env.reset()
    (tmp_path / "manual_stop.request").touch()

    _, _, terminated, truncated, _ = env.step(NavPrimitive.FORWARD)
    end = json.loads((tmp_path / "episode_end.json").read_text(encoding="utf-8"))

    assert terminated
    assert not truncated
    assert transport.actions == [(NavPrimitive.STOP, 1, 1)]
    assert end["reason"] == "manual_stop"


def test_default_episode_ids_are_unique_across_process_like_env_instances(monkeypatch):
    timestamp_values = iter([1_000_000_000, 2_000_000_000])
    monkeypatch.setattr(
        "rlinf.envs.realworld.go2.go2_vln_env.time.time_ns",
        lambda: next(timestamp_values),
    )
    first_transport = ScriptedTransport()
    second_transport = ScriptedTransport()
    first_env = Go2VLNEnv(
        override_cfg={"image_width": 4, "image_height": 3},
        transport=first_transport,
    )
    second_env = Go2VLNEnv(
        override_cfg={"image_width": 4, "image_height": 3},
        transport=second_transport,
    )

    first_env.reset()
    second_env.reset()
    first_env.step(NavPrimitive.FORWARD)
    second_env.step(NavPrimitive.FORWARD)

    assert first_transport.actions[0][1:] == (1_000_000_001, 1)
    assert second_transport.actions[0][1:] == (2_000_000_001, 1)


def test_obstacle_allows_replanning_but_control_failure_truncates():
    transport = ScriptedTransport(
        [
            PrimitiveResult(PrimitiveStatus.OBSTACLE, "blocked"),
            PrimitiveResult(PrimitiveStatus.STALE_STATE, "stale"),
        ]
    )
    env = make_env(transport)
    env.reset()

    _, _, terminated, truncated, info = env.step(NavPrimitive.FORWARD)
    assert not terminated
    assert not truncated
    assert info["action_status"] == "obstacle"

    _, _, terminated, truncated, info = env.step(NavPrimitive.LEFT)
    assert not terminated
    assert truncated
    assert info["action_status"] == "stale_state"


def test_executor_timeout_stops_primitive_but_allows_replanning():
    transport = ScriptedTransport(
        [PrimitiveResult(PrimitiveStatus.TIMEOUT, "primitive deadline")]
    )
    env = make_env(transport)
    env.reset()

    _, _, terminated, truncated, info = env.step(NavPrimitive.LEFT)

    assert not terminated
    assert not truncated
    assert info["action_status"] == "timeout"
    assert info["action_timeout_recoverable"] is True


def test_executor_timeout_can_be_configured_to_truncate():
    transport = ScriptedTransport(
        [PrimitiveResult(PrimitiveStatus.TIMEOUT, "primitive deadline")]
    )
    env = make_env(transport, truncate_on_action_timeout=True)
    env.reset()

    _, _, terminated, truncated, info = env.step(NavPrimitive.LEFT)

    assert not terminated
    assert truncated
    assert info["action_timeout_recoverable"] is False


def test_stop_terminates_and_close_is_idempotent():
    transport = ScriptedTransport()
    env = make_env(transport)
    env.reset()

    _, _, terminated, truncated, _ = env.step(NavPrimitive.STOP)
    env.close()
    env.close()

    assert terminated
    assert not truncated
    assert transport.closed


def test_actions_after_stop_are_latched_and_never_executed():
    transport = ScriptedTransport()
    env = make_env(transport)
    env.reset()

    env.step(NavPrimitive.STOP)
    _, _, terminated, truncated, info = env.step(NavPrimitive.FORWARD)

    assert terminated
    assert not truncated
    assert info["action_status"] == "episode_finished"
    assert transport.actions == [(NavPrimitive.STOP, 1, 1)]


def test_next_image_is_requested_only_after_action_and_settle(monkeypatch):
    events = []

    class OrderedTransport(ScriptedTransport):
        def wait_for_image(self, *, timeout_sec, after_sequence=None):
            events.append("image")
            return super().wait_for_image(
                timeout_sec=timeout_sec,
                after_sequence=after_sequence,
            )

        def execute(self, action, **kwargs):
            events.append("action_completed")
            return super().execute(action, **kwargs)

    monkeypatch.setattr(
        "rlinf.envs.realworld.go2.go2_vln_env.time.sleep",
        lambda seconds: events.append(("settle", seconds)),
    )
    transport = OrderedTransport()
    env = make_env(transport, post_action_settle_sec=0.3)

    env.reset()
    env.step(NavPrimitive.LEFT)

    assert events == ["image", "action_completed", ("settle", 0.3), "image"]


def test_realworld_uninavid_chunk_exposes_both_post_action_frames():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.cfg = SimpleNamespace(model_type="uninavid")
    obs_list = [
        {"main_images": torch.full((1, 3, 4, 3), 1, dtype=torch.uint8)},
        {"main_images": torch.full((1, 3, 4, 3), 2, dtype=torch.uint8)},
    ]

    env._attach_rgb_frame_history(obs_list)

    history = obs_list[-1]["rgb_frame_history"]
    assert history.shape == (1, 2, 3, 4, 3)
    assert history[0, :, 0, 0, 0].tolist() == [1, 2]
    assert obs_list[-1]["rgb_frame_history_lengths"].tolist() == [2]


def test_stop_marker_is_written_only_after_stop_result_and_final_image(tmp_path):
    marker_path = tmp_path / "model_stop.marker"
    events = []

    class StopTransport(ScriptedTransport):
        def execute(self, action, **kwargs):
            assert action == NavPrimitive.STOP
            assert not marker_path.exists()
            events.append("stop_completed")
            return super().execute(action, **kwargs)

        def wait_for_image(self, *, timeout_sec, after_sequence=None):
            if after_sequence is not None:
                assert not marker_path.exists()
                events.append("final_image")
            return super().wait_for_image(
                timeout_sec=timeout_sec,
                after_sequence=after_sequence,
            )

    env = make_env(
        StopTransport(),
        episode_output_dir=str(tmp_path),
    )
    env.reset()
    _, _, terminated, _, _ = env.step(NavPrimitive.STOP)

    assert terminated
    assert events == ["stop_completed", "final_image"]
    assert marker_path.is_file()


def test_dry_run_records_actions_and_frames_without_executing(tmp_path, capsys):
    transport = ScriptedTransport()
    env = make_env(
        transport,
        dry_run_no_motion=True,
        episode_output_dir=str(tmp_path),
        task_description="Go to the door and stop.",
    )

    env.reset()
    _, _, terminated, truncated, info = env.step(NavPrimitive.FORWARD)
    assert not terminated
    assert not truncated
    assert info["dry_run_no_motion"] is True
    assert transport.actions == []

    _, _, terminated, _, _ = env.step(NavPrimitive.STOP)
    assert terminated
    assert transport.actions == []

    output = capsys.readouterr().out
    assert "predicted_action=FORWARD(1) execute=false" in output
    assert "predicted_action=STOP(0) execute=false" in output
    assert sorted(path.name for path in (tmp_path / "frames").glob("*.ppm")) == [
        "frame_000000.ppm",
        "frame_000001.ppm",
        "frame_000002.ppm",
    ]
    actions = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text().splitlines()
    ]
    assert [item["predicted_action"] for item in actions] == ["forward", "stop"]
    assert all(not item["executed"] for item in actions)
    results = [
        json.loads(line)
        for line in (tmp_path / "action_results.jsonl").read_text().splitlines()
    ]
    assert [item["status"] for item in results] == ["succeeded", "succeeded"]
    assert "[GO2 RESULT]" in output
    assert (tmp_path / "model_stop.marker").is_file()


def test_uninavid_uses_realworld_main_image_when_history_is_absent():
    main_images = torch.arange(2 * 3 * 4 * 3, dtype=torch.uint8).reshape(2, 3, 4, 3)
    frames = select_slot_rgb_frames({"main_images": main_images}, slot_id=1)

    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], main_images[1].numpy())


def test_uninavid_preserves_habitat_rgb_history_path():
    history = torch.zeros((1, 3, 2, 2, 3), dtype=torch.uint8)
    history[:, 1] = 7
    frames = select_slot_rgb_frames(
        {
            "rgb_frame_history": history,
            "rgb_frame_history_lengths": torch.tensor([2]),
            "main_images": torch.full((1, 2, 2, 3), 99, dtype=torch.uint8),
        },
        slot_id=0,
    )

    assert len(frames) == 2
    assert frames[1][0, 0, 0] == 7
