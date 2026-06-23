# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import torch

from rlinf.envs.habitat.extensions import video


def test_video_frame_concatenates_rgb_and_topdown_map(monkeypatch):
    rgb = np.full((2, 3, 3), 7, dtype=np.uint8)
    topdown = np.full((2, 3, 3), 19, dtype=np.uint8)

    monkeypatch.setattr(video, "render_topdown_map", lambda info: topdown)

    frame = video.build_rollout_video_frame(
        {"rgb": rgb, "instruction": {"text": "go"}},
        {"top_down_map": {"map": np.zeros((2, 3), dtype=np.uint8)}},
    )

    assert frame.shape[0] > 2
    assert frame.shape[1] == 6
    assert np.array_equal(frame[:2, :3], rgb)
    assert np.array_equal(frame[:2, 3:], topdown)


def test_video_instruction_display_is_not_all_caps():
    assert video.normalize_instruction_for_video("GO DOWN THE HALLWAY") == (
        "Go down the hallway"
    )
    assert video.normalize_instruction_for_video("Go down the hallway") == (
        "Go down the hallway"
    )


def test_instruction_panel_pads_frame_height_to_macro_block_size():
    base_frame = np.full((480, 1280, 3), 7, dtype=np.uint8)

    frame = video.append_instruction_panel(base_frame, "")

    assert frame.shape == (592, 1280, 3)
    assert frame.shape[0] % 16 == 0
    assert np.array_equal(frame[:480], base_frame)
    assert np.all(frame[480:] == 255)


def test_should_save_rollout_video_defaults_to_habitat_mode():
    assert video.should_save_rollout_video(SimpleNamespace(save_video=True)) is True


def test_should_save_rollout_video_skips_wrapper_mode():
    assert (
        video.should_save_rollout_video(
            SimpleNamespace(save_video=True, save_mode="wrapper")
        )
        is False
    )


def test_record_rollout_video_frames_appends_numpy_frames():
    render_images = {}
    obs = {
        "main_images": torch.tensor(
            [
                [[[10, 10, 10]]],
                [[[11, 11, 11]]],
            ],
            dtype=torch.uint8,
        )
    }

    video.record_rollout_video_frames(
        render_images,
        obs,
        episode_ids=["10", "11"],
    )

    assert set(render_images) == {"episode_10", "episode_11"}
    assert np.array_equal(render_images["episode_10"][0], np.array([[[10, 10, 10]]]))
    assert np.array_equal(render_images["episode_11"][0], np.array([[[11, 11, 11]]]))


def test_flush_rollout_videos_writes_success_and_failure_dirs(tmp_path, monkeypatch):
    render_images = {
        "episode_10": [np.full((2, 2, 3), 10, dtype=np.uint8)],
        "episode_11": [np.full((2, 2, 3), 11, dtype=np.uint8)],
    }
    calls = []

    monkeypatch.setattr(
        video,
        "save_rollout_video",
        lambda rollout_images, output_dir, video_name, fps: calls.append(
            (rollout_images, output_dir, video_name, fps)
        ),
    )

    video.flush_rollout_videos(
        render_images,
        np.array([True, True]),
        {"episode": {"success": torch.tensor([1.0, 0.0])}},
        episode_ids=["10", "11"],
        video_cfg=SimpleNamespace(video_base_dir=str(tmp_path), fps=12),
    )

    assert len(calls) == 2
    assert np.array_equal(calls[0][0][0], np.full((2, 2, 3), 10, dtype=np.uint8))
    assert calls[0][1:] == (str(tmp_path / "success"), "episode_10", 12)
    assert np.array_equal(calls[1][0][0], np.full((2, 2, 3), 11, dtype=np.uint8))
    assert calls[1][1:] == (str(tmp_path / "failure"), "episode_11", 12)
    assert render_images == {}
