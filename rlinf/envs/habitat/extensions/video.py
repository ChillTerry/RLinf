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

import os
import textwrap

import cv2
import numpy as np
import torch

from rlinf.envs.habitat.extensions.utils import render_topdown_map
from rlinf.envs.utils import save_rollout_video


EPISODE_VIDEO_SAVE_MODE = "episode"
WRAPPER_VIDEO_SAVE_MODE = "wrapper"
VIDEO_SAVE_MODES = {EPISODE_VIDEO_SAVE_MODE, WRAPPER_VIDEO_SAVE_MODE}


def cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_save_mode(video_cfg, default=EPISODE_VIDEO_SAVE_MODE):
    save_mode = str(cfg_get(video_cfg, "save_mode", default)).lower()
    if save_mode not in VIDEO_SAVE_MODES:
        raise ValueError(
            f"Unsupported Habitat video save_mode={save_mode!r}. "
            f"Expected one of {sorted(VIDEO_SAVE_MODES)}."
        )
    return save_mode


def should_save_rollout_video(video_cfg):
    return bool(cfg_get(video_cfg, "save_video", False)) and (
        get_save_mode(video_cfg) == EPISODE_VIDEO_SAVE_MODE
    )


def get_video_cfg(owner):
    return getattr(
        owner,
        "video_cfg",
        getattr(getattr(owner, "cfg", None), "video_cfg", None),
    )


def build_rollout_video_frame(raw_obs, info):
    rgb = raw_obs["rgb"][:, :, :3]
    if not has_topdown_map(info):
        frame = rgb.astype(np.uint8, copy=False)
        return append_instruction_panel(
            frame,
            raw_obs["instruction"].get("text", ""),
        )

    topdown_map = render_topdown_map(info)[:, :, :3]
    image_size = (rgb.shape[1], rgb.shape[0])
    if topdown_map.shape[:2] != rgb.shape[:2]:
        topdown_map = cv2.resize(
            topdown_map,
            dsize=image_size,
            interpolation=cv2.INTER_LINEAR,
        )
    frame = np.concatenate(
        (
            rgb.astype(np.uint8, copy=False),
            topdown_map.astype(np.uint8, copy=False),
        ),
        axis=1,
    )
    return append_instruction_panel(
        frame,
        raw_obs["instruction"].get("text", ""),
    )


def append_instruction_panel(frame, instruction):
    instruction = normalize_instruction_for_video(instruction)
    panel_height = max(64, int(frame.shape[0] * 0.22))
    panel = np.full((panel_height, frame.shape[1], 3), 255, dtype=np.uint8)
    if instruction:
        draw_instruction_text(panel, instruction)
    return pad_frame_height_to_macro_block(np.concatenate((frame, panel), axis=0))


def pad_frame_height_to_macro_block(frame, macro_block_size=16):
    remainder = frame.shape[0] % macro_block_size
    if remainder == 0:
        return frame

    pad_height = macro_block_size - remainder
    padding = np.full(
        (pad_height, frame.shape[1], frame.shape[2]),
        255,
        dtype=frame.dtype,
    )
    return np.concatenate((frame, padding), axis=0)


def normalize_instruction_for_video(instruction):
    instruction = " ".join(str(instruction or "").split())
    if not instruction:
        return ""
    has_upper = any(char.isupper() for char in instruction)
    has_lower = any(char.islower() for char in instruction)
    if has_upper and not has_lower:
        instruction = instruction.lower()
        return instruction[:1].upper() + instruction[1:]
    return instruction


def draw_instruction_text(panel, instruction):
    margin = max(8, panel.shape[1] // 100)
    max_width = panel.shape[1] - 2 * margin
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    for font_scale in (0.55, 0.48, 0.42, 0.36):
        line_height = cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 6
        max_lines = max(1, (panel.shape[0] - 2 * margin) // line_height)
        lines = wrap_instruction_lines(
            instruction,
            max_width=max_width,
            font=font,
            font_scale=font_scale,
            thickness=thickness,
        )
        if len(lines) <= max_lines or font_scale == 0.36:
            lines = lines[:max_lines]
            if len(lines) == max_lines and len(
                lines
            ) < len(
                wrap_instruction_lines(
                    instruction,
                    max_width=max_width,
                    font=font,
                    font_scale=font_scale,
                    thickness=thickness,
                )
            ):
                lines[-1] = ellipsize_line(
                    lines[-1],
                    max_width=max_width,
                    font=font,
                    font_scale=font_scale,
                    thickness=thickness,
                )
            break

    y = margin + line_height
    for line in lines:
        cv2.putText(
            panel,
            line,
            (margin, y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )
        y += line_height


def wrap_instruction_lines(instruction, max_width, font, font_scale, thickness):
    words = instruction.split()
    if not words:
        return []

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        width = cv2.getTextSize(candidate, font, font_scale, thickness)[0][0]
        if width <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
        if cv2.getTextSize(word, font, font_scale, thickness)[0][0] <= max_width:
            current = word
        else:
            split_words = textwrap.wrap(word, width=max(1, len(word) // 2))
            lines.extend(split_words[:-1])
            current = split_words[-1]
    if current:
        lines.append(current)
    return lines


def ellipsize_line(line, max_width, font, font_scale, thickness):
    suffix = "..."
    while (
        line
        and cv2.getTextSize(line + suffix, font, font_scale, thickness)[0][0]
        > max_width
    ):
        line = line[:-1]
    return line + suffix if line else suffix


def has_topdown_map(info):
    if not isinstance(info, dict):
        return False
    return "top_down_map_vlnce" in info or "top_down_map" in info


def record_rollout_video_frames(render_images, obs, episode_ids):
    if not isinstance(obs, dict) or "main_images" not in obs:
        return

    frames = obs["main_images"]
    if torch.is_tensor(frames):
        frames = frames.detach().cpu().numpy()
    for i, episode_id in enumerate(episode_ids):
        video_name = f"episode_{episode_id}"
        if video_name not in render_images:
            render_images[video_name] = []
        render_images[video_name].append(frames[i])


def flush_rollout_videos(render_images, done_mask, infos, episode_ids, video_cfg):
    done_mask = np.asarray(done_mask, dtype=bool)
    if not done_mask.any():
        return

    for env_idx, is_done in enumerate(done_mask):
        if not is_done:
            continue
        video_name = f"episode_{episode_ids[env_idx]}"
        frames = render_images.pop(video_name, None)
        if not frames:
            continue
        outcome_dir = "success" if episode_success(infos, env_idx) else "failure"
        save_rollout_video(
            frames,
            output_dir=os.path.join(video_cfg.video_base_dir, outcome_dir),
            video_name=video_name,
            fps=int(getattr(video_cfg, "fps", 8)),
        )


def episode_success(infos, env_idx):
    if not isinstance(infos, dict):
        return False
    episode = infos.get("episode", {})
    if not isinstance(episode, dict) or "success" not in episode:
        return False
    success = episode["success"]
    if torch.is_tensor(success):
        success = success.detach().cpu().numpy()
    success = np.asarray(success)
    if success.shape == ():
        return bool(success.item())
    return bool(success[env_idx] > 0)
