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

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

UNINAVID_NAV_PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. You have been given a "
    "video of historical observations and an image of the current observation "
    "<image>. Your assigned task is: '{}'. Analyze this series of images to determine "
    "your next four actions. The predicted action should be one of the following: "
    "forward, left, right, or stop."
)

STOP_ACTION_ID = 0
FORWARD_ACTION_ID = 1
LEFT_ACTION_ID = 2
RIGHT_ACTION_ID = 3
NO_OP_ACTION_ID = 4

HABITAT_NAV_ACTION_TO_ID = {
    "stop": STOP_ACTION_ID,
    "forward": FORWARD_ACTION_ID,
    "left": LEFT_ACTION_ID,
    "right": RIGHT_ACTION_ID,
    "no-op": NO_OP_ACTION_ID,
    "no_op": NO_OP_ACTION_ID,
}

_ACTION_PATTERN = re.compile(r"\b(stop|forward|left|right)\b", re.IGNORECASE)
ACTION_TOKEN_IDS = {
    "forward": (6375, 11333),
    "left": (2175, 1563),
    "right": (1492, 1266),
    "stop": (5040, 9847),
}


@dataclass
class UniNaVidNavCache:
    episode_id: int | None = None
    feat_cache: torch.Tensor | None = None
    long_feat_cache: torch.Tensor | None = None
    weight: int = 1
    new_frames: int = 0


def build_navigation_prompt(instruction: str) -> str:
    return UNINAVID_NAV_PROMPT_TEMPLATE.format(instruction)


def empty_rollout_metadata() -> dict[str, Any]:
    return {
        "prev_logprobs": None,
        "prev_values": None,
        "forward_inputs": {},
    }


def get_slot_cache(
    caches: dict[int, UniNaVidNavCache],
    *,
    slot_id: int,
    episode_id: int,
) -> UniNaVidNavCache:
    cache = caches.get(slot_id)
    if cache is None or cache.episode_id != episode_id:
        cache = UniNaVidNavCache(episode_id=episode_id)
        caches[slot_id] = cache
    return cache


def parse_uninavid_actions(
    output_text: str,
    num_action_chunks: int = 4,
) -> torch.Tensor:
    action_ids: list[int] = []
    for match in _ACTION_PATTERN.finditer(output_text):
        action = match.group(1).lower()
        action_ids.append(HABITAT_NAV_ACTION_TO_ID[action])
        if action == "stop" or len(action_ids) == num_action_chunks:
            break

    while len(action_ids) < num_action_chunks:
        action_ids.append(NO_OP_ACTION_ID)

    return torch.tensor(action_ids[:num_action_chunks], dtype=torch.long).view(
        num_action_chunks,
        1,
    )


def parse_uninavid_action_names(output_text: str, num_action_chunks: int) -> list[str]:
    action_names = []
    for match in _ACTION_PATTERN.finditer(output_text):
        action = match.group(1).lower()
        action_names.append(action)
        if action == "stop" or len(action_names) == num_action_chunks:
            break
    return action_names


def build_action_token_mask(
    output_text: str,
    response_ids: list[int] | torch.Tensor,
    num_action_chunks: int,
) -> list[bool]:
    return [
        slot_id >= 0
        for slot_id in build_action_token_slot_ids(
            output_text,
            response_ids,
            num_action_chunks,
        )
    ]


def build_action_token_slot_ids(
    output_text: str,
    response_ids: list[int] | torch.Tensor,
    num_action_chunks: int,
) -> list[int]:
    action_names = parse_uninavid_action_names(output_text, num_action_chunks)
    slot_ids = [-1] * len(response_ids)
    search_start = 0
    response_id_list = (
        response_ids.detach().cpu().tolist()
        if isinstance(response_ids, torch.Tensor)
        else list(response_ids)
    )

    for action_slot, action in enumerate(action_names):
        token_ids = ACTION_TOKEN_IDS[action]
        for index in range(search_start, len(response_id_list)):
            if response_id_list[index] in token_ids:
                slot_ids[index] = action_slot
                search_start = index + 1
                break

    return slot_ids


def count_parsed_action_chars(output_text: str, num_action_chunks: int) -> int:
    action_chars = 0
    for action in parse_uninavid_action_names(output_text, num_action_chunks):
        action_chars += sum(ch.isalpha() for ch in action)
    return action_chars


def count_response_alpha_chars(output_text: str) -> int:
    return sum(ch.isalpha() for ch in output_text)


def tensor_to_rgb_numpy(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def select_slot_rgb_frames(env_obs: dict[str, Any], slot_id: int) -> list[np.ndarray]:
    slot_images = env_obs["rgb_frame_history"][slot_id]
    history_length = env_obs["rgb_frame_history_lengths"][slot_id]
    if isinstance(history_length, torch.Tensor):
        history_length = int(history_length.detach().cpu().item())
    else:
        history_length = int(history_length)
    return [tensor_to_rgb_numpy(frame) for frame in slot_images[:history_length]]


def episode_id_from_obs(env_obs: dict[str, Any], slot_id: int) -> int:
    states = env_obs["states"]
    if isinstance(states, torch.Tensor):
        return int(states[slot_id].detach().cpu().item())
    return int(states[slot_id])
