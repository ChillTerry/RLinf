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

import torch

from rlinf.envs.habitat.habitat_env import HabitatEnv


def test_habitat_attaches_uninavid_chunk_rgb_history():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type="uninavid")
    obs_list = [
        {"wrist_images": torch.full((2, 4, 4, 3), 1, dtype=torch.uint8)},
        {"wrist_images": torch.full((2, 4, 4, 3), 2, dtype=torch.uint8)},
        {"wrist_images": torch.full((2, 4, 4, 3), 3, dtype=torch.uint8)},
    ]

    env._attach_uninavid_chunk_history(obs_list)

    assert "wrist_images_history" in obs_list[-1]
    assert obs_list[-1]["wrist_images_history"].shape == (2, 3, 4, 4, 3)
    assert obs_list[-1]["wrist_images_history"][0, :, 0, 0, 0].tolist() == [1, 2, 3]


def test_habitat_does_not_attach_chunk_history_for_other_models():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(model_type="openvla")
    obs_list = [
        {"wrist_images": torch.full((1, 4, 4, 3), 1, dtype=torch.uint8)},
        {"wrist_images": torch.full((1, 4, 4, 3), 2, dtype=torch.uint8)},
    ]

    env._attach_uninavid_chunk_history(obs_list)

    assert "wrist_images_history" not in obs_list[-1]


def test_habitat_model_type_access_allows_missing_field():
    env = object.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(video_cfg=SimpleNamespace(save_video=False))

    assert env._habitat_model_type() is None
