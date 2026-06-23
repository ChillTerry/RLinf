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

import importlib.util
from pathlib import Path


def load_create_tiny_module():
    script_path = Path(".vscode/create_tiny_val_unseen.py")
    spec = importlib.util.spec_from_file_location(
        "create_tiny_val_unseen", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_episode(episode_id: int, scene_id: str, trajectory_id: int) -> dict:
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "trajectory_id": trajectory_id,
    }


def test_build_subset_keeps_instruction_vocab_when_source_has_it():
    module = load_create_tiny_module()
    main_data = {
        "episodes": [
            make_episode(1, "scene-a", 10),
            make_episode(2, "scene-a", 11),
        ],
        "instruction_vocab": {"word_list": ["go"]},
    }
    gt_data = {"1": {"locations": []}, "2": {"locations": []}}

    subset_main_data, subset_gt_data, _, _ = module.build_subset(
        main_data,
        gt_data,
        num_trajectories_per_scene=1,
        seed=0,
    )

    assert subset_main_data["instruction_vocab"] == main_data["instruction_vocab"]
    assert set(subset_gt_data).issubset({"1", "2"})


def test_build_subset_allows_rxr_data_without_instruction_vocab():
    module = load_create_tiny_module()
    main_data = {
        "episodes": [
            make_episode(1, "scene-a", 10),
            make_episode(2, "scene-a", 11),
        ]
    }
    gt_data = {"1": {"locations": []}, "2": {"locations": []}}

    subset_main_data, subset_gt_data, _, _ = module.build_subset(
        main_data,
        gt_data,
        num_trajectories_per_scene=1,
        seed=0,
    )

    assert "instruction_vocab" not in subset_main_data
    assert set(subset_gt_data).issubset({"1", "2"})
