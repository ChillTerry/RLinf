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

import gzip
import json
import os
from typing import Optional

from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.vln.r2r_vln_dataset import DEFAULT_SCENE_PATH_PREFIX
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNDatasetV1(Dataset):
    episodes: list[VLNEpisode]

    @staticmethod
    def check_config_paths_exist(config) -> bool:
        return os.path.exists(
            config.data_path.format(split=config.split)
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional[object] = None) -> None:
        self.episodes = []
        if config is None:
            return

        dataset_filename = config.data_path.format(split=config.split)
        languages = getattr(config, "LANGUAGES", None)
        with gzip.open(dataset_filename, "rt", encoding="utf-8") as file_obj:
            self.from_json(
                file_obj.read(),
                scenes_dir=config.scenes_dir,
                languages=languages,
            )

        self.episodes = list(
            filter(self.build_content_scenes_filter(config), self.episodes)
        )

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        languages: Optional[list[str]] = None,
    ) -> None:
        deserialized = json.loads(json_str)
        language_set = _filter_set(languages)

        for episode_data in deserialized["episodes"]:
            instruction_data = episode_data["instruction"]
            if language_set is not None:
                language = instruction_data.get("language")
                if language not in language_set:
                    continue

            episode_data = dict(episode_data)
            episode_data["instruction"] = {
                "instruction_text": instruction_data["instruction_text"],
                "instruction_tokens": instruction_data.get("instruction_tokens"),
            }
            episode = VLNEpisode(**episode_data)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            for goal_idx, goal in enumerate(episode.goals):
                episode.goals[goal_idx] = NavigationGoal(**goal)
            self.episodes.append(episode)


def _filter_set(values) -> Optional[set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}
