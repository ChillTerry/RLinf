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

from rlinf.envs.habitat.extensions.filter_reachable_rxr import (
    build_reachable_gt_data,
    discover_main_dataset_files,
    get_gt_path,
    with_reachable_suffix,
)


def test_reachable_rxr_paths_use_expected_suffix(tmp_path):
    main_path = tmp_path / "val_unseen_guide.json.gz"
    gt_path = tmp_path / "val_unseen_guide_gt.json.gz"

    assert with_reachable_suffix(main_path).name == (
        "val_unseen_guide_reachable.json.gz"
    )
    assert with_reachable_suffix(gt_path).name == (
        "val_unseen_guide_gt_reachable.json.gz"
    )
    assert get_gt_path(main_path).name == "val_unseen_guide_gt.json.gz"


def test_reachable_rxr_discovers_only_main_source_files(tmp_path):
    for name in [
        "train_guide.json.gz",
        "train_guide_gt.json.gz",
        "train_guide_reachable.json.gz",
        "train_guide_gt_reachable.json.gz",
    ]:
        (tmp_path / name).touch()

    main_paths = discover_main_dataset_files(tmp_path)

    assert [path.name for path in main_paths] == ["train_guide.json.gz"]


def test_reachable_rxr_gt_is_synchronized_to_kept_episodes():
    gt_data = {
        "1": {"locations": [[0.0, 0.0, 0.0]]},
        "2": {"locations": [[1.0, 0.0, 0.0]]},
    }
    kept_episodes = [{"episode_id": "2"}]

    reachable_gt = build_reachable_gt_data(gt_data, kept_episodes)

    assert reachable_gt == {"2": {"locations": [[1.0, 0.0, 0.0]]}}
