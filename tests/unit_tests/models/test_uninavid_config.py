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

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_uninavid_stage_one_config_allows_resume_override():
    config_dir = str(Path("examples/sft/config").resolve())
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="uninavid_stage_1_sft",
            overrides=["runner.resume_dir=/tmp/uninavid-step-1"],
        )

    assert cfg.runner.resume_dir == "/tmp/uninavid-step-1"
