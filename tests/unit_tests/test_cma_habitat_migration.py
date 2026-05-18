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

"""Tests for CMA Habitat migration contracts."""

from pathlib import Path

import numpy as np

from rlinf.config import SupportedModel
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import _MODEL_REGISTRY


def test_cma_policy_is_registered_in_model_registry():
    assert SupportedModel.CMA_POLICY.value == "cma"
    assert SupportedModel.CMA_POLICY.value in _MODEL_REGISTRY


def test_habitat_action_adapter_squeezes_cma_singleton_action_dim():
    raw_actions = np.array(
        [
            [[0], [1], [2]],
            [[3], [2], [1]],
        ],
        dtype=np.int64,
    )

    prepared_actions = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type="habitat",
        model_type="cma",
        num_action_chunks=3,
        action_dim=1,
    )

    assert prepared_actions.shape == (2, 3)
    np.testing.assert_array_equal(
        prepared_actions,
        np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64),
    )


def test_cma_migration_does_not_depend_on_gt_prefix_or_gt_data_path():
    feature_paths = (
        Path("examples/embodiment/config/model/cma.yaml"),
        Path("examples/embodiment/config/habitat_r2r_ppo_cma.yaml"),
        Path("examples/embodiment/config/habitat_r2r_grpo_cma.yaml"),
        Path("rlinf/models/embodiment/cma/cma_action_model.py"),
    )

    missing_paths = [str(path) for path in feature_paths if not path.exists()]
    assert missing_paths == []

    matches = []
    for path in feature_paths:
        text = path.read_text()
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if (
                "gt_prefix" in lowered
                or "gt prefix" in lowered
                or "gt-prefix" in lowered
                or "gt_data_path" in lowered
            ):
                matches.append(f"{path}:{line_number}: {line.strip()}")

    assert matches == []
