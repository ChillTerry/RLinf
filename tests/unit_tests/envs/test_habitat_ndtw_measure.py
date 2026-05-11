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
import math
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.habitat.extensions.measures import NDTW


class AgentStateStub:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float32)


class SimStub:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=np.float32)

    def get_agent_state(self):
        return AgentStateStub(self.position)


def _write_gt(tmp_path, payload):
    gt_path = tmp_path / "train_gt.json.gz"
    with gzip.open(gt_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f)
    return gt_path


def _config(gt_path, *, fdtw=False, success_distance=3.0):
    return SimpleNamespace(
        SPLIT="train",
        GT_PATH=str(gt_path),
        SUCCESS_DISTANCE=success_distance,
        FDTW=fdtw,
    )


def test_ndtw_measure_tracks_unique_positions_and_stays_bounded(tmp_path):
    gt_path = _write_gt(
        tmp_path,
        {"episode-1": {"locations": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]}},
    )
    sim = SimStub([0.0, 0.0, 0.0])
    measure = NDTW(sim=sim, config=_config(gt_path))

    measure.reset_metric(episode=SimpleNamespace(episode_id="episode-1"))
    first_metric = measure.get_metric()
    first_path_len = len(measure.locations)

    sim.position = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    measure.update_metric()

    assert len(measure.locations) == first_path_len
    assert measure.get_metric() == first_metric

    sim.position = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    measure.update_metric()

    assert len(measure.locations) == 2
    assert math.isclose(measure.get_metric(), 1.0, rel_tol=1e-6)
    assert 0.0 <= measure.get_metric() <= 1.0


def test_ndtw_measure_fails_when_gt_path_is_missing(tmp_path):
    missing_gt_path = tmp_path / "missing_gt.json.gz"

    with pytest.raises(FileNotFoundError, match="NDTW GT_PATH does not exist"):
        NDTW(sim=SimStub([0.0, 0.0, 0.0]), config=_config(missing_gt_path))


def test_ndtw_measure_fails_when_episode_id_is_missing(tmp_path):
    gt_path = _write_gt(tmp_path, {})
    measure = NDTW(sim=SimStub([0.0, 0.0, 0.0]), config=_config(gt_path))

    with pytest.raises(KeyError, match="missing NDTW ground-truth"):
        measure.reset_metric(episode=SimpleNamespace(episode_id="episode-404"))


def test_ndtw_measure_fails_when_gt_path_has_non_finite_coordinate(tmp_path):
    gt_path = _write_gt(
        tmp_path,
        {"episode-1": {"locations": [[0.0, float("nan"), 0.0]]}},
    )
    measure = NDTW(sim=SimStub([0.0, 0.0, 0.0]), config=_config(gt_path))

    with pytest.raises(ValueError, match="finite"):
        measure.reset_metric(episode=SimpleNamespace(episode_id="episode-1"))


@pytest.mark.parametrize("success_distance", [0.0, -1.0])
def test_ndtw_measure_fails_when_success_distance_is_not_positive(
    tmp_path, success_distance
):
    gt_path = _write_gt(
        tmp_path,
        {"episode-1": {"locations": [[0.0, 0.0, 0.0]]}},
    )

    with pytest.raises(ValueError, match="SUCCESS_DISTANCE"):
        NDTW(
            sim=SimStub([0.0, 0.0, 0.0]),
            config=_config(gt_path, success_distance=success_distance),
        )


def test_ndtw_measure_matches_hand_computed_dtw_value(tmp_path):
    gt_path = _write_gt(
        tmp_path,
        {"episode-1": {"locations": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]}},
    )
    sim = SimStub([0.0, 0.0, 0.0])
    measure = NDTW(sim=sim, config=_config(gt_path, success_distance=2.0))

    measure.reset_metric(episode=SimpleNamespace(episode_id="episode-1"))
    sim.position = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    measure.update_metric()

    assert math.isclose(measure.get_metric(), math.exp(-0.25), rel_tol=1e-6)


@pytest.mark.parametrize(
    "payload, match",
    [
        ({"episode-1": {"locations": []}}, "empty"),
        ({"episode-1": {"locations": [[0.0, 0.0]]}}, "3D"),
        ({"episode-1": {"locations": [["x", 0.0, 0.0]]}}, "numeric"),
    ],
)
def test_ndtw_measure_fails_when_gt_path_is_malformed(tmp_path, payload, match):
    gt_path = _write_gt(tmp_path, payload)
    measure = NDTW(sim=SimStub([0.0, 0.0, 0.0]), config=_config(gt_path))

    with pytest.raises(ValueError, match=match):
        measure.reset_metric(episode=SimpleNamespace(episode_id="episode-1"))
