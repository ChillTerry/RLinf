# Copyright 2025 The RLinf Authors.
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

# This file contains code copied from the VLN-CE project: https://github.com/jacobkrantz/VLN-CE.
# The original code is licensed under the MIT License.

import gzip
import json
import os
from typing import Any, Union

import numpy as np
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import DistanceToGoal
from numpy import ndarray


def euclidean_distance(
    pos_a: Union[list[float], ndarray], pos_b: Union[list[float], ndarray]
) -> float:
    return np.linalg.norm(np.array(pos_b) - np.array(pos_a), ord=2)


def _format_ndtw_gt_path(config: Any) -> str:
    gt_path = str(config.GT_PATH)
    split = str(config.SPLIT)
    return gt_path.format(split=split)


def _load_ndtw_gt_json(config: Any) -> dict[str, Any]:
    gt_path = _format_ndtw_gt_path(config)
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"NDTW GT_PATH does not exist: {gt_path}")

    with gzip.open(gt_path, "rt", encoding="utf-8") as f:
        gt_json = json.load(f)

    if not isinstance(gt_json, dict):
        raise ValueError(f"NDTW GT_PATH must contain a JSON object: {gt_path}")
    return gt_json


def _validate_ndtw_locations(episode_id: str, gt_entry: Any) -> list[list[float]]:
    if not isinstance(gt_entry, dict) or "locations" not in gt_entry:
        raise ValueError(
            f"Episode {episode_id} NDTW ground-truth must contain `locations`."
        )

    locations = gt_entry["locations"]
    if not isinstance(locations, list) or len(locations) == 0:
        raise ValueError(f"Episode {episode_id} NDTW ground-truth locations are empty.")

    validated = []
    for idx, position in enumerate(locations):
        try:
            position_array = np.asarray(position, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Episode {episode_id} NDTW location {idx} must be numeric."
            ) from exc
        if position_array.shape != (3,):
            raise ValueError(
                f"Episode {episode_id} NDTW location {idx} must be a 3D position."
            )
        validated.append(position_array.tolist())
    return validated


def _dtw_distance(path: list[list[float]], gt_path: list[list[float]]) -> float:
    costs = np.full(len(gt_path) + 1, np.inf, dtype=np.float64)
    costs[0] = 0.0

    for path_position in path:
        next_costs = np.full(len(gt_path) + 1, np.inf, dtype=np.float64)
        for gt_idx, gt_position in enumerate(gt_path, start=1):
            step_cost = euclidean_distance(path_position, gt_position)
            next_costs[gt_idx] = step_cost + min(
                next_costs[gt_idx - 1], costs[gt_idx], costs[gt_idx - 1]
            )
        costs = next_costs

    return float(costs[len(gt_path)])


@registry.register_measure
class TrajectoryLength(Measure):
    """Trajectory Length (PL)
    TL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    cls_uuid: str = "trajectory_Length"

    def __init__(self, sim: Simulator, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position
        self._metric = 0.0

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position
        self._metric += euclidean_distance(current_position, self._previous_position)
        self._previous_position = current_position


@registry.register_measure
class NDTW(Measure):
    """Normalized Dynamic Time Warping for Habitat VLN trajectories."""

    cls_uuid: str = "ndtw"

    def __init__(self, *args: Any, sim: Simulator, config: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._fastdtw = None
        if bool(getattr(config, "FDTW", False)):
            try:
                from fastdtw import fastdtw
            except ImportError as exc:
                raise ImportError(
                    "NDTW with FDTW=True requires the optional `fastdtw` package. "
                    "Set FDTW=False to use exact DTW without extra dependencies."
                ) from exc
            self._fastdtw = fastdtw

        self.gt_json = _load_ndtw_gt_json(config)
        self.locations: list[list[float]] = []
        self.gt_locations: list[list[float]] = []
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, **kwargs: Any):
        episode_id = str(episode.episode_id)
        if episode_id not in self.gt_json:
            raise KeyError(f"Episode {episode_id} missing NDTW ground-truth.")
        self.locations = []
        self.gt_locations = _validate_ndtw_locations(
            episode_id, self.gt_json[episode_id]
        )
        self.update_metric()

    def update_metric(self, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) > 0 and current_position == self.locations[-1]:
            return

        self.locations.append(current_position)
        if self._fastdtw is None:
            dtw_distance = _dtw_distance(self.locations, self.gt_locations)
        else:
            dtw_distance = self._fastdtw(
                self.locations, self.gt_locations, dist=euclidean_distance
            )[0]

        self._metric = float(
            np.exp(
                -dtw_distance
                / (len(self.gt_locations) * float(self._config.SUCCESS_DISTANCE))
            )
        )


@registry.register_measure
class OracleNavigationError(Measure):
    """Oracle Navigation Error (ONE)
    ONE = min(geosdesic_distance(agent_pos, goal)) over all points in the
    agent path.
    """

    cls_uuid: str = "oracle_navigation_error"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = float("inf")
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        self._metric = min(self._metric, distance_to_target)


@registry.register_measure
class OracleSuccess(Measure):
    """Oracle Success Rate (OSR). OSR = I(ONE <= goal_radius)"""

    cls_uuid: str = "oracle_success"

    def __init__(self, *args: Any, config: Any, **kwargs: Any):
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToGoal.cls_uuid]
        )
        self._metric = 0.0
        self.update_metric(task=task)

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        d = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        success_distance = getattr(self._config, "success_distance", 3.0)
        self._metric = float(self._metric or d < success_distance)


@registry.register_measure
class OracleSPL(Measure):
    """OracleSPL (Oracle Success weighted by Path Length)
    OracleSPL = max(SPL) over all points in the agent path.
    """

    cls_uuid: str = "oracle_spl"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(self.uuid, ["spl"])
        self._metric = 0.0

    def update_metric(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        spl = task.measurements.measures["spl"].get_metric()
        self._metric = max(self._metric, spl)


def pass_format_check():
    pass
