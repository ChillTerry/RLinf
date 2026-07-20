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

import numpy as np
import pytest

from rlinf.envs.habitat.venv import (
    HabitatRLEnv,
    ReconfigureSubprocEnv,
    ReconfigureSubprocEnvWorker,
)


class SimStub:
    def __init__(self):
        self.agent_position = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    def get_agent_state(self):
        return SimpleNamespace(position=self.agent_position)

    def geodesic_distance(self, source, target):
        source = np.asarray(source, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        return float(np.linalg.norm(source - target))


class WorkerStub:
    def __init__(self, payload):
        self.payload = payload
        self.parent_remote = self
        self.sent = []

    def send(self, message):
        self.sent.append(message)

    def recv(self):
        return self.payload


def test_habitat_rl_env_returns_current_episode_goal_distances():
    env = object.__new__(HabitatRLEnv)
    env._env = SimpleNamespace(
        current_episode=SimpleNamespace(
            episode_id="episode-7",
            goals=[
                SimpleNamespace(position=[1.0, 0.0, 3.0]),
                SimpleNamespace(position=[4.0, 0.0, 1.0]),
            ],
        ),
        sim=SimStub(),
    )

    metadata = env.get_current_episode_goal_distances()

    assert metadata["episode_id"] == "episode-7"
    assert metadata["agent_position"] == [1.0, 0.0, 1.0]
    assert metadata["goals"] == [[1.0, 0.0, 3.0], [4.0, 0.0, 1.0]]
    assert metadata["distances_to_goals"] == [2.0, 3.0]


def test_habitat_rl_env_activates_cached_episodes_without_reconstruction():
    episodes = {
        "7": SimpleNamespace(episode_id="7"),
        "19": SimpleNamespace(episode_id="19"),
    }
    env = object.__new__(HabitatRLEnv)
    env._episode_registry = episodes
    env._env = SimpleNamespace(episodes=[episodes["7"]])

    env.activate_episode_ids([19, "7"])

    assert env._env.episodes == [episodes["19"], episodes["7"]]


def test_habitat_rl_env_rejects_invalid_cached_episode_activation():
    env = object.__new__(HabitatRLEnv)
    env._episode_registry = {"7": SimpleNamespace(episode_id="7")}
    env._env = SimpleNamespace(episodes=[])

    with pytest.raises(ValueError, match="at least one ID"):
        env.activate_episode_ids([])
    with pytest.raises(KeyError, match="Unknown Habitat episode IDs"):
        env.activate_episode_ids(["missing"])


def test_reconfigure_subproc_worker_sends_lightweight_episode_activation():
    remote = WorkerStub(None)
    worker = object.__new__(ReconfigureSubprocEnvWorker)
    worker.parent_remote = remote

    worker.activate_episode_ids(["7", "19"])

    assert remote.sent == [["activate_episode_ids", ["7", "19"]]]


def test_reconfigure_subproc_env_aggregates_goal_distance_metadata():
    env = object.__new__(ReconfigureSubprocEnv)
    env.is_async = False
    env.closed = False
    env.workers = [
        WorkerStub(
            {
                "episode_id": "1",
                "goals": [[0.0, 0.0, 0.0]],
                "distances_to_goals": [1.0],
                "agent_position": [1.0, 0.0, 0.0],
            }
        ),
        WorkerStub(
            {
                "episode_id": "2",
                "goals": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                "distances_to_goals": [2.0, 0.5],
                "agent_position": [2.0, 0.0, 0.0],
            }
        ),
    ]
    env._assert_is_not_closed = lambda: None
    env._assert_id = lambda ids: None
    env._wrap_id = lambda ids=None: list(range(len(env.workers))) if ids is None else ids

    metadata = env.get_current_episode_goal_distances()

    assert metadata["episode_id"] == ["1", "2"]
    assert metadata["distances_to_goals"] == [[1.0], [2.0, 0.5]]
    assert env.workers[0].sent == [["get_current_episode_goal_distances", None]]
    assert env.workers[1].sent == [["get_current_episode_goal_distances", None]]
