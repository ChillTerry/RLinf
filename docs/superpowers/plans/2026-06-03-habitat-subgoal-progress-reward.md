# Habitat Subgoal Progress Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Habitat R2R stateful subgoal progress reward described in `docs/superpowers/specs/2026-06-03-habitat-subgoal-progress-reward-design.md`.

**Architecture:** Add a pure Python reward tracker under `rlinf/envs/habitat/` so reward state-machine behavior is unit-testable without Habitat simulator. Add a Habitat-specific vector-env RPC that returns current episode goal positions and geodesic distances to every goal. Wire `HabitatEnv` to select either existing terminal weighted reward or the new `subgoal_progress` reward mode based on explicit config.

**Tech Stack:** Python, NumPy, PyTorch, OmegaConf, pytest, Habitat-Lab vector env wrappers.

---

## File Structure

- Create `rlinf/envs/habitat/subgoal_reward.py`
  - Owns `SubgoalRewardConfig` and `SubgoalRewardTracker`.
  - Has no Habitat imports.
  - Computes reward, switches active subgoals, tracks cumulative reward components.

- Create `tests/unit_tests/envs/test_habitat_subgoal_reward.py`
  - Pure unit tests for reward math and state transitions.

- Modify `rlinf/envs/habitat/venv.py`
  - Add `HabitatRLEnv.get_current_episode_goal_distances`.
  - Add worker command `get_current_episode_goal_distances`.
  - Add `ReconfigureSubprocEnv.get_current_episode_goal_distances`.

- Create `tests/unit_tests/envs/test_habitat_goal_distance_rpc.py`
  - Unit tests for the new Habitat-specific RPC with stubs.

- Modify `rlinf/envs/habitat/habitat_env.py`
  - Instantiate the reward tracker when `cfg.reward_mode == "subgoal_progress"`.
  - Reset tracker state on env resets.
  - Dispatch reward calculation to terminal weighted reward or subgoal progress reward.
  - Attach reward component diagnostics to `infos["episode"]`.

- Modify `tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py`
  - Keep existing weighted terminal reward tests.
  - Add dispatch/reset tests for subgoal reward mode.
  - Update UniNaVid config assertions.

- Modify `examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml`
  - Add explicit subgoal reward coefficients.
  - Point train split to `r2r_train_with_subgoals.json`.
  - Disable reward filtering for this first dense-reward experiment.

---

### Task 1: Add Pure Subgoal Reward Tests

**Files:**
- Create: `tests/unit_tests/envs/test_habitat_subgoal_reward.py`
- Test target to be created in Task 2: `rlinf/envs/habitat/subgoal_reward.py`

- [ ] **Step 1: Write failing tests for normalized progress, success bonus, and switching**

Create `tests/unit_tests/envs/test_habitat_subgoal_reward.py` with this content:

```python
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

import math

import numpy as np

from rlinf.envs.habitat.subgoal_reward import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)


def _tracker(num_envs=1, **kwargs):
    config = SubgoalRewardConfig(
        progress_reward_coef=kwargs.get("progress_reward_coef", 1.0),
        subgoal_success_reward_coef=kwargs.get("subgoal_success_reward_coef", 6.0),
        subgoal_switch_distance=kwargs.get("subgoal_switch_distance", 1.0),
        subgoal_success_distance=kwargs.get("subgoal_success_distance", 0.5),
        stop_success_reward_coef=kwargs.get("stop_success_reward_coef", 10.0),
        final_success_distance=kwargs.get("final_success_distance", 3.0),
        premature_stop_coeff=kwargs.get("premature_stop_coeff", 4.0),
        stall_patience=kwargs.get("stall_patience", 3),
        stall_penalty_coeff=kwargs.get("stall_penalty_coeff", 1.0),
    )
    return SubgoalRewardTracker(num_envs=num_envs, config=config)


def test_progress_is_normalized_by_active_subgoal_initial_distance():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 8.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[3.0, 7.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(reward[0], 0.25, rel_tol=1e-6)
    assert math.isclose(components["r_progress"][0], 0.25, rel_tol=1e-6)
    assert components["normalized_progress"][0] == 0.25
    assert tracker.active_subgoal_index.tolist() == [0]


def test_progress_is_clipped_to_unit_scale():
    tracker = _tracker()
    tracker.reset([0], [[2.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert reward[0] == 7.0
    assert components["r_progress"][0] == 1.0
    assert components["r_subgoal_success"][0] == 6.0
    assert components["normalized_progress"][0] == 1.0


def test_subgoal_success_bonus_is_normalized_by_subgoal_count():
    tracker = _tracker(subgoal_success_reward_coef=6.0)
    tracker.reset([0], [[3.0, 4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.4, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(components["r_subgoal_success"][0], 2.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 2.0, rel_tol=1e-6)
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert math.isclose(tracker.initial_distance_to_active_subgoal[0], 2.0, rel_tol=1e-6)
    assert reward[0] > 2.0


def test_switch_without_precision_bonus_permanently_loses_that_bonus():
    tracker = _tracker(subgoal_success_reward_coef=6.0)
    tracker.reset([0], [[3.0, 4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.75, 2.0, 4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [1]
    assert tracker.completed_subgoal_count.tolist() == [1]
    assert tracker.cumulative_subgoal_success[0] == 0.0
    assert reward[0] > 0.0

    _, second_components = tracker.compute_step(
        distances_to_subgoals=[[0.2, 0.4, 3.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert math.isclose(second_components["r_subgoal_success"][0], 2.0, rel_tol=1e-6)
    assert math.isclose(tracker.cumulative_subgoal_success[0], 2.0, rel_tol=1e-6)


def test_stall_penalty_starts_after_patience_threshold():
    tracker = _tracker(stall_patience=2, stall_penalty_coeff=1.5)
    tracker.reset([0], [[4.0]])

    first_reward, first_components = tracker.compute_step(
        distances_to_subgoals=[[4.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_subgoals=[[4.5]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_penalty"][0] == 0.0
    assert second_components["r_penalty"][0] == -1.5
    assert first_reward[0] == 0.0
    assert second_reward[0] < 0.0


def test_premature_stop_is_penalized_before_all_subgoals_finish():
    tracker = _tracker(premature_stop_coeff=4.0)
    tracker.reset([0], [[3.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[2.5, 4.5]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert components["r_stop"][0] == -4.0
    assert reward[0] < 0.0
    assert tracker.active_subgoal_index.tolist() == [0]


def test_final_stop_success_is_scaled_by_final_distance_after_all_subgoals_finish():
    tracker = _tracker(stop_success_reward_coef=10.0, final_success_distance=3.0)
    tracker.reset([0], [[2.0]])

    first_reward, first_components = tracker.compute_step(
        distances_to_subgoals=[[0.4]],
        is_stop=np.array([False]),
        valid_mask=np.array([True]),
    )
    second_reward, second_components = tracker.compute_step(
        distances_to_subgoals=[[0.75]],
        is_stop=np.array([True]),
        valid_mask=np.array([True]),
    )

    assert first_components["r_subgoal_success"][0] == 6.0
    assert tracker.all_subgoals_finished.tolist() == [True]
    assert math.isclose(second_components["r_stop"][0], 7.5, rel_tol=1e-6)
    assert second_reward[0] == 7.5


def test_invalid_mask_keeps_reward_and_state_unchanged():
    tracker = _tracker()
    tracker.reset([0], [[4.0, 5.0]])

    reward, components = tracker.compute_step(
        distances_to_subgoals=[[0.2, 1.0]],
        is_stop=np.array([False]),
        valid_mask=np.array([False]),
    )

    assert reward[0] == 0.0
    assert components["r_progress"][0] == 0.0
    assert components["r_subgoal_success"][0] == 0.0
    assert tracker.active_subgoal_index.tolist() == [0]
    assert tracker.completed_subgoal_count.tolist() == [0]
```

- [ ] **Step 2: Run tests and verify import failure**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_subgoal_reward.py -v
```

Expected: FAIL because `rlinf.envs.habitat.subgoal_reward` does not exist.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/unit_tests/envs/test_habitat_subgoal_reward.py
git commit -m "test: cover habitat subgoal reward tracker"
```

---

### Task 2: Implement Pure Subgoal Reward Tracker

**Files:**
- Create: `rlinf/envs/habitat/subgoal_reward.py`
- Test: `tests/unit_tests/envs/test_habitat_subgoal_reward.py`

- [ ] **Step 1: Create the reward tracker module**

Create `rlinf/envs/habitat/subgoal_reward.py` with this content:

```python
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

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubgoalRewardConfig:
    progress_reward_coef: float
    subgoal_success_reward_coef: float
    subgoal_switch_distance: float
    subgoal_success_distance: float
    stop_success_reward_coef: float
    final_success_distance: float
    premature_stop_coeff: float
    stall_patience: int
    stall_penalty_coeff: float
    distance_epsilon: float = 1.0e-6


class SubgoalRewardTracker:
    def __init__(self, num_envs: int, config: SubgoalRewardConfig):
        self.num_envs = int(num_envs)
        self.config = config
        self.active_subgoal_index = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_subgoal_count = np.zeros(self.num_envs, dtype=np.int32)
        self.num_subgoals = np.ones(self.num_envs, dtype=np.int32)
        self.previous_distance_to_active_subgoal = np.zeros(
            self.num_envs, dtype=np.float32
        )
        self.initial_distance_to_active_subgoal = np.ones(
            self.num_envs, dtype=np.float32
        )
        self.non_positive_progress_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.cumulative_progress = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_subgoal_success = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_penalty = np.zeros(self.num_envs, dtype=np.float32)
        self.cumulative_stop = np.zeros(self.num_envs, dtype=np.float32)
        self.subgoal_success_given = [[False] for _ in range(self.num_envs)]
        self.all_subgoals_finished = np.zeros(self.num_envs, dtype=bool)

    def reset(self, env_indices, distances_to_subgoals):
        for env_idx, distances in zip(env_indices, distances_to_subgoals):
            env_idx = int(env_idx)
            distances = self._validate_distances(distances)
            self.active_subgoal_index[env_idx] = 0
            self.completed_subgoal_count[env_idx] = 0
            self.num_subgoals[env_idx] = len(distances)
            self.previous_distance_to_active_subgoal[env_idx] = distances[0]
            self.initial_distance_to_active_subgoal[env_idx] = max(
                distances[0], self.config.distance_epsilon
            )
            self.non_positive_progress_steps[env_idx] = 0
            self.cumulative_progress[env_idx] = 0.0
            self.cumulative_subgoal_success[env_idx] = 0.0
            self.cumulative_penalty[env_idx] = 0.0
            self.cumulative_stop[env_idx] = 0.0
            self.subgoal_success_given[env_idx] = [False] * len(distances)
            self.all_subgoals_finished[env_idx] = False

    def compute_step(self, distances_to_subgoals, is_stop, valid_mask):
        is_stop = np.asarray(is_stop, dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        components = self._empty_components()

        for env_idx, distances in enumerate(distances_to_subgoals):
            if not valid_mask[env_idx]:
                continue

            distances = self._validate_distances(distances)
            active_idx = int(self.active_subgoal_index[env_idx])
            final_distance = float(distances[-1])

            if self.all_subgoals_finished[env_idx]:
                stop_reward = self._stop_reward(env_idx, final_distance, is_stop[env_idx])
                components["r_stop"][env_idx] = stop_reward
                reward[env_idx] = stop_reward
                self.cumulative_stop[env_idx] += stop_reward
                self._write_state_components(components, env_idx, 0.0, final_distance)
                continue

            active_distance = float(distances[active_idx])
            progress_delta = (
                float(self.previous_distance_to_active_subgoal[env_idx])
                - active_distance
            )
            denominator = max(
                float(self.initial_distance_to_active_subgoal[env_idx]),
                self.config.distance_epsilon,
            )
            normalized_progress = float(
                np.clip(progress_delta / denominator, -1.0, 1.0)
            )
            progress_reward = (
                float(self.config.progress_reward_coef) * normalized_progress
            )

            if normalized_progress <= 0.0:
                self.non_positive_progress_steps[env_idx] += 1
            else:
                self.non_positive_progress_steps[env_idx] = 0

            penalty = 0.0
            if self.non_positive_progress_steps[env_idx] >= int(
                self.config.stall_patience
            ):
                penalty = -float(self.config.stall_penalty_coeff)

            subgoal_success = 0.0
            if (
                active_distance <= float(self.config.subgoal_success_distance)
                and not self.subgoal_success_given[env_idx][active_idx]
            ):
                subgoal_success = float(
                    self.config.subgoal_success_reward_coef
                ) / float(self.num_subgoals[env_idx])
                self.subgoal_success_given[env_idx][active_idx] = True

            should_switch = active_distance <= float(self.config.subgoal_switch_distance)
            if should_switch:
                self.completed_subgoal_count[env_idx] += 1
                next_idx = active_idx + 1
                if next_idx >= int(self.num_subgoals[env_idx]):
                    self.all_subgoals_finished[env_idx] = True
                    self.active_subgoal_index[env_idx] = active_idx
                else:
                    self.active_subgoal_index[env_idx] = next_idx
                    next_distance = float(distances[next_idx])
                    self.previous_distance_to_active_subgoal[env_idx] = next_distance
                    self.initial_distance_to_active_subgoal[env_idx] = max(
                        next_distance, self.config.distance_epsilon
                    )
                    self.non_positive_progress_steps[env_idx] = 0
            else:
                self.previous_distance_to_active_subgoal[env_idx] = active_distance

            stop_reward = self._stop_reward(env_idx, final_distance, is_stop[env_idx])

            components["r_progress"][env_idx] = progress_reward
            components["r_subgoal_success"][env_idx] = subgoal_success
            components["r_penalty"][env_idx] = penalty
            components["r_stop"][env_idx] = stop_reward
            self._write_state_components(
                components,
                env_idx,
                normalized_progress,
                active_distance,
            )

            reward[env_idx] = (
                progress_reward + subgoal_success + penalty + stop_reward
            )
            self.cumulative_progress[env_idx] += progress_reward
            self.cumulative_subgoal_success[env_idx] += subgoal_success
            self.cumulative_penalty[env_idx] += penalty
            self.cumulative_stop[env_idx] += stop_reward

        return reward, components

    def _stop_reward(self, env_idx: int, final_distance: float, is_stop: bool) -> float:
        if not is_stop:
            return 0.0
        if not self.all_subgoals_finished[env_idx]:
            return -float(self.config.premature_stop_coeff)
        success_scale = 1.0 - min(
            final_distance / float(self.config.final_success_distance), 1.0
        )
        return float(self.config.stop_success_reward_coef) * success_scale

    def _empty_components(self):
        return {
            "r_progress": np.zeros(self.num_envs, dtype=np.float32),
            "r_subgoal_success": np.zeros(self.num_envs, dtype=np.float32),
            "r_penalty": np.zeros(self.num_envs, dtype=np.float32),
            "r_stop": np.zeros(self.num_envs, dtype=np.float32),
            "active_subgoal_index": self.active_subgoal_index.astype(np.float32).copy(),
            "completed_subgoal_count": self.completed_subgoal_count.astype(
                np.float32
            ).copy(),
            "distance_to_active_subgoal": np.zeros(self.num_envs, dtype=np.float32),
            "normalized_progress": np.zeros(self.num_envs, dtype=np.float32),
            "cumulative_progress": self.cumulative_progress.copy(),
            "cumulative_subgoal_success": self.cumulative_subgoal_success.copy(),
            "cumulative_penalty": self.cumulative_penalty.copy(),
            "cumulative_stop": self.cumulative_stop.copy(),
        }

    def _write_state_components(
        self,
        components,
        env_idx: int,
        normalized_progress: float,
        distance_to_active_subgoal: float,
    ):
        components["active_subgoal_index"][env_idx] = float(
            self.active_subgoal_index[env_idx]
        )
        components["completed_subgoal_count"][env_idx] = float(
            self.completed_subgoal_count[env_idx]
        )
        components["distance_to_active_subgoal"][env_idx] = float(
            distance_to_active_subgoal
        )
        components["normalized_progress"][env_idx] = float(normalized_progress)
        components["cumulative_progress"][env_idx] = float(
            self.cumulative_progress[env_idx]
        )
        components["cumulative_subgoal_success"][env_idx] = float(
            self.cumulative_subgoal_success[env_idx]
        )
        components["cumulative_penalty"][env_idx] = float(
            self.cumulative_penalty[env_idx]
        )
        components["cumulative_stop"][env_idx] = float(self.cumulative_stop[env_idx])

    @staticmethod
    def _validate_distances(distances):
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim != 1 or len(distances) == 0:
            raise ValueError("Subgoal distances must be a non-empty 1D sequence.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("Subgoal distances must be finite.")
        return distances
```

- [ ] **Step 2: Run pure reward tests**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_subgoal_reward.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit reward tracker**

```bash
git add rlinf/envs/habitat/subgoal_reward.py
git commit -m "feat: add habitat subgoal reward tracker"
```

---

### Task 3: Add Habitat Goal Distance RPC

**Files:**
- Modify: `rlinf/envs/habitat/venv.py`
- Create: `tests/unit_tests/envs/test_habitat_goal_distance_rpc.py`

- [ ] **Step 1: Write failing RPC tests**

Create `tests/unit_tests/envs/test_habitat_goal_distance_rpc.py` with this content:

```python
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

from rlinf.envs.habitat.venv import HabitatRLEnv, ReconfigureSubprocEnv


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
    env.habitat_env = SimpleNamespace(
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
```

- [ ] **Step 2: Run RPC tests and verify failure**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_goal_distance_rpc.py -v
```

Expected: FAIL because the new methods and worker command do not exist.

- [ ] **Step 3: Add `HabitatRLEnv.get_current_episode_goal_distances`**

In `rlinf/envs/habitat/venv.py`, add this method to `HabitatRLEnv`:

```python
    def get_current_episode_goal_distances(self):
        episode = self.habitat_env.current_episode
        sim = self.habitat_env.sim
        agent_position = sim.get_agent_state().position
        agent_position_list = np.asarray(agent_position, dtype=np.float32).tolist()
        goals = [
            np.asarray(goal.position, dtype=np.float32).tolist()
            for goal in episode.goals
        ]
        distances_to_goals = [
            float(sim.geodesic_distance(agent_position, goal_position))
            for goal_position in goals
        ]
        return {
            "episode_id": getattr(episode, "episode_id", None),
            "agent_position": agent_position_list,
            "goals": goals,
            "distances_to_goals": distances_to_goals,
        }
```

- [ ] **Step 4: Add the worker command**

In `_worker` in `rlinf/envs/habitat/venv.py`, add this command branch after `get_current_metrics`:

```python
            elif cmd == "get_current_episode_goal_distances":
                if hasattr(env, "get_current_episode_goal_distances"):
                    p.send(env.get_current_episode_goal_distances())
                else:
                    p.send({})
```

- [ ] **Step 5: Add vector aggregation**

In `ReconfigureSubprocEnv`, add this method:

```python
    def get_current_episode_goal_distances(self, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        goal_metadata: dict[str, list[Any]] = {}
        for i in id:
            self.workers[i].parent_remote.send(
                ["get_current_episode_goal_distances", None]
            )
            worker_metadata = self.workers[i].parent_remote.recv()
            for key, value in worker_metadata.items():
                if key not in goal_metadata:
                    goal_metadata[key] = []
                goal_metadata[key].append(value)

        return goal_metadata
```

- [ ] **Step 6: Run RPC tests**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_goal_distance_rpc.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit RPC changes**

```bash
git add rlinf/envs/habitat/venv.py tests/unit_tests/envs/test_habitat_goal_distance_rpc.py
git commit -m "feat: expose habitat subgoal geodesic distances"
```

---

### Task 4: Wire Subgoal Reward Into HabitatEnv

**Files:**
- Modify: `rlinf/envs/habitat/habitat_env.py`
- Modify: `tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py`

- [ ] **Step 1: Update imports**

In `rlinf/envs/habitat/habitat_env.py`, add:

```python
from rlinf.envs.habitat.subgoal_reward import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)
```

- [ ] **Step 2: Add config and tracker initialization**

In `HabitatEnv.__init__`, after `self.initial_distance_to_goal` is initialized, add:

```python
        self.reward_mode = getattr(self.cfg, "reward_mode", "weighted_success_ndtw")
        self.subgoal_reward = None
        if self.reward_mode == "subgoal_progress":
            self.subgoal_reward = SubgoalRewardTracker(
                num_envs=self.num_envs,
                config=SubgoalRewardConfig(
                    progress_reward_coef=float(self.cfg.progress_reward_coef),
                    subgoal_success_reward_coef=float(
                        self.cfg.subgoal_success_reward_coef
                    ),
                    subgoal_switch_distance=float(self.cfg.subgoal_switch_distance),
                    subgoal_success_distance=float(self.cfg.subgoal_success_distance),
                    stop_success_reward_coef=float(self.cfg.stop_success_reward_coef),
                    final_success_distance=float(self.cfg.final_success_distance),
                    premature_stop_coeff=float(self.cfg.premature_stop_coeff),
                    stall_patience=int(self.cfg.stall_patience),
                    stall_penalty_coeff=float(self.cfg.stall_penalty_coeff),
                ),
            )
```

- [ ] **Step 3: Add reset helper**

Add this method to `HabitatEnv`:

```python
    def _reset_subgoal_reward_state(self, env_idx):
        if self.subgoal_reward is None:
            return
        metadata = self.env.get_current_episode_goal_distances(id=env_idx)
        self.subgoal_reward.reset(
            env_idx,
            metadata["distances_to_goals"],
        )
```

- [ ] **Step 4: Call reset helper**

In `HabitatEnv.reset`, after seeding `initial_distance_to_goal`, add:

```python
        self._reset_subgoal_reward_state(env_idx)
```

- [ ] **Step 5: Preserve valid reward mask before metrics update**

In `HabitatEnv.step`, before `_record_metrics`, define:

```python
        valid_reward_mask = ~self.first_done_cached_mask
```

Then change reward calculation to:

```python
        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(infos, terminations, first_done_mask)
        step_reward = self._calc_step_reward(
            infos["episode"],
            first_done_reward_mask,
            is_stop=is_stop,
            valid_reward_mask=valid_reward_mask,
            infos=infos,
        )
```

- [ ] **Step 6: Rename terminal reward implementation**

Replace the current `_calc_step_reward` body with a dispatcher:

```python
    def _calc_step_reward(
        self,
        episode,
        first_done_reward_mask,
        is_stop=None,
        valid_reward_mask=None,
        infos=None,
    ):
        if self.reward_mode == "subgoal_progress":
            return self._calc_subgoal_progress_reward(
                is_stop=is_stop,
                valid_reward_mask=valid_reward_mask,
                infos=infos,
            )
        return self._calc_weighted_success_ndtw_reward(
            episode,
            first_done_reward_mask,
        )
```

Add the old terminal reward as a new method:

```python
    def _calc_weighted_success_ndtw_reward(self, episode, first_done_reward_mask):
        device = episode["success"].device
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        first_done_reward_mask = torch.as_tensor(
            first_done_reward_mask,
            dtype=torch.bool,
            device=device,
        )
        if not first_done_reward_mask.any():
            return reward

        success = episode["success"].to(dtype=torch.float32)
        distance_to_goal = episode["distance_to_goal"].to(dtype=torch.float32)
        ndtw = episode["ndtw"].to(dtype=torch.float32)
        success_distance = float(
            self.env_config.task.measurements.success.success_distance
        )

        success_scale = 1.0 - torch.minimum(
            distance_to_goal / success_distance,
            torch.ones_like(distance_to_goal),
        )
        success_reward = success * float(self.cfg.success_reward_coef) * success_scale
        ndtw_reward = ndtw * float(self.cfg.ndtw_reward_coef)
        reward[first_done_reward_mask] = (
            success_reward[first_done_reward_mask] + ndtw_reward[first_done_reward_mask]
        )
        return reward
```

- [ ] **Step 7: Add subgoal reward calculation**

Add this method:

```python
    def _calc_subgoal_progress_reward(self, is_stop, valid_reward_mask, infos):
        if self.subgoal_reward is None:
            raise RuntimeError("subgoal_progress reward mode requires subgoal_reward.")
        if is_stop is None:
            raise RuntimeError("subgoal_progress reward mode requires is_stop.")
        if valid_reward_mask is None:
            raise RuntimeError("subgoal_progress reward mode requires valid_reward_mask.")

        metadata = self.env.get_current_episode_goal_distances()
        reward, components = self.subgoal_reward.compute_step(
            distances_to_subgoals=metadata["distances_to_goals"],
            is_stop=is_stop,
            valid_mask=valid_reward_mask,
        )
        if infos is not None:
            self._attach_subgoal_reward_metrics(infos, components)
        return to_tensor(reward)
```

- [ ] **Step 8: Add reward diagnostics attachment**

Add this method:

```python
    def _attach_subgoal_reward_metrics(self, infos, components):
        episode = infos.setdefault("episode", {})
        for key, value in components.items():
            tensor_value = to_tensor(value)
            episode[key] = tensor_value
            if self.episode_info is not None:
                if key not in self.episode_info:
                    self.episode_info[key] = torch.zeros_like(tensor_value)
                self.episode_info[key][:] = tensor_value
```

- [ ] **Step 9: Write failing dispatch/reset tests**

Append these tests to `tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py`:

```python
def test_habitat_subgoal_reward_reset_initializes_tracker():
    env = object.__new__(HabitatEnv)
    env.num_envs = 2
    env.cfg = SimpleNamespace(
        reward_mode="subgoal_progress",
        progress_reward_coef=1.0,
        subgoal_success_reward_coef=6.0,
        subgoal_switch_distance=1.0,
        subgoal_success_distance=0.5,
        stop_success_reward_coef=10.0,
        final_success_distance=3.0,
        premature_stop_coeff=4.0,
        stall_patience=3,
        stall_penalty_coeff=1.0,
    )
    env.reward_mode = "subgoal_progress"
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=2,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
        ),
    )
    env.env = SimpleNamespace(
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[4.0, 8.0], [3.0]]
        }
    )

    env._reset_subgoal_reward_state(np.array([0, 1]))

    assert env.subgoal_reward.num_subgoals.tolist() == [2, 1]
    assert env.subgoal_reward.previous_distance_to_active_subgoal.tolist() == [4.0, 3.0]


def test_habitat_subgoal_reward_dispatch_uses_dense_reward_and_attaches_metrics():
    env = object.__new__(HabitatEnv)
    env.num_envs = 1
    env.reward_mode = "subgoal_progress"
    env.episode_info = {}
    env.subgoal_reward = SubgoalRewardTracker(
        num_envs=1,
        config=SubgoalRewardConfig(
            progress_reward_coef=1.0,
            subgoal_success_reward_coef=6.0,
            subgoal_switch_distance=1.0,
            subgoal_success_distance=0.5,
            stop_success_reward_coef=10.0,
            final_success_distance=3.0,
            premature_stop_coeff=4.0,
            stall_patience=3,
            stall_penalty_coeff=1.0,
        ),
    )
    env.subgoal_reward.reset([0], [[4.0, 8.0]])
    env.env = SimpleNamespace(
        get_current_episode_goal_distances=lambda id=None: {
            "distances_to_goals": [[3.0, 7.0]]
        }
    )
    infos = {"episode": {}}

    reward = env._calc_step_reward(
        episode={},
        first_done_reward_mask=np.array([False]),
        is_stop=np.array([False]),
        valid_reward_mask=np.array([True]),
        infos=infos,
    )

    assert reward.tolist() == [0.25]
    assert infos["episode"]["r_progress"].tolist() == [0.25]
    assert infos["episode"]["active_subgoal_index"].tolist() == [0.0]


def test_habitat_weighted_reward_dispatch_remains_default():
    env = _make_reward_test_env(num_envs=1)
    env.reward_mode = "weighted_success_ndtw"
    episode = {
        "success": torch.tensor([1.0]),
        "distance_to_goal": torch.tensor([1.5]),
        "ndtw": torch.tensor([0.2]),
    }

    reward = env._calc_step_reward(
        episode,
        first_done_reward_mask=np.array([True]),
    )

    assert reward.tolist() == [6.0]
```

Also add these imports near the top of the test file:

```python
from rlinf.envs.habitat.subgoal_reward import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)
```

Update `_make_reward_test_env` in the same file so existing terminal reward tests still exercise the default reward branch:

```python
def _make_reward_test_env(num_envs):
    env = object.__new__(HabitatEnv)
    env.num_envs = num_envs
    env.reward_mode = "weighted_success_ndtw"
    env.cfg = SimpleNamespace(success_reward_coef=10.0, ndtw_reward_coef=5.0)
    env.env_config = SimpleNamespace(
        task=SimpleNamespace(
            measurements=SimpleNamespace(success=SimpleNamespace(success_distance=3.0))
        )
    )
    return env
```

- [ ] **Step 10: Run HabitatEnv tests**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py -v
```

Expected: PASS.

- [ ] **Step 11: Commit HabitatEnv wiring**

```bash
git add rlinf/envs/habitat/habitat_env.py tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py
git commit -m "feat: wire habitat subgoal progress reward"
```

---

### Task 5: Update UniNaVid Habitat Config

**Files:**
- Modify: `examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml`
- Modify: `tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py`

- [ ] **Step 1: Update config test expectations**

In `test_habitat_grpo_uninavid_uses_weighted_reward_config`, rename the test to:

```python
def test_habitat_grpo_uninavid_uses_subgoal_progress_reward_config():
```

Replace that test body with:

```python
    cfg = OmegaConf.load("examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml")
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)

    assert raw_cfg["algorithm"]["reward_mode"] == "subgoal_progress"
    assert raw_cfg["algorithm"]["progress_reward_coef"] == 1.0
    assert raw_cfg["algorithm"]["subgoal_success_reward_coef"] == 6.0
    assert raw_cfg["algorithm"]["subgoal_switch_distance"] == 1.0
    assert raw_cfg["algorithm"]["subgoal_success_distance"] == 0.5
    assert raw_cfg["algorithm"]["stop_success_reward_coef"] == 10.0
    assert raw_cfg["algorithm"]["final_success_distance"] == 3.0
    assert raw_cfg["algorithm"]["premature_stop_coeff"] == 4.0
    assert raw_cfg["algorithm"]["stall_patience"] == 3
    assert raw_cfg["algorithm"]["stall_penalty_coeff"] == 1.0
    assert raw_cfg["algorithm"]["filter_rewards"] is False

    assert raw_cfg["env"]["train"]["reward_mode"] == "${algorithm.reward_mode}"
    assert raw_cfg["env"]["train"]["progress_reward_coef"] == "${algorithm.progress_reward_coef}"
    assert (
        raw_cfg["env"]["train"]["subgoal_success_reward_coef"]
        == "${algorithm.subgoal_success_reward_coef}"
    )
    assert raw_cfg["env"]["train"]["subgoal_switch_distance"] == "${algorithm.subgoal_switch_distance}"
    assert raw_cfg["env"]["train"]["subgoal_success_distance"] == "${algorithm.subgoal_success_distance}"
    assert raw_cfg["env"]["train"]["stop_success_reward_coef"] == "${algorithm.stop_success_reward_coef}"
    assert raw_cfg["env"]["train"]["final_success_distance"] == "${algorithm.final_success_distance}"
    assert raw_cfg["env"]["train"]["premature_stop_coeff"] == "${algorithm.premature_stop_coeff}"
    assert raw_cfg["env"]["train"]["stall_patience"] == "${algorithm.stall_patience}"
    assert raw_cfg["env"]["train"]["stall_penalty_coeff"] == "${algorithm.stall_penalty_coeff}"
    assert raw_cfg["env"]["train"]["data_path"] == (
        "${env.data_path_dir}/${env.train.split}/r2r_train_with_subgoals.json"
    )

    assert raw_cfg["env"]["eval"]["reward_mode"] == "${algorithm.reward_mode}"
    assert raw_cfg["env"]["eval"]["data_path"] == (
        "${env.data_path_dir}/${env.eval.split}/${env.eval.split}.json.gz"
    )
```

- [ ] **Step 2: Run config test and verify failure**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py::test_habitat_grpo_uninavid_uses_subgoal_progress_reward_config -v
```

Expected: FAIL because the YAML still has terminal reward fields.

- [ ] **Step 3: Update algorithm reward config**

In `examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml`, replace:

```yaml
  success_reward_coef: 15.0
  ndtw_reward_coef: 0.0
  filter_rewards: True
  rewards_lower_bound: 1.0
  rewards_upper_bound: 12.0
```

with:

```yaml
  reward_mode: subgoal_progress
  progress_reward_coef: 1.0
  subgoal_success_reward_coef: 6.0
  subgoal_switch_distance: 1.0
  subgoal_success_distance: 0.5
  stop_success_reward_coef: 10.0
  final_success_distance: 3.0
  premature_stop_coeff: 4.0
  stall_patience: 3
  stall_penalty_coeff: 1.0
  filter_rewards: False
  rewards_lower_bound: -1000000000.0
  rewards_upper_bound: 1000000000.0
```

- [ ] **Step 4: Update train env config**

In `env.train`, replace:

```yaml
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
```

with:

```yaml
    reward_mode: ${algorithm.reward_mode}
    progress_reward_coef: ${algorithm.progress_reward_coef}
    subgoal_success_reward_coef: ${algorithm.subgoal_success_reward_coef}
    subgoal_switch_distance: ${algorithm.subgoal_switch_distance}
    subgoal_success_distance: ${algorithm.subgoal_success_distance}
    stop_success_reward_coef: ${algorithm.stop_success_reward_coef}
    final_success_distance: ${algorithm.final_success_distance}
    premature_stop_coeff: ${algorithm.premature_stop_coeff}
    stall_patience: ${algorithm.stall_patience}
    stall_penalty_coeff: ${algorithm.stall_penalty_coeff}
```

Replace train data path:

```yaml
    data_path: ${env.data_path_dir}/${env.train.split}/${env.train.split}.json.gz
```

with:

```yaml
    data_path: ${env.data_path_dir}/${env.train.split}/r2r_train_with_subgoals.json
```

- [ ] **Step 5: Update eval env config**

In `env.eval`, replace:

```yaml
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
```

with:

```yaml
    reward_mode: ${algorithm.reward_mode}
    progress_reward_coef: ${algorithm.progress_reward_coef}
    subgoal_success_reward_coef: ${algorithm.subgoal_success_reward_coef}
    subgoal_switch_distance: ${algorithm.subgoal_switch_distance}
    subgoal_success_distance: ${algorithm.subgoal_success_distance}
    stop_success_reward_coef: ${algorithm.stop_success_reward_coef}
    final_success_distance: ${algorithm.final_success_distance}
    premature_stop_coeff: ${algorithm.premature_stop_coeff}
    stall_patience: ${algorithm.stall_patience}
    stall_penalty_coeff: ${algorithm.stall_penalty_coeff}
```

Keep eval data path unchanged:

```yaml
    data_path: ${env.data_path_dir}/${env.eval.split}/${env.eval.split}.json.gz
```

- [ ] **Step 6: Run config tests**

Run:

```bash
pytest tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py::test_habitat_grpo_uninavid_uses_subgoal_progress_reward_config -v
```

Expected: PASS.

- [ ] **Step 7: Commit config changes**

```bash
git add examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py
git commit -m "config: enable habitat subgoal progress reward"
```

---

### Task 6: Verification

**Files:**
- No new files.
- Validate all files touched in Tasks 1-5.

- [ ] **Step 1: Run focused reward tests**

Run:

```bash
pytest \
  tests/unit_tests/envs/test_habitat_subgoal_reward.py \
  tests/unit_tests/envs/test_habitat_goal_distance_rpc.py \
  tests/unit_tests/envs/test_habitat_uninavid_chunk_history.py \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run all Habitat unit tests**

Run:

```bash
pytest tests/unit_tests/envs -v
```

Expected: PASS. If Habitat dependencies for config composition are unavailable, stop and report the exact missing dependency instead of installing packages.

- [ ] **Step 3: Inspect final diff**

Run:

```bash
git status --short
git diff --stat HEAD
```

Expected:

- only intended files are modified or already committed
- no edits outside `rlinf/envs/habitat/`, `examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml`, and `tests/unit_tests/envs/`

- [ ] **Step 4: Commit any remaining verification-only corrections**

If Step 1 or Step 2 required small corrections, commit them:

```bash
git add rlinf/envs/habitat tests/unit_tests/envs examples/embodiment/config/habitat_r2r_grpo_uninavid.yaml
git commit -m "test: verify habitat subgoal progress reward"
```

If no corrections were needed, do not create an empty commit.

---

## Self-Review

Spec coverage:

- Active subgoal state is covered by Tasks 1 and 2.
- Explicit geodesic distance to every episode goal is covered by Task 3.
- HabitatEnv reward-mode isolation is covered by Task 4.
- Subgoal switch distance `1.0` and success distance `0.5` are covered by Tasks 1, 2, and 5.
- Subgoal success reward normalization by `num_subgoals` is covered by Tasks 1 and 2.
- Progress normalization and clipping are covered by Tasks 1 and 2.
- Premature stop and final stop scaling are covered by Tasks 1 and 2.
- Single-goal behavior is covered by Task 1 through final stop on a one-goal episode.
- UniNaVid train data path and explicit config fields are covered by Task 5.
- Verification commands are covered by Task 6.

Unresolved-marker scan:

- No unresolved marker strings or unspecified implementation steps remain.

Type consistency:

- `SubgoalRewardConfig` field names match YAML fields.
- `SubgoalRewardTracker.compute_step` receives `distances_to_subgoals`, `is_stop`, and `valid_mask` in both tests and HabitatEnv integration.
- `ReconfigureSubprocEnv.get_current_episode_goal_distances` returns the same `distances_to_goals` key consumed by `HabitatEnv`.
