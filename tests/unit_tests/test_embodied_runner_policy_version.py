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

from rlinf.runners.embodied_runner import EmbodiedRunner


class _RecordingHandle:
    def __init__(self, events: list[str], name: str):
        self.events = events
        self.name = name

    def wait(self):
        self.events.append(f"{self.name}.wait")


class _RecordingWorkerGroup:
    def __init__(self, events: list[str], name: str):
        self.events = events
        self.name = name

    def set_global_step(self, global_step: int):
        self.events.append(f"{self.name}.set({global_step})")
        return _RecordingHandle(self.events, self.name)


def _make_runner(curriculum_enabled: bool, events: list[str]) -> EmbodiedRunner:
    runner = object.__new__(EmbodiedRunner)
    runner.curriculum_enabled = curriculum_enabled
    runner.actor = _RecordingWorkerGroup(events, "actor")
    runner.rollout = _RecordingWorkerGroup(events, "rollout")
    return runner


def test_curriculum_policy_version_update_waits_for_both_worker_groups():
    events = []
    runner = _make_runner(curriculum_enabled=True, events=events)

    runner._set_worker_global_step(5)

    assert events == [
        "actor.set(5)",
        "rollout.set(5)",
        "actor.wait",
        "rollout.wait",
    ]


def test_default_policy_version_update_remains_nonblocking():
    events = []
    runner = _make_runner(curriculum_enabled=False, events=events)

    runner._set_worker_global_step(5)

    assert events == ["actor.set(5)", "rollout.set(5)"]
