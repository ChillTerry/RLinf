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

import sys

from rlinf.envs.habitat import gpu_selection
from rlinf.envs.habitat.gpu_selection import (
    GpuSnapshot,
    choose_gpu_ids,
    parse_nvidia_smi_csv,
)


def test_parse_nvidia_smi_csv():
    snapshots = parse_nvidia_smi_csv(
        "0, 40536, 81920, 3\n"
        "1, 21000, 81920, 77\n"
    )

    assert snapshots == [
        GpuSnapshot(gpu_id=0, memory_free=40536, memory_total=81920, utilization_gpu=3),
        GpuSnapshot(gpu_id=1, memory_free=21000, memory_total=81920, utilization_gpu=77),
    ]


def test_choose_gpu_ids_sorts_by_idle_then_free_memory_then_id():
    snapshots = [
        GpuSnapshot(gpu_id=2, memory_free=20000, memory_total=81920, utilization_gpu=10),
        GpuSnapshot(gpu_id=0, memory_free=10000, memory_total=81920, utilization_gpu=10),
        GpuSnapshot(gpu_id=1, memory_free=70000, memory_total=81920, utilization_gpu=80),
    ]

    selected = choose_gpu_ids(snapshots, min_gpus=2)

    assert selected == [2, 0]


def test_choose_gpu_ids_expands_when_memory_target_requires_more_devices():
    snapshots = [
        GpuSnapshot(gpu_id=0, memory_free=20000, memory_total=81920, utilization_gpu=0),
        GpuSnapshot(gpu_id=1, memory_free=18000, memory_total=81920, utilization_gpu=0),
        GpuSnapshot(gpu_id=2, memory_free=17000, memory_total=81920, utilization_gpu=0),
    ]

    selected = choose_gpu_ids(snapshots, min_gpus=1, target_free_memory=35000)

    assert selected == [0, 1]


def test_query_nvidia_smi_returns_parsed_snapshots(monkeypatch):
    def fake_check_output(command, text):
        assert command == [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        assert text is True
        return "3, 1000, 2000, 12\n"

    monkeypatch.setattr(gpu_selection.subprocess, "check_output", fake_check_output)

    assert gpu_selection.query_nvidia_smi() == [
        GpuSnapshot(gpu_id=3, memory_free=1000, memory_total=2000, utilization_gpu=12)
    ]


def test_main_prints_selection_policy_and_defaults(monkeypatch, capsys):
    def fake_choose_gpu_ids(snapshots, *, min_gpus, target_free_memory=None):
        assert snapshots == [
            GpuSnapshot(gpu_id=0, memory_free=10, memory_total=20, utilization_gpu=0)
        ]
        assert min_gpus == 1
        assert target_free_memory is None
        return [0]

    monkeypatch.setattr(
        gpu_selection,
        "query_nvidia_smi",
        lambda: [GpuSnapshot(gpu_id=0, memory_free=10, memory_total=20, utilization_gpu=0)],
    )
    monkeypatch.setattr(gpu_selection, "choose_gpu_ids", fake_choose_gpu_ids)
    monkeypatch.setattr(sys, "argv", ["gpu_selection"])

    gpu_selection.main()

    output = capsys.readouterr().out
    assert '"selected_ids": [\n    0\n  ]' in output
    assert '"selection_priority": [\n    "highest idle capacity",\n    "highest free memory",\n    "ascending gpu_id"\n  ]' in output
