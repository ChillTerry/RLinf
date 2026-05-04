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

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class GpuSnapshot:
    gpu_id: int
    memory_free: int
    memory_total: int
    utilization_gpu: int

    @property
    def idle_capacity(self) -> int:
        return 100 - self.utilization_gpu


def parse_nvidia_smi_csv(output: str) -> list[GpuSnapshot]:
    snapshots = []
    for line in output.splitlines():
        if not line.strip():
            continue
        gpu_id, memory_free, memory_total, utilization_gpu = (
            int(value.strip()) for value in line.split(",")
        )
        snapshots.append(
            GpuSnapshot(
                gpu_id=gpu_id,
                memory_free=memory_free,
                memory_total=memory_total,
                utilization_gpu=utilization_gpu,
            )
        )
    return snapshots


def sort_gpu_snapshots(snapshots: list[GpuSnapshot]) -> list[GpuSnapshot]:
    return sorted(
        snapshots,
        key=lambda snapshot: (
            -snapshot.idle_capacity,
            -snapshot.memory_free,
            snapshot.gpu_id,
        ),
    )


def choose_gpu_ids(
    snapshots: list[GpuSnapshot],
    *,
    min_gpus: int,
    target_free_memory: int | None = None,
) -> list[int]:
    selected_ids = []
    selected_memory = 0

    for snapshot in sort_gpu_snapshots(snapshots):
        if len(selected_ids) >= min_gpus and (
            target_free_memory is None or selected_memory >= target_free_memory
        ):
            break
        selected_ids.append(snapshot.gpu_id)
        selected_memory += snapshot.memory_free

    return selected_ids


def query_nvidia_smi() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpus", type=int, required=True)
    parser.add_argument("--target-free-memory", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    snapshots = parse_nvidia_smi_csv(query_nvidia_smi())
    selection_priority = sort_gpu_snapshots(snapshots)
    selected_ids = choose_gpu_ids(
        snapshots,
        min_gpus=args.min_gpus,
        target_free_memory=args.target_free_memory,
    )
    payload = {
        "snapshots": [asdict(snapshot) for snapshot in snapshots],
        "selected_ids": selected_ids,
        "selection_priority": [asdict(snapshot) for snapshot in selection_priority],
    }
    text = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
