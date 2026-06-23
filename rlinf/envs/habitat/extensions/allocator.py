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

from __future__ import annotations

import heapq
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
_MB_PER_GB = 1024.0

EpisodeId = int | str
EpisodeSequences = list[list[list[EpisodeId]]]


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: EpisodeId
    scene_id: str
    weight: int


def load_scene_vram_profile(profile_path: str | Path | None = None) -> dict[str, int]:
    if profile_path is None:
        profile_path = (
            Path(__file__).resolve().parent / "config" / "scene_vram_profile.json"
        )
    with Path(profile_path).open("r", encoding="utf-8") as file_obj:
        scene_profile = json.load(file_obj)
    return {
        scene_key: int(scene_info["vram_cost_mb"])
        for scene_key, scene_info in scene_profile["scenes"].items()
    }


def get_assignment_mode(auto_reset: bool) -> str:
    """Map Habitat's auto_reset behavior onto eval/train assignment mode."""
    if auto_reset:
        return "eval"
    return "train"


def vram_balance_episode_sequences(
    episodes,
    *,
    auto_reset: bool,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
    scene_weights: dict[str, int] | None = None,
) -> EpisodeSequences:
    if scene_weights is None:
        scene_weights = load_scene_vram_profile()
    episode_records = build_episode_records(episodes, scene_weights)
    total_episode_count = len(episode_records)
    episode_records = trim_episode_records(
        episode_records,
        mode=get_assignment_mode(auto_reset),
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )
    dropped_episode_count = total_episode_count - len(episode_records)
    episode_sequences = assign_episode_sequences(
        episode_records,
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        dropped_episode_count=dropped_episode_count,
    )
    return episode_sequences


def random_episode_sequences(
    episodes,
    *,
    seed: int,
    auto_reset: bool,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> EpisodeSequences:
    episode_ids = [episode.episode_id for episode in episodes]
    total_episode_count = len(episode_ids)
    episode_ids = trim_episode_ids(
        episode_ids,
        mode=get_assignment_mode(auto_reset),
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(episode_ids)

    dropped_episode_count = total_episode_count - len(episode_ids)
    episode_sequences = assign_random_episode_sequences(
        episode_ids,
        total_num_processes=total_num_processes,
        num_group=num_group,
    )

    logger.info(
        "Randomly assigned %s Habitat episodes across %s processes and %s groups "
        "with seed=%s; dropped %s episodes.",
        len(episode_ids),
        total_num_processes,
        num_group,
        seed,
        dropped_episode_count,
    )
    return episode_sequences


def build_episode_records(
    episodes, scene_weights: dict[str, int]
) -> list[EpisodeRecord]:
    records: list[EpisodeRecord] = []
    for episode in episodes:
        scene_key = normalize_scene_key(episode.scene_id)
        records.append(
            EpisodeRecord(
                episode_id=episode.episode_id,
                scene_id=episode.scene_id,
                weight=int(scene_weights[scene_key]),
            )
        )
    return records


def normalize_scene_key(scene_id: str) -> str:
    scene_path = Path(scene_id)
    if scene_path.suffix:
        return scene_path.stem
    return scene_id


def trim_episode_records(
    records: list[EpisodeRecord],
    *,
    mode: str,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> list[EpisodeRecord]:
    target_count = get_trimmed_episode_count(
        len(records),
        mode=mode,
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )
    return records[:target_count]


def trim_episode_ids(
    episode_ids: list[EpisodeId],
    *,
    mode: str,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> list[EpisodeId]:
    target_count = get_trimmed_episode_count(
        len(episode_ids),
        mode=mode,
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=total_num_envs,
        max_steps_per_rollout_epoch=max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )
    return episode_ids[:target_count]


def get_trimmed_episode_count(
    episode_count: int,
    *,
    mode: str,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int,
    max_steps_per_rollout_epoch: int,
    max_episode_steps: int,
) -> int:
    if mode == "eval":
        assert max_steps_per_rollout_epoch % max_episode_steps == 0, (
            "max_steps_per_rollout_epoch must be divisible by max_episode_steps"
        )
        slot_count = int(max_steps_per_rollout_epoch / max_episode_steps)
        return total_num_envs * slot_count
    if mode == "train":
        granularity = total_num_processes * num_group
        return (episode_count // granularity) * granularity
    raise ValueError(f"Unknown mode: {mode}")


def assign_random_episode_sequences(
    episode_ids: list[EpisodeId],
    *,
    total_num_processes: int,
    num_group: int,
) -> EpisodeSequences:
    """Return shuffled episode ids as [process_idx][group_idx][slot_idx]."""
    if not episode_ids:
        return [[[] for _ in range(num_group)] for _ in range(total_num_processes)]

    total_streams = total_num_processes * num_group
    if len(episode_ids) % total_streams != 0:
        raise ValueError(
            "episode count must be divisible by total_num_processes * num_group"
        )

    slot_count = len(episode_ids) // total_streams
    sequences = [
        [[] for _ in range(num_group)] for _ in range(total_num_processes)
    ]

    cursor = 0
    for process_idx in range(total_num_processes):
        for group_idx in range(num_group):
            sequences[process_idx][group_idx] = episode_ids[
                cursor : cursor + slot_count
            ]
            cursor += slot_count

    return sequences


def assign_episode_sequences(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
    total_num_envs: int | None = None,
    dropped_episode_count: int = 0,
) -> EpisodeSequences:
    """Return episode ids as [process_idx][group_idx][slot_idx]."""
    if not records:
        return [[[] for _ in range(num_group)] for _ in range(total_num_processes)]

    total_streams = total_num_processes * num_group
    if len(records) % total_streams != 0:
        raise ValueError(
            "record count must be divisible by total_num_processes * num_group"
        )

    slot_count = len(records) // total_streams
    sequences = _assign_greedy(
        records,
        total_num_processes=total_num_processes,
        num_group=num_group,
        slot_count=slot_count,
    )

    logger.info(
        "%s",
        render_assignment_vram_summary(
            records=records,
            sequences=sequences,
            total_num_processes=total_num_processes,
            num_group=num_group,
            total_num_envs=total_num_envs,
            dropped_episode_count=dropped_episode_count,
        ),
    )
    return sequences


def _assign_greedy(
    records: list[EpisodeRecord],
    *,
    total_num_processes: int,
    num_group: int,
    slot_count: int,
) -> EpisodeSequences:
    placements = [
        [[None for _ in range(slot_count)] for _ in range(num_group)]
        for _ in range(total_num_processes)
    ]
    process_slot_loads = [
        [0 for _ in range(slot_count)] for _ in range(total_num_processes)
    ]
    process_peak_loads = [0 for _ in range(total_num_processes)]
    process_total_loads = [0 for _ in range(total_num_processes)]
    open_cells = [
        (0, 0, 0, slot_idx, process_idx, 0)
        for slot_idx in range(slot_count)
        for process_idx in range(total_num_processes)
    ]
    heapq.heapify(open_cells)
    ordered_records = sorted(
        records,
        key=lambda record: (
            -record.weight,
            isinstance(record.episode_id, str),
            record.episode_id,
        ),
    )

    for record in ordered_records:
        _, _, _, slot_idx, process_idx, group_idx = heapq.heappop(open_cells)
        placements[process_idx][group_idx][slot_idx] = record.episode_id
        process_slot_loads[process_idx][slot_idx] += record.weight
        process_peak_loads[process_idx] = max(
            process_peak_loads[process_idx],
            process_slot_loads[process_idx][slot_idx],
        )
        process_total_loads[process_idx] += record.weight
        next_group_idx = group_idx + 1
        if next_group_idx < num_group:
            heapq.heappush(
                open_cells,
                (
                    process_slot_loads[process_idx][slot_idx],
                    process_peak_loads[process_idx],
                    process_total_loads[process_idx],
                    slot_idx,
                    process_idx,
                    next_group_idx,
                ),
            )

    return [
        [list(group_slots) for group_slots in process_slots]
        for process_slots in placements
    ]


def _compute_slot_vram_series(
    process_group_episode_ids: list[list[EpisodeId]],
    episode_weight_map: dict[EpisodeId, int],
) -> list[int]:
    if not process_group_episode_ids:
        return []
    slot_count = len(process_group_episode_ids[0])
    slot_series: list[int] = []
    for slot_idx in range(slot_count):
        slot_vram = 0
        for group_episode_ids in process_group_episode_ids:
            slot_vram += episode_weight_map[group_episode_ids[slot_idx]]
        slot_series.append(slot_vram)
    return slot_series


def _summarize_slot_series(slot_vram_mb: list[int]) -> dict[str, float]:
    if not slot_vram_mb:
        return {"max_gb": 0.0, "min_gb": 0.0, "mean_gb": 0.0, "std_gb": 0.0}
    max_gb = float(max(slot_vram_mb) / _MB_PER_GB)
    min_gb = float(min(slot_vram_mb) / _MB_PER_GB)
    mean_gb = float(sum(slot_vram_mb) / len(slot_vram_mb) / _MB_PER_GB)
    variance = sum((value / _MB_PER_GB - mean_gb) ** 2 for value in slot_vram_mb) / len(
        slot_vram_mb
    )
    return {
        "max_gb": max_gb,
        "min_gb": min_gb,
        "mean_gb": mean_gb,
        "std_gb": float(math.sqrt(variance)),
    }


def render_assignment_vram_summary(
    *,
    records: list[EpisodeRecord],
    sequences: EpisodeSequences,
    total_num_processes: int,
    num_group: int | None = None,
    total_num_envs: int | None = None,
    dropped_episode_count: int = 0,
) -> str:
    denom = total_num_processes * num_group
    assert total_num_envs % denom == 0, (
        "total_num_envs must be divisible by total_num_processes * num_group"
    )
    group_size = total_num_envs // denom
    episode_weight_map = {record.episode_id: int(record.weight) for record in records}

    per_gpu: dict[str, dict] = {}
    for process_idx in range(total_num_processes):
        slot_series = _compute_slot_vram_series(
            sequences[process_idx], episode_weight_map
        )
        slot_series = [value * group_size for value in slot_series]
        per_gpu[f"gpu_{process_idx}"] = {
            "slot_vram_gb": [value / _MB_PER_GB for value in slot_series],
            **_summarize_slot_series(slot_series),
        }

    total_episode_count = len(records) + dropped_episode_count
    lines = [
        (
            "Episodes: total={total}, kept={kept}, dropped={dropped}; per-gpu stats:"
        ).format(
            total=total_episode_count,
            kept=len(records),
            dropped=dropped_episode_count,
        )
    ]
    for gpu_name, gpu_data in per_gpu.items():
        lines.append(
            (
                "  {gpu}: max={max_gb:.2f} GB, min={min_gb:.2f} GB, "
                "mean={mean_gb:.2f} GB, std={std_gb:.2f} GB, slots={slot_len}"
            ).format(
                gpu=gpu_name,
                max_gb=gpu_data["max_gb"],
                min_gb=gpu_data["min_gb"],
                mean_gb=gpu_data["mean_gb"],
                std_gb=gpu_data["std_gb"],
                slot_len=len(gpu_data["slot_vram_gb"]),
            )
        )
    return "\n".join(lines)
