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

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


def _load_plot_module():
    module_path = Path(".vscode/plot_rxr_prompt_token_histogram.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "plot_rxr_prompt_token_histogram",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_stat_values_reads_jsonl_and_filters_language(tmp_path):
    module = _load_plot_module()
    stats_path = tmp_path / "response_stats.jsonl"
    rows = [
        {"episode_id": "1", "language": "en-US", "response_token_len": 8},
        {"episode_id": "2", "language": "hi-IN", "response_token_len": 13},
        {"episode_id": "3", "language": "en-IN", "response_token_len": 21},
    ]
    stats_path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )

    values = module.load_stat_values(
        [str(stats_path)],
        value_key="response_token_len",
        languages={"en-US", "en-IN"},
    )

    assert values == [8, 21]


def test_load_stat_values_reads_csv(tmp_path):
    module = _load_plot_module()
    stats_path = tmp_path / "response_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=["episode_id", "language", "response_token_len"],
        )
        writer.writeheader()
        writer.writerow(
            {"episode_id": "1", "language": "en-US", "response_token_len": "5"}
        )
        writer.writerow(
            {"episode_id": "2", "language": "en-IN", "response_token_len": "11"}
        )

    values = module.load_stat_values(
        [str(stats_path)],
        value_key="response_token_len",
        languages=None,
    )

    assert values == [5, 11]
