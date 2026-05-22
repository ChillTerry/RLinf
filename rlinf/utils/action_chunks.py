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

from typing import Any, Literal


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def get_effective_num_action_chunks(
    model_cfg: Any,
    mode: Literal["train", "eval"],
) -> int:
    if (
        mode == "eval"
        and _cfg_get(model_cfg, "model_type") == "uninavid"
        and _cfg_get(model_cfg, "eval_num_action_chunks") is not None
    ):
        return int(_cfg_get(model_cfg, "eval_num_action_chunks"))
    return int(_cfg_get(model_cfg, "num_action_chunks"))
