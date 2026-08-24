# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.go2.go2_vln_env import Go2VLNEnv


def create_go2_vln_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    del env_cfg
    return Go2VLNEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )


register(
    id="Go2VLNEnv-v1",
    entry_point="rlinf.envs.realworld.go2.tasks:create_go2_vln_env",
)

__all__ = ["Go2VLNEnv", "create_go2_vln_env"]

