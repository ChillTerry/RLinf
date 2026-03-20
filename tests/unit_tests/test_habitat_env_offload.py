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

import importlib
import sys
import types
from typing import Any, cast

import numpy as np
import pytest


def _import_habitat_env_module(monkeypatch):
    class _Registry:
        def register_task_action(self, cls):
            return cls

    class _GlobalHydraSingleton:
        def is_initialized(self):
            return False

        def clear(self):
            return None

    class _GlobalHydra:
        _instance = _GlobalHydraSingleton()

        @classmethod
        def instance(cls):
            return cls._instance

    habitat_module = types.ModuleType("habitat")
    cast(Any, habitat_module).datasets = types.SimpleNamespace(
        make_dataset=lambda *args, **kwargs: types.SimpleNamespace(episodes=[])
    )

    monkeypatch.setitem(sys.modules, "habitat", habitat_module)
    monkeypatch.setitem(
        sys.modules,
        "habitat.core.embodied_task",
        types.SimpleNamespace(SimulatorTaskAction=type("SimulatorTaskAction", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "habitat.core.registry",
        types.SimpleNamespace(registry=_Registry()),
    )
    monkeypatch.setitem(
        sys.modules,
        "habitat_baselines.config.default",
        types.SimpleNamespace(get_config=lambda *args, **kwargs: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hydra.core.global_hydra",
        types.SimpleNamespace(GlobalHydra=_GlobalHydra),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.habitat.extensions.measures",
        types.SimpleNamespace(pass_format_check=lambda: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.habitat.extensions.utils",
        types.SimpleNamespace(
            observations_to_image=lambda *args, **kwargs: None,
            resize_observation_images=lambda *args, **kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "rlinf.envs.habitat.venv",
        types.SimpleNamespace(
            HabitatRLEnv=type("HabitatRLEnv", (), {}),
            ReconfigureSubprocEnv=type("ReconfigureSubprocEnv", (), {}),
        ),
    )

    sys.modules.pop("rlinf.envs.habitat.habitat_env", None)
    return importlib.import_module("rlinf.envs.habitat.habitat_env")


def test_habitat_env_close_is_idempotent_and_clears_live_handle(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    close_calls = []

    class _DummyEnv:
        def close(self):
            close_calls.append("closed")

    habitat_env.env = _DummyEnv()

    habitat_env_module.HabitatEnv.close(habitat_env)

    assert close_calls == ["closed"]
    assert not hasattr(habitat_env, "env")

    habitat_env_module.HabitatEnv.close(habitat_env)

    assert close_calls == ["closed"]
    assert not hasattr(habitat_env, "env")


def test_habitat_env_get_env_fns_reuses_cached_params_without_recomputing(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    habitat_env._cached_env_fn_params = None
    get_env_fn_params_calls = []

    def _fake_get_env_fn_params():
        get_env_fn_params_calls.append("called")
        return [
            {
                "config_path": "dummy-config.yaml",
                "episode_ids": ["episode-0"],
                "seed": 7,
            }
        ]

    monkeypatch.setattr(habitat_env, "_get_env_fn_params", _fake_get_env_fn_params)

    first_env_fns = habitat_env_module.HabitatEnv._get_env_fns(habitat_env)
    first_env_fns[0].__defaults__[0]["seed"] = 999
    second_env_fns = habitat_env_module.HabitatEnv._get_env_fns(habitat_env)

    assert get_env_fn_params_calls == ["called"]
    assert second_env_fns[0].__defaults__[0]["seed"] == 7


def test_habitat_env_init_env_rebuild_uses_cached_params(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    habitat_env._cached_env_fn_params = None
    get_env_fn_params_calls = []

    def _fake_get_env_fn_params():
        get_env_fn_params_calls.append("called")
        return [
            {
                "config_path": "dummy-config.yaml",
                "episode_ids": ["episode-0"],
                "seed": 13,
            }
        ]

    class _DummyReconfigureSubprocEnv:
        def __init__(self, env_fns):
            self.env_fns = env_fns

    monkeypatch.setattr(habitat_env, "_get_env_fn_params", _fake_get_env_fn_params)
    monkeypatch.setattr(
        habitat_env_module, "ReconfigureSubprocEnv", _DummyReconfigureSubprocEnv
    )
    monkeypatch.setattr(habitat_env, "close", lambda: None)

    habitat_env_module.HabitatEnv._init_env(habitat_env)
    habitat_env.env.env_fns[0].__defaults__[0]["seed"] = 2048
    habitat_env_module.HabitatEnv._init_env(habitat_env)

    assert get_env_fn_params_calls == ["called"]
    assert habitat_env.env.env_fns[0].__defaults__[0]["seed"] == 13


def test_habitat_env_offload_closes_live_env_marks_offloaded_and_clears_runtime_state(
    monkeypatch,
):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    close_calls = []

    class _DummyEnv:
        def close(self):
            close_calls.append("closed")

    habitat_env.env = _DummyEnv()
    habitat_env.num_envs = 2
    habitat_env._is_offloaded = False
    habitat_env.current_raw_obs = {"rgb": "live"}
    habitat_env.render_images = {0: "frame"}
    habitat_env.record_first_done_infos = [{"done": False}]
    habitat_env._elapsed_steps = np.array([3, 4], dtype=np.int32)
    habitat_env.prev_step_reward = np.array([0.5, 1.5])
    habitat_env.dones_once = np.array([True, False], dtype=bool)
    habitat_env.initial_distance_to_goal = np.array([8.0, 9.0])
    habitat_env._cached_env_fn_params = [{"seed": 11, "episode_ids": ["episode-0"]}]

    habitat_env_module.HabitatEnv.offload(habitat_env)

    assert close_calls == ["closed"]
    assert not hasattr(habitat_env, "env")
    assert habitat_env._is_offloaded is True
    assert habitat_env.current_raw_obs is None
    assert habitat_env.render_images == {}
    assert habitat_env.record_first_done_infos is None
    np.testing.assert_array_equal(
        habitat_env._elapsed_steps, np.zeros(2, dtype=np.int32)
    )
    np.testing.assert_array_equal(habitat_env.prev_step_reward, np.zeros(2))
    np.testing.assert_array_equal(habitat_env.dones_once, np.zeros(2, dtype=bool))
    np.testing.assert_array_equal(habitat_env.initial_distance_to_goal, np.zeros(2))
    assert habitat_env._cached_env_fn_params == [
        {"seed": 11, "episode_ids": ["episode-0"]}
    ]


def test_habitat_env_offload_is_idempotent_and_preserves_cached_rebuild_metadata(
    monkeypatch,
):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    close_calls = []

    class _DummyEnv:
        def close(self):
            close_calls.append("closed")

    cached_env_fn_params = [
        {
            "config_path": "dummy-config.yaml",
            "episode_ids": ["episode-0"],
            "seed": 17,
        }
    ]
    habitat_env.env = _DummyEnv()
    habitat_env.num_envs = 1
    habitat_env._is_offloaded = False
    habitat_env.current_raw_obs = {"depth": "live"}
    habitat_env.render_images = {0: "frame"}
    habitat_env.record_first_done_infos = [{"done": True}]
    habitat_env._elapsed_steps = np.array([5], dtype=np.int32)
    habitat_env.prev_step_reward = np.array([2.0])
    habitat_env.dones_once = np.array([True], dtype=bool)
    habitat_env.initial_distance_to_goal = np.array([6.0])
    habitat_env._cached_env_fn_params = cached_env_fn_params

    habitat_env_module.HabitatEnv.offload(habitat_env)
    habitat_env.current_raw_obs = "should-stay-cleared"
    habitat_env.render_images[1] = "mutation"
    habitat_env.record_first_done_infos = "should-stay-cleared"
    habitat_env._elapsed_steps[0] = 99
    habitat_env.prev_step_reward[0] = 99.0
    habitat_env.dones_once[0] = True
    habitat_env.initial_distance_to_goal[0] = 99.0

    habitat_env_module.HabitatEnv.offload(habitat_env)

    assert close_calls == ["closed"]
    assert habitat_env._is_offloaded is True
    assert habitat_env.current_raw_obs == "should-stay-cleared"
    assert habitat_env.render_images == {1: "mutation"}
    assert habitat_env.record_first_done_infos == "should-stay-cleared"
    np.testing.assert_array_equal(
        habitat_env._elapsed_steps, np.array([99], dtype=np.int32)
    )
    np.testing.assert_array_equal(habitat_env.prev_step_reward, np.array([99.0]))
    np.testing.assert_array_equal(habitat_env.dones_once, np.array([True], dtype=bool))
    np.testing.assert_array_equal(
        habitat_env.initial_distance_to_goal, np.array([99.0])
    )
    assert habitat_env._cached_env_fn_params is cached_env_fn_params
    assert habitat_env._cached_env_fn_params == [
        {
            "config_path": "dummy-config.yaml",
            "episode_ids": ["episode-0"],
            "seed": 17,
        }
    ]


def test_habitat_env_reset_reloads_offloaded_env_and_succeeds(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    rebuild_calls = []
    reset_calls = []

    class _ReloadedEnv:
        def reset(self, env_idx):
            reset_calls.append(np.array(env_idx).tolist())
            return [{"rgb": "reloaded-obs"}]

        def get_env_attr(self, attr_name):
            assert attr_name == "config"
            return [{"config": "from-reloaded-env"}]

    def _fake_init_env():
        rebuild_calls.append("rebuilt")
        habitat_env.env = _ReloadedEnv()
        habitat_env._is_offloaded = False

    monkeypatch.setattr(habitat_env, "_init_env", _fake_init_env)
    monkeypatch.setattr(habitat_env, "_wrap_obs", lambda raw_obs: {"wrapped": raw_obs})

    habitat_env.num_envs = 1
    habitat_env._is_offloaded = True
    habitat_env.env_config = {"config": "stale"}
    habitat_env.record_first_done_infos = None
    habitat_env.current_raw_obs = None
    habitat_env._elapsed_steps = np.array([9], dtype=np.int32)
    habitat_env.dones_once = np.array([True], dtype=bool)

    obs, infos = habitat_env_module.HabitatEnv.reset(habitat_env)

    assert rebuild_calls == ["rebuilt"]
    assert reset_calls == [[0]]
    assert habitat_env._is_offloaded is False
    assert obs == {"wrapped": [{"rgb": "reloaded-obs"}]}
    assert infos == {}
    assert habitat_env.current_raw_obs == [{"rgb": "reloaded-obs"}]
    np.testing.assert_array_equal(
        habitat_env._elapsed_steps, np.zeros(1, dtype=np.int32)
    )
    np.testing.assert_array_equal(habitat_env.dones_once, np.zeros(1, dtype=bool))


def test_habitat_env_subset_reset_after_offload_bootstraps_full_obs_list(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    rebuild_calls = []
    reset_calls = []

    class _ReloadedEnv:
        def reset(self, env_idx):
            env_idx = np.asarray(env_idx)
            reset_calls.append(env_idx.tolist())
            return [{"rgb": f"obs-{idx}"} for idx in env_idx]

        def get_env_attr(self, attr_name):
            assert attr_name == "config"
            return [{"config": "from-reloaded-env"}]

    def _fake_init_env():
        rebuild_calls.append("rebuilt")
        habitat_env.env = _ReloadedEnv()
        habitat_env._is_offloaded = False

    def _fake_wrap_obs(raw_obs):
        assert len(raw_obs) == habitat_env.num_envs
        assert all(obs is not None for obs in raw_obs)
        assert raw_obs == [
            {"rgb": "obs-0"},
            {"rgb": "obs-1"},
            {"rgb": "obs-2"},
        ]
        return {"wrapped": raw_obs}

    monkeypatch.setattr(habitat_env, "_init_env", _fake_init_env)
    monkeypatch.setattr(habitat_env, "_wrap_obs", _fake_wrap_obs)

    habitat_env.num_envs = 3
    habitat_env._is_offloaded = True
    habitat_env.env_config = {"config": "stale"}
    habitat_env.record_first_done_infos = None
    habitat_env.current_raw_obs = None
    habitat_env._elapsed_steps = np.array([5, 6, 7], dtype=np.int32)
    habitat_env.dones_once = np.array([True, True, False], dtype=bool)

    obs, infos = habitat_env_module.HabitatEnv.reset(habitat_env, env_idx=np.array([1]))

    assert rebuild_calls == ["rebuilt"]
    assert reset_calls == [[0, 1, 2], [1]]
    assert habitat_env._is_offloaded is False
    assert obs == {"wrapped": [{"rgb": "obs-0"}, {"rgb": "obs-1"}, {"rgb": "obs-2"}]}
    assert infos == {}
    assert habitat_env.current_raw_obs == [
        {"rgb": "obs-0"},
        {"rgb": "obs-1"},
        {"rgb": "obs-2"},
    ]
    np.testing.assert_array_equal(
        habitat_env._elapsed_steps, np.array([5, 0, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        habitat_env.dones_once, np.array([True, False, False], dtype=bool)
    )


def test_habitat_env_scalar_subset_reset_after_offload_accepts_int_env_idx(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    rebuild_calls = []
    reset_calls = []

    class _ReloadedEnv:
        def reset(self, env_idx):
            env_idx = np.asarray(env_idx)
            reset_calls.append(env_idx.tolist())
            return [{"rgb": f"obs-{idx}"} for idx in env_idx]

        def get_env_attr(self, attr_name):
            assert attr_name == "config"
            return [{"config": "from-reloaded-env"}]

    def _fake_init_env():
        rebuild_calls.append("rebuilt")
        habitat_env.env = _ReloadedEnv()
        habitat_env._is_offloaded = False

    def _fake_wrap_obs(raw_obs):
        assert len(raw_obs) == habitat_env.num_envs
        assert all(obs is not None for obs in raw_obs)
        assert raw_obs == [
            {"rgb": "obs-0"},
            {"rgb": "obs-1"},
            {"rgb": "obs-2"},
        ]
        return {"wrapped": raw_obs}

    monkeypatch.setattr(habitat_env, "_init_env", _fake_init_env)
    monkeypatch.setattr(habitat_env, "_wrap_obs", _fake_wrap_obs)

    habitat_env.num_envs = 3
    habitat_env._is_offloaded = True
    habitat_env.env_config = {"config": "stale"}
    habitat_env.record_first_done_infos = None
    habitat_env.current_raw_obs = None
    habitat_env._elapsed_steps = np.array([5, 6, 7], dtype=np.int32)
    habitat_env.dones_once = np.array([True, True, False], dtype=bool)

    obs, infos = habitat_env_module.HabitatEnv.reset(habitat_env, env_idx=1)

    assert rebuild_calls == ["rebuilt"]
    assert reset_calls == [[0, 1, 2], [1]]
    assert habitat_env._is_offloaded is False
    assert obs == {"wrapped": [{"rgb": "obs-0"}, {"rgb": "obs-1"}, {"rgb": "obs-2"}]}
    assert infos == {}
    assert habitat_env.current_raw_obs == [
        {"rgb": "obs-0"},
        {"rgb": "obs-1"},
        {"rgb": "obs-2"},
    ]
    np.testing.assert_array_equal(
        habitat_env._elapsed_steps, np.array([5, 0, 7], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        habitat_env.dones_once, np.array([True, False, False], dtype=bool)
    )


def test_habitat_env_onload_refreshes_env_config_from_rebuilt_env(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    rebuild_calls = []
    refreshed_config = {"config": "from-rebuilt-subprocess"}

    class _ReloadedEnv:
        def get_env_attr(self, attr_name):
            assert attr_name == "config"
            return [refreshed_config]

    def _fake_init_env():
        rebuild_calls.append("rebuilt")
        habitat_env.env = _ReloadedEnv()
        habitat_env._is_offloaded = False

    monkeypatch.setattr(habitat_env, "_init_env", _fake_init_env)

    habitat_env._is_offloaded = True
    habitat_env.env_config = {"config": "stale"}

    habitat_env_module.HabitatEnv.onload(habitat_env)

    assert rebuild_calls == ["rebuilt"]
    assert habitat_env._is_offloaded is False
    assert habitat_env.env_config is refreshed_config


def test_habitat_env_onload_is_noop_when_env_is_already_loaded(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    rebuild_calls = []
    live_env = object()
    live_config = {"config": "already-loaded"}

    monkeypatch.setattr(
        habitat_env, "_init_env", lambda: rebuild_calls.append("rebuilt")
    )

    habitat_env._is_offloaded = False
    habitat_env.env = live_env
    habitat_env.env_config = live_config

    habitat_env_module.HabitatEnv.onload(habitat_env)

    assert rebuild_calls == []
    assert habitat_env.env is live_env
    assert habitat_env.env_config is live_config


def test_habitat_env_step_raises_reset_required_error_when_offloaded(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    habitat_env._is_offloaded = True

    with pytest.raises(
        RuntimeError,
        match=r"HabitatEnv is offloaded. Call reset\(\) before stepping\.",
    ):
        habitat_env_module.HabitatEnv.step(habitat_env, np.array(["stop"]))


def test_habitat_env_chunk_step_raises_reset_required_error_when_offloaded(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    habitat_env._is_offloaded = True

    with pytest.raises(
        RuntimeError,
        match=r"HabitatEnv is offloaded. Call reset\(\) before stepping\.",
    ):
        habitat_env_module.HabitatEnv.chunk_step(
            habitat_env, np.array([["stop"]], dtype="U12")
        )


def test_habitat_env_flush_video_is_noop_when_offloaded(monkeypatch):
    habitat_env_module = _import_habitat_env_module(monkeypatch)
    habitat_env = object.__new__(habitat_env_module.HabitatEnv)
    habitat_env._is_offloaded = True
    habitat_env.render_images = {"episode_1": ["frame"]}
    habitat_env.video_cfg = types.SimpleNamespace(video_base_dir="unused", fps=30)

    class _EnvAccessMustNotHappen:
        def __getattribute__(self, name):
            raise AssertionError(f"flush_video() touched self.env via {name}")

    habitat_env.env = _EnvAccessMustNotHappen()

    habitat_env_module.HabitatEnv.flush_video(habitat_env)

    assert habitat_env.render_images == {"episode_1": ["frame"]}
