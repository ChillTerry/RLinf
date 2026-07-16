from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.habitat.habitat_env import HabitatEnv


class _FakeVectorEnv:
    def __init__(self):
        self.params = None

    def reconfigure_env_fns(self, params):
        self.params = params


def _habitat_env(*, enabled=True):
    env = HabitatEnv.__new__(HabitatEnv)
    env.cfg = SimpleNamespace(
        action_length_bucketing=enabled,
        global_plan={
            "config_path": "habitat.yaml",
            "overrides": ["habitat.environment.max_episode_steps=128"],
            "episode_sequences": [[["legacy-a"], ["legacy-b"]]],
            "bucket_plans": {
                "B0": {
                    "horizon_steps": 84,
                    "episode_sequences": [
                        [["short-a", "short-a2"], ["short-b", "short-b2"]]
                    ],
                }
            },
        },
    )
    env.env = _FakeVectorEnv()
    env.seed_offset = 0
    env.seed = 10
    env.num_envs = 2
    env.group_size = 1
    env.num_group = 2
    env._elapsed_steps = np.ones(2, dtype=np.int32)
    env.first_done_cached_mask = np.ones(2, dtype=bool)
    env.current_raw_obs = [object(), object()]
    env.max_episode_steps = 128
    env._active_bucket_id = None
    env._bucket_cursors = {}
    return env


def test_activate_bucket_reconfigures_only_selected_episode_assignment():
    env = _habitat_env()

    env.activate_bucket("B0", 84)

    assert env._active_bucket_id == "B0"
    assert env.max_episode_steps == 84
    assert [params["episode_ids"] for params in env.env.params] == [
        ["short-a", "short-a2"],
        ["short-b", "short-b2"],
    ]
    assert all(
        "habitat.environment.max_episode_steps=84" in params["overrides"]
        for params in env.env.params
    )
    assert not env._elapsed_steps.any()
    assert not env.first_done_cached_mask.any()
    assert env.current_raw_obs is None

    state = env.get_curriculum_state()
    env.activate_bucket("B0", 84)
    assert [params["episode_ids"] for params in env.env.params] == [
        ["short-a2", "short-a"],
        ["short-b2", "short-b"],
    ]
    env.load_curriculum_state(state)
    assert env.get_curriculum_state() == state


def test_activate_bucket_is_unavailable_when_feature_is_off():
    env = _habitat_env(enabled=False)

    with pytest.raises(RuntimeError, match="not enabled"):
        env.activate_bucket("B0", 84)
