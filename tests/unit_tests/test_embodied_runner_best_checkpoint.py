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

from pathlib import Path
from unittest.mock import MagicMock

from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner


class _ImmediateHandle:
    def wait(self):
        return None


def _make_runner(tmp_path: Path) -> EmbodiedRunner:
    cfg = OmegaConf.create(
        {
            "runner": {
                "weight_sync_interval": 1,
                "overlap_env_bootstrap": False,
                "save_best_ckpt": True,
                "best_metric": "success",
                "best_metric_criteria": "max",
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "save_policy",
                },
            }
        }
    )
    actor = MagicMock()
    actor.save_checkpoint.return_value = _ImmediateHandle()

    runner = object.__new__(EmbodiedRunner)
    runner.cfg = cfg
    runner.actor = actor
    runner.logger = MagicMock()
    runner.save_best_ckpt = True
    runner.best_metric = "success"
    runner.best_metric_criteria = "max"
    runner.best_metric_value = float("-inf")
    runner.global_step = 3
    return runner


def test_regular_checkpoint_is_independent_from_best_checkpoint(tmp_path):
    runner = _make_runner(tmp_path)

    runner._save_checkpoint()

    runner.actor.save_checkpoint.assert_called_once()
    save_path = runner.actor.save_checkpoint.call_args.args[0]
    assert save_path.endswith("checkpoints/global_step_3/actor")


def test_save_best_checkpoint_when_metric_improves(tmp_path):
    runner = _make_runner(tmp_path)

    assert runner._save_best_checkpoint({"success": 0.2})
    assert runner.best_metric_value == 0.2
    runner.actor.save_checkpoint.assert_called_once()
    save_path = runner.actor.save_checkpoint.call_args.args[0]
    assert save_path.endswith("checkpoints/best_model/actor")

    runner.actor.save_checkpoint.reset_mock()
    assert not runner._save_best_checkpoint({"success": 0.1})
    runner.actor.save_checkpoint.assert_not_called()


def test_save_best_ckpt_flag_controls_best_checkpoint(tmp_path):
    runner = _make_runner(tmp_path)
    runner.save_best_ckpt = False

    if runner.save_best_ckpt:
        runner._save_best_checkpoint({"success": 0.2})

    runner.actor.save_checkpoint.assert_not_called()


def _make_validate_cfg(val_check_interval: int):
    return OmegaConf.create(
        {
            "runner": {
                "save_best_ckpt": True,
                "best_metric": "success",
                "best_metric_criteria": "max",
                "val_check_interval": val_check_interval,
                "only_eval": False,
            },
            "actor": {
                "model": {
                    "model_type": "uninavid",
                    "num_action_chunks": 4,
                    "add_value_head": False,
                }
            },
            "algorithm": {"loss_type": "actor"},
            "rollout": {"pipeline_stage_num": 1},
            "env": {
                "train": {
                    "total_num_envs": 8,
                    "max_steps_per_rollout_epoch": 4,
                    "group_size": 1,
                },
                "eval": {
                    "total_num_envs": 8,
                    "max_steps_per_rollout_epoch": 4,
                    "group_size": 1,
                },
            },
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"actor,env,rollout": "all"},
            },
        }
    )


def test_save_best_ckpt_requires_validation_interval():
    cfg = _make_validate_cfg(val_check_interval=-1)

    try:
        validate_embodied_cfg(cfg)
    except AssertionError as exc:
        assert "save_best_ckpt requires runner.val_check_interval > 0" in str(exc)
    else:
        raise AssertionError("validate_embodied_cfg should reject disabled eval")
