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

"""Capture rollout_batch to disk for actor-only debug/replay.

Runs one full env → rollout step, saves each actor worker's ``rollout_batch``
(the dict of tensors produced after ``recv_rollout_trajectories`` +
``_process_received_rollout_batch``) to ``<save_dir>/rollout_batch_rank<R>.pt``,
then exits immediately **before** ``compute_advantages_and_returns``.

Usage::

    # Using the same config as your normal training run:
    python toolkits/debug_actor/save_rollout_batch.py \\
        --config-path ../../examples/embodiment/config \\
        --config-name <config_name> \\
        [+save_dir=/tmp/debug_actor_cache]

The saved files can later be loaded by ``replay_actor_training.py`` to debug
the actor training loop without waiting for env/rollout generation.
"""

import json
import os

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)

logger = get_logger()

_DEFAULT_SAVE_DIR = "/tmp/debug_actor_cache"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EMBODIED_PATH = os.path.join(_REPO_ROOT, "examples", "embodiment")

os.environ.setdefault("REPO_PATH", _REPO_ROOT)
os.environ.setdefault("EMBODIED_PATH", _EMBODIED_PATH)


class CaptureFSDPActor(EmbodiedFSDPActor):
    """EmbodiedFSDPActor with an extra method to persist rollout_batch."""

    def save_rollout_batch(self, save_dir: str) -> dict:
        """Save ``self.rollout_batch`` to disk (runs on each Ray worker).

        Returns:
            dict with ``rank``, ``path``, and ``keys`` for logging.
        """
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"rollout_batch_rank{self._rank}.pt")

        batch = getattr(self, "rollout_batch", None)
        if batch is None:
            raise RuntimeError(
                f"[rank {self._rank}] rollout_batch not set — "
                "recv_rollout_trajectories may not have been called."
            )

        def _to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: _to_cpu(v) for k, v in obj.items()}
            return obj

        cpu_batch = _to_cpu(batch)
        torch.save(cpu_batch, save_path)
        self.log_info(f"Saved rollout_batch → {save_path}")
        return {"rank": self._rank, "path": save_path, "keys": list(cpu_batch.keys())}


class CaptureRunner(EmbodiedRunner):
    """Runs one env+rollout step, saves the rollout_batch, then exits."""

    def __init__(self, save_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = save_dir

    def run(self):
        """Single-step capture — no training."""
        self.actor.set_global_step(self.global_step)
        self.rollout.set_global_step(self.global_step)

        logger.info("=== [CaptureRunner] Syncing weights to rollout ===")
        self.update_rollout_weights()

        logger.info("=== [CaptureRunner] Running env + rollout generation ===")
        self.env.interact(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
            actor_channel=self.actor_channel,
        )
        self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel,
        ).wait()
        rollout_handle.wait()

        logger.info(
            f"=== [CaptureRunner] Rollout complete — saving to {self.save_dir} ==="
        )
        results = self.actor.save_rollout_batch(self.save_dir).wait()

        # Also persist a resolved copy of the config for reference / replay.
        cfg_path = os.path.join(self.save_dir, "config.yaml")
        OmegaConf.save(self.cfg, cfg_path)

        for r in results:
            if r is not None:
                logger.info(f"  rank {r['rank']} → {r['path']}  (keys: {r['keys']})")

        logger.info(
            f"=== [CaptureRunner] Done — {len(results)} file(s) + config.yaml "
            f"saved to {self.save_dir} ==="
        )


@hydra.main(
    version_base="1.1",
    config_path="../../examples/embodiment/config",
    config_name="maniskill_ppo_openvlaoft",
)
def main(cfg: DictConfig) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    save_dir: str = cfg.get("save_dir", _DEFAULT_SAVE_DIR)

    if cfg.algorithm.loss_type == "embodied_sac":
        raise ValueError(
            "save_rollout_batch.py only supports PPO/GRPO (non-SAC) configs. "
            "Got loss_type='embodied_sac'."
        )

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Actor — use CaptureFSDPActor so save_rollout_batch is dispatchable
    actor_placement = component_placement.get_strategy("actor")
    actor_group = CaptureFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    # Rollout
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    # Env
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = CaptureRunner(
        save_dir=save_dir,
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
