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

"""Replay actor training from a previously-captured rollout_batch.

Loads the ``rollout_batch_rank<R>.pt`` files saved by ``save_rollout_batch.py``,
initialises **only** the actor worker group (no env, no rollout workers), injects
the batch, and runs ``compute_advantages_and_returns()`` →
``run_training()``.  This lets you iterate on actor-side code (losses,
advantages, model forward) without waiting for env/rollout generation each time.

Usage::

    python toolkits/debug_actor/replay_actor_training.py \\
        --config-path ../../examples/embodiment/config \\
        --config-name <config_name> \\
        [+load_dir=/tmp/debug_actor_cache] \\
        [+num_train_steps=1]

Set ``num_train_steps`` to run multiple training iterations on the same batch
(useful for profiling or convergence checks).

Breakpoint tip — add ``breakpoint()`` inside
``rlinf/workers/actor/fsdp_actor_worker.py`` in ``run_training`` or
``compute_advantages_and_returns`` and launch this script with a single worker
(``cluster.num_nodes: 1``, single-GPU placement) to step through interactively.
"""

import json
import os
from typing import cast

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

mp.set_start_method("spawn", force=True)

logger = get_logger()

_DEFAULT_LOAD_DIR = "/tmp/debug_actor_cache"

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_EMBODIED_PATH = os.path.join(_REPO_ROOT, "examples", "embodiment")

os.environ.setdefault("REPO_PATH", _REPO_ROOT)
os.environ.setdefault("EMBODIED_PATH", _EMBODIED_PATH)


class ReplayFSDPActor(EmbodiedFSDPActor):
    """EmbodiedFSDPActor that can load a rollout_batch from disk and skips
    rollout-weight-sync setup (no rollout workers are present)."""

    def _setup_rollout_weight_dst_ranks(self) -> None:
        self._weight_dst_rank_in_rollout = []

    def load_rollout_batch(self, load_dir: str) -> dict:
        """Load a previously-saved rollout_batch into ``self.rollout_batch``.

        Args:
            load_dir: directory containing ``rollout_batch_rank<R>.pt`` files.

        Returns:
            dict with ``rank`` and ``keys`` for logging.
        """
        load_path = os.path.join(load_dir, f"rollout_batch_rank{self._rank}.pt")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(
                f"[rank {self._rank}] Expected {load_path} but file not found. "
                "Did you run save_rollout_batch.py first?"
            )

        self.rollout_batch = torch.load(
            load_path, map_location="cpu", weights_only=True
        )
        self.log_info(
            f"Loaded rollout_batch from {load_path} "
            f"(keys: {list(self.rollout_batch.keys())})"
        )
        return {"rank": self._rank, "keys": list(self.rollout_batch.keys())}

    def optimizer_step(self) -> tuple[float, list[float]]:
        self.optimizer_steps += 1
        self.grad_scaler.unscale_(self.optimizer)

        max_norm = float(self.cfg.actor.optim.clip_grad)
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(
                cast(torch.nn.Module, self.model).parameters(),
                max_norm=max_norm,
                foreach=False,
            )
            .detach()
            .cpu()
            .item()
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[Replay] Non-finite grad norm {grad_norm} detected. "
                "Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        self.grad_scaler.update()

        if self.critic_warmup_steps > 0:
            lr_list = [0.0 for _ in self.optimizer.param_groups]
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.optimizer = self.build_optimizer(model=self.model)
                self.critic_warmup_steps = 0
        else:
            lr_list = [group["lr"] for group in self.optimizer.param_groups]

        return grad_norm, lr_list


@hydra.main(
    version_base="1.1",
    config_path="../../examples/embodiment/config",
    config_name="maniskill_ppo_openvlaoft",
)
def main(cfg: DictConfig) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    load_dir: str = cfg.get("load_dir", _DEFAULT_LOAD_DIR)
    num_train_steps: int = int(cfg.get("num_train_steps", 1))

    if cfg.algorithm.loss_type == "embodied_sac":
        raise ValueError(
            "replay_actor_training.py only supports PPO/GRPO (non-SAC) configs."
        )

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor = ReplayFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    logger.info("=== [Replay] Initialising actor workers (model + optimizer) ===")
    actor.init_worker().wait()

    logger.info(f"=== [Replay] Loading rollout_batch from {load_dir} ===")
    load_results = actor.load_rollout_batch(load_dir).wait()
    for r in load_results:
        if r is not None:
            logger.info(f"  rank {r['rank']}: keys={r['keys']}")

    for step in range(num_train_steps):
        logger.info(f"=== [Replay] Step {step + 1}/{num_train_steps} ===")

        actor.set_global_step(step)

        logger.info("  Computing advantages and returns …")
        adv_metrics = actor.compute_advantages_and_returns().wait()
        logger.info(f"  Rollout metrics: {adv_metrics}")

        logger.info("  Running actor training …")
        train_metrics = actor.run_training().wait()
        logger.info(f"  Training metrics: {train_metrics}")

        if num_train_steps > 1:
            logger.info("  Reloading rollout_batch for next step …")
            actor.load_rollout_batch(load_dir).wait()

    logger.info("=== [Replay] Done ===")


if __name__ == "__main__":
    main()
