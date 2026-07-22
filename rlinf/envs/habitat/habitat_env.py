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

import copy
import gzip
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Union

import gym
import habitat
import numpy as np
import torch
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat_baselines.config.default import get_config
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra

from rlinf.envs.habitat.extensions import measures, rxr_dataset, video  # noqa: F401
from rlinf.envs.habitat.extensions.allocator import (
    build_action_length_buckets,
    filter_episodes_by_gt_action_length,
    random_episode_sequences,
)
from rlinf.envs.habitat.extensions.bucket_scheduler import (
    build_round_robin_bucket_order,
    normalize_bucket_curriculum_plan,
)
from rlinf.envs.habitat.extensions.subgoal import (
    SubgoalRewardConfig,
    SubgoalRewardTracker,
)
from rlinf.envs.habitat.venv import ReconfigureSubprocEnv, make_habitat_env
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)

logger = logging.getLogger(__name__)


def build_habitat_global_plan(
    cfg,
    *,
    num_group: int,
    total_num_processes: int,
    max_episode_steps: int,
) -> dict:
    bucketing_enabled = bool(getattr(cfg, "action_length_bucketing", False))
    normalized_curriculum_plan = None
    compatibility_max_episode_steps = max_episode_steps
    if bucketing_enabled:
        normalized_curriculum_plan = normalize_bucket_curriculum_plan(
            bucket_step_range_map=cfg.bucket_step_range_map,
            bucket_max_steps=cfg.bucket_max_steps,
            curriculum_stages_map=cfg.curriculum_stages_map,
            rollout_epoch=int(cfg.rollout_epoch),
            curriculum_interval=int(cfg.curriculum_interval),
            effective_num_action_chunks=int(cfg.effective_num_action_chunks),
            bucket_schedule_seed=int(cfg.bucket_schedule_seed),
        )
        compatibility_max_episode_steps = max(
            int(value) for value in cfg.bucket_max_steps
        )

    # Habitat uses hydra to load the config, but hydra may already be
    # initialized elsewhere in the process.
    hydra_initialized = GlobalHydra.instance().is_initialized()
    if hydra_initialized:
        GlobalHydra.instance().clear()

    config_path = cfg.init_params.config_path
    overrides = build_habitat_overrides(
        cfg, max_episode_steps=compatibility_max_episode_steps
    )
    habitat_config = get_config(config_path, overrides=overrides)

    habitat_dataset = habitat.datasets.make_dataset(
        habitat_config.habitat.dataset.type,
        config=habitat_config.habitat.dataset,
    )

    sampled_scene_ids = get_sampled_habitat_scene_ids(
        habitat_dataset.episodes,
        getattr(cfg, "sample_num_scenes", None),
        cfg.seed,
    )
    if sampled_scene_ids is not None:
        sampled_scene_id_set = set(sampled_scene_ids)
        habitat_dataset.episodes = [
            ep for ep in habitat_dataset.episodes if ep.scene_id in sampled_scene_id_set
        ]

    gt_data = None
    filter_result = None
    min_gt_action_length = getattr(cfg, "min_gt_action_length", None)
    max_gt_action_length = getattr(cfg, "max_gt_action_length", None)
    if min_gt_action_length is not None or max_gt_action_length is not None:
        if max_gt_action_length is None:
            raise ValueError("min_gt_action_length requires max_gt_action_length.")
        gt_path = getattr(cfg, "gt_path", None)
        if gt_path is None:
            raise ValueError("GT action-length filtering requires gt_path.")
        gt_data = load_habitat_gt_data(gt_path)
        filter_result = filter_episodes_by_gt_action_length(
            habitat_dataset.episodes,
            gt_data,
            min_action_length=int(min_gt_action_length or 0),
            max_action_length=int(max_gt_action_length),
        )
        habitat_dataset.episodes = list(filter_result.episodes)
        if not habitat_dataset.episodes:
            raise ValueError(
                "No Habitat episodes remain after filtering GT action length to "
                f"[{int(min_gt_action_length or 0)}, {int(max_gt_action_length)}]."
            )
        if filter_result.dropped_below_ids:
            logger.info(
                "Dropped %s Habitat episodes with GT action length < %s; first ids: %s",
                len(filter_result.dropped_below_ids),
                int(min_gt_action_length or 0),
                filter_result.dropped_below_ids[:50],
            )
        if filter_result.dropped_above_ids:
            logger.info(
                "Dropped %s Habitat episodes with GT action length > %s; first ids: %s",
                len(filter_result.dropped_above_ids),
                int(max_gt_action_length),
                filter_result.dropped_above_ids[:50],
            )

    episode_sequences = random_episode_sequences(
        habitat_dataset.episodes,
        seed=cfg.seed,
        auto_reset=cfg.auto_reset,
        total_num_processes=total_num_processes,
        num_group=num_group,
        total_num_envs=cfg.total_num_envs,
        max_steps_per_rollout_epoch=cfg.max_steps_per_rollout_epoch,
        max_episode_steps=max_episode_steps,
    )

    plan = {
        "config_path": config_path,
        "overrides": overrides,
        "sampled_scene_ids": sampled_scene_ids,
        "episode_sequences": episode_sequences,
    }
    if bucketing_enabled:
        if gt_data is None:
            raise ValueError("Habitat action-length bucketing requires GT data.")
        active_bucket_ids = {
            bucket_id
            for stage in normalized_curriculum_plan["stages"]
            for bucket_id, quota in zip(
                normalized_curriculum_plan["bucket_ids"], stage["quotas"]
            )
            if int(quota) > 0
        }
        buckets, gap_episodes = build_action_length_buckets(
            habitat_dataset.episodes,
            gt_data,
            bucket_ids=normalized_curriculum_plan["bucket_ids"],
            bucket_plans=normalized_curriculum_plan["bucket_plans"],
            bucket_schedule_seed=int(cfg.bucket_schedule_seed),
            minimum_bucket_size=total_num_processes * num_group,
            minimum_bucket_ids=tuple(active_bucket_ids),
        )
        for bucket in buckets:
            bucket_plan = normalized_curriculum_plan["bucket_plans"][bucket.bucket_id]
            bucket_plan["episode_ids"] = list(bucket.episode_ids)
            bucket_plan["episode_count"] = len(bucket.episode_ids)
        normalized_curriculum_plan["dropped_below_global_min"] = len(
            filter_result.dropped_below_ids
        )
        normalized_curriculum_plan["dropped_above_global_max"] = len(
            filter_result.dropped_above_ids
        )
        normalized_curriculum_plan["dropped_in_bucket_gaps"] = len(gap_episodes)
        normalized_curriculum_plan["gap_episode_samples"] = [
            {"episode_id": episode_id, "action_length": action_length}
            for episode_id, action_length in gap_episodes[:50]
        ]
        plan.update(normalized_curriculum_plan)

        if gap_episodes:
            logger.info(
                "Dropped %s Habitat episodes in configured bucket gaps; first "
                "(episode_id, action_length) values: %s",
                len(gap_episodes),
                gap_episodes[:50],
            )
        logger.info(
            "Normalized Habitat UniNaVid bucket plan: %s",
            [
                {
                    "bucket_id": bucket_id,
                    "range": [
                        plan["bucket_plans"][bucket_id]["lower_bound"],
                        plan["bucket_plans"][bucket_id]["upper_bound"],
                    ],
                    "upper_inclusive": plan["bucket_plans"][bucket_id][
                        "upper_inclusive"
                    ],
                    "episode_count": plan["bucket_plans"][bucket_id][
                        "episode_count"
                    ],
                    "max_steps": plan["bucket_plans"][bucket_id]["horizon_steps"],
                }
                for bucket_id in plan["bucket_ids"]
            ],
        )
        for stage in plan["stages"]:
            logger.info(
                "Habitat UniNaVid curriculum stage %s quotas=%s order=%s",
                stage["stage_name"],
                stage["quotas"],
                build_round_robin_bucket_order(plan["bucket_ids"], stage["quotas"]),
            )
    return plan


def build_habitat_overrides(cfg, *, max_episode_steps: int) -> list[str]:
    overrides = [
        f"habitat.dataset.split={cfg.split}",
        f"habitat.dataset.data_path={cfg.data_path}",
        f"habitat.dataset.scenes_dir={cfg.scenes_dir}",
        f"habitat.environment.max_episode_steps={max_episode_steps}",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.iterator_options.group_by_scene=False",
    ]
    gt_path = getattr(cfg, "gt_path", None)
    if gt_path is not None:
        overrides.extend(
            [
                f"habitat.task.measurements.ndtw.SPLIT={cfg.split}",
                f"habitat.task.measurements.ndtw.GT_PATH={gt_path}",
            ]
        )
    rxr_roles = getattr(cfg, "rxr_roles", None)
    if rxr_roles is not None:
        overrides.append(f"+habitat.dataset.ROLES={_format_hydra_list(rxr_roles)}")
    rxr_languages = getattr(cfg, "rxr_languages", None)
    if rxr_languages is not None:
        overrides.append(
            f"+habitat.dataset.LANGUAGES={_format_hydra_list(rxr_languages)}"
        )
    return overrides


def _format_hydra_list(values) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def load_habitat_gt_data(gt_path: str) -> dict:
    with gzip.open(gt_path, "rt", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def get_sampled_habitat_scene_ids(
    episodes, sample_num_scenes: Optional[int], seed: int
):
    if sample_num_scenes is None:
        return None
    scene_ids = list(dict.fromkeys(ep.scene_id for ep in episodes))
    if sample_num_scenes > len(scene_ids):
        raise ValueError(
            f"sample_num_scenes={sample_num_scenes} exceeds available scenes={len(scene_ids)}"
        )
    scene_rng = np.random.default_rng(seed)
    return scene_rng.choice(scene_ids, size=sample_num_scenes, replace=False).tolist()


@dataclass
class NDTWMeasurementConfig(MeasurementConfig):
    type: str = "NDTW"
    SPLIT: str = ""
    GT_PATH: str = ""
    SUCCESS_DISTANCE: float = 3.0
    FDTW: bool = False


ConfigStore.instance().store(
    package="habitat.task.measurements.ndtw",
    group="habitat/task/measurements",
    name="ndtw",
    node=NDTWMeasurementConfig,
)


@registry.register_task_action
class NoOpAction(SimulatorTaskAction):
    """Register manually defined No-operation action for habitat env."""

    def step(self, *args, **kwargs):
        return self._sim.get_sensor_observations()


class HabitatEnv(gym.Env):
    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes, worker_info=None
    ):
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = cfg.max_episode_steps
        self.ignore_terminations = cfg.ignore_terminations
        self.first_done_cached_mask = np.zeros(self.num_envs, dtype=bool)
        self.episode_info = None
        self._active_bucket_id = None

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self._init_env()

        self.metrics_cfg = cfg.metrics_cfg
        self.video_cfg = cfg.video_cfg
        self.render_images = {}
        self.current_raw_obs = None

        self.env_config = self.env.get_env_attr("config")[0]
        self.initial_distance_to_goal = np.full(self.num_envs, np.nan, dtype=np.float32)
        self.reward_mode = getattr(self.cfg, "reward_mode", "weighted_success_ndtw")
        self.subgoal_reward = None
        if self.reward_mode == "subgoal_progress":
            self.subgoal_reward = SubgoalRewardTracker(
                num_envs=self.num_envs,
                config=SubgoalRewardConfig(
                    progress_reward_coef=float(self.cfg.progress_reward_coef),
                    subgoal_success_reward_coef=float(
                        self.cfg.subgoal_success_reward_coef
                    ),
                    subgoal_switch_distance=float(self.cfg.subgoal_switch_distance),
                    subgoal_success_distance=float(self.cfg.subgoal_success_distance),
                    final_success_distance=float(self.cfg.final_success_distance),
                    stop_success_reward_coef=float(self.cfg.stop_success_reward_coef),
                    premature_stop_coeff=float(self.cfg.premature_stop_coeff),
                    failure_stop_coeff=float(self.cfg.failure_stop_coeff),
                    stall_patience=int(self.cfg.stall_patience),
                    stall_penalty_coeff=float(self.cfg.stall_penalty_coeff),
                    stall_recovery_patience=int(self.cfg.stall_recovery_patience),
                    stall_observation_patience=int(self.cfg.stall_observation_patience),
                    step_cost_coeff=float(getattr(self.cfg, "step_cost_coeff", 0.0)),
                ),
            )

        self.action_map = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
            4: "no_op",
        }

    def activate_bucket(
        self,
        bucket_id: str,
        horizon_steps: int,
        group_episode_ids: list[str] | tuple[str, ...],
    ) -> None:
        if not bool(getattr(self.cfg, "action_length_bucketing", False)):
            raise RuntimeError("Habitat action-length bucketing is not enabled.")
        global_plan = getattr(self.cfg, "global_plan", None)
        if global_plan is None or "bucket_plans" not in global_plan:
            raise RuntimeError("Habitat bucket plan is unavailable.")
        if bucket_id not in global_plan["bucket_plans"]:
            raise KeyError(f"Unknown Habitat action-length bucket: {bucket_id}")
        bucket_plan = global_plan["bucket_plans"][bucket_id]
        if int(bucket_plan["horizon_steps"]) != int(horizon_steps):
            raise ValueError(
                f"Bucket {bucket_id} horizon mismatch: expected "
                f"{bucket_plan['horizon_steps']}, got {horizon_steps}."
            )
        if len(group_episode_ids) != self.num_group:
            raise ValueError(
                f"Bucket {bucket_id} activation requires {self.num_group} group "
                f"episode IDs, got {len(group_episode_ids)}."
            )
        known_episode_ids = set(bucket_plan["episode_ids"])
        unknown_episode_ids = [
            str(episode_id)
            for episode_id in group_episode_ids
            if str(episode_id) not in known_episode_ids
        ]
        if unknown_episode_ids:
            raise ValueError(
                f"Bucket {bucket_id} activation contains unknown episode IDs: "
                f"{unknown_episode_ids[:10]}."
            )
        self._active_bucket_id = bucket_id
        self.max_episode_steps = int(horizon_steps)
        process_group_episode_ids = [
            [str(episode_id)] for episode_id in group_episode_ids
        ]
        self.env.activate_episode_ids(
            self._expand_group_episode_ids(process_group_episode_ids)
        )
        self._elapsed_steps.fill(0)
        self.first_done_cached_mask.fill(False)
        self.current_raw_obs = None

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_actions = np.vectorize(lambda x: self.action_map[x])(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i].copy(),
                auto_reset=False,
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        self._attach_rgb_frame_history(obs_list)

        if past_dones.any() and self.auto_reset:
            final_obs = obs_list[-1]
            final_info = infos_list[-1]
            reset_obs, reset_infos = self._handle_auto_reset(
                past_dones.cpu().numpy(),
                final_obs,
                final_info,
            )
            self._update_rgb_frame_history(
                final_obs,
                reset_obs,
                past_dones.cpu().numpy(),
            )
            obs_list[-1], infos_list[-1] = reset_obs, reset_infos

        # [num_envs, chunk_steps]
        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = raw_chunk_terminations.any(dim=1)

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = raw_chunk_truncations.any(dim=1)
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def step(self, actions=None, auto_reset=True):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        self._elapsed_steps += 1

        # Habitat cannot execute STOP and continue stepping the same episode, so forward
        # it as no_op to the simulator while still marking the policy action as terminal.
        env_actions = actions.astype("U12").copy()
        is_stop = env_actions == "stop"
        env_actions[is_stop] = "no_op"

        raw_obs, _reward, terminations, info_lists = self.env.step(
            self._format_habitat_actions(env_actions)
        )

        # If some envs execute "no_op", manually normalize depth observations
        # according to Habitat's depth sensor config.
        self._normalize_depth(env_actions, raw_obs)

        # TODO: what if termination means failure? (e.g. robot falling down)
        terminations[is_stop] = True
        truncations = self._elapsed_steps >= self.max_episode_steps
        done_mask = terminations | truncations
        first_done_mask = done_mask & (~self.first_done_cached_mask)
        first_done_reward_mask = first_done_mask & terminations & (~truncations)
        valid_reward_mask = ~self.first_done_cached_mask

        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(infos, terminations, first_done_mask)
        step_reward = self._calc_step_reward(
            infos["episode"],
            first_done_reward_mask,
            is_stop=is_stop,
            truncations=truncations,
            valid_reward_mask=valid_reward_mask,
            infos=infos,
        )
        if self.metrics_cfg.save_metrics:
            self._write_first_done_metrics(self.episode_info, first_done_mask)
        self.first_done_cached_mask[first_done_mask] = True

        self.current_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs, info_lists)
        video_cfg = video.get_video_cfg(self)
        if video.should_save_rollout_video(video_cfg):
            episode_ids = self.env.get_current_episode_metadata()["episode_id"]
            video.record_rollout_video_frames(self.render_images, obs, episode_ids)
            video.flush_rollout_videos(
                self.render_images,
                first_done_mask,
                infos,
                episode_ids,
                video_cfg,
            )

        if self.ignore_terminations:
            terminations[:] = False

        dones = terminations | truncations
        if dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        raw_obs = self.env.reset(env_idx)
        self._elapsed_steps[env_idx] = 0
        self.first_done_cached_mask[env_idx] = False
        self.initial_distance_to_goal[env_idx] = np.nan
        current_metrics = self.env.get_current_metrics(env_idx)
        distance_to_goal = current_metrics.get("distance_to_goal", None)
        if distance_to_goal is not None:
            distance_to_goal = np.asarray(distance_to_goal, dtype=np.float32)
            self.initial_distance_to_goal[env_idx] = distance_to_goal
        self._reset_subgoal_reward_state(env_idx)
        if self.episode_info is not None:
            device = next(iter(self.episode_info.values())).device
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            mask[env_idx] = True
            for v in self.episode_info.values():
                v[mask] = 0
        infos = {}

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]
        obs = self._wrap_obs(self.current_raw_obs)
        self._attach_rgb_frame_history([obs])

        return obs, infos

    def update_reset_state_ids(self):
        pass

    def _reset_subgoal_reward_state(self, env_idx):
        if getattr(self, "subgoal_reward", None) is None:
            return
        metadata = self.env.get_current_episode_subgoal_distances(id=env_idx)
        self.subgoal_reward.reset(
            env_idx,
            distances_to_subgoals=metadata["distances_to_subgoals"],
            distances_to_final_goal=metadata["distance_to_final_goal"],
        )

    def _format_habitat_actions(self, actions):
        formatted_actions = []
        for action in actions:
            action_array = np.asarray(action)
            if action_array.shape:
                if action_array.size != 1:
                    raise ValueError(
                        "Habitat navigation expects one discrete action per env."
                    )
                action = action_array.reshape(-1)[0]
            formatted_actions.append({"action": str(action)})
        return formatted_actions

    def _normalize_depth(self, actions, raw_obs):
        """Normalize depth for envs whose action is 'no_op', following
        Habitat's depth sensor configuration.
        """
        is_no_op = actions == "no_op"
        if not np.any(is_no_op):
            return

        depth_cfg = self.env_config.simulator.agents.main_agent.sim_sensors.depth_sensor
        if not getattr(depth_cfg, "normalize_depth", False):
            return

        min_depth = float(depth_cfg.min_depth)
        max_depth = float(depth_cfg.max_depth)

        for env_idx, flag_no_op in enumerate(is_no_op):
            if not flag_no_op:
                continue
            obs = raw_obs[env_idx]
            if "depth" not in obs:
                continue
            depth = obs["depth"]
            depth = np.clip(depth, min_depth, max_depth)
            depth = (depth - min_depth) / (max_depth - min_depth)
            if depth.ndim == 2:
                depth = depth[..., None]
            obs["depth"] = depth
            raw_obs[env_idx] = obs

    def _wrap_obs(self, obs_list, info_lists=None):
        image_list = []
        task_descs = []
        token_list = []
        episode_ids = self.env.get_current_episode_metadata()["episode_id"]
        video_cfg = self.cfg.video_cfg
        should_render_video = info_lists is not None and video.cfg_get(
            video_cfg, "save_video", False
        )

        for i in range(len(obs_list)):
            image = {}
            obs = obs_list[i]
            info = info_lists[i] if info_lists is not None else None

            image["rgb"] = obs["rgb"][:, :, :3]
            if self.cfg.model_type == "cma":
                image["depth"] = obs["depth"]
            if should_render_video:
                image["main_images"] = video.build_rollout_video_frame(obs, info)

            inst = str(obs["instruction"].get("text", ""))
            # token is used for CMA algorithm, please refer to
            # https://github.com/jacobkrantz/VLN-CE for more details.
            token = obs["instruction"].get("tokens", [])

            image_list.append(image)
            task_descs.append(inst)
            token_list.append(token)
        image_tensor = to_tensor(list_of_dict_to_dict_of_list(image_list))

        obs = {}
        if should_render_video:
            obs["main_images"] = image_tensor["main_images"].clone()  # [N_ENV, H, W, C]
        obs["wrist_images"] = image_tensor[
            "rgb"
        ].clone()  # Temporarily use wrist_images to store rgb images
        if "depth" in image_tensor:
            depth_tensor = image_tensor["depth"].clone()
            obs["extra_view_images"] = depth_tensor.unsqueeze(1)  # [N_ENV, 1, H, W, C]
        if self.cfg.model_type == "cma":
            obs["task_descriptions"] = token_list
        else:
            obs["task_descriptions"] = task_descs
        obs["states"] = torch.tensor([int(episode_id) for episode_id in episode_ids])

        return obs

    def _attach_rgb_frame_history(self, obs_list):
        if getattr(self.cfg, "model_type", None) != "uninavid":
            return

        frame_history = torch.stack(
            [obs["wrist_images"] for obs in obs_list],
            dim=1,
        )
        obs_list[-1]["rgb_frame_history"] = frame_history
        obs_list[-1]["rgb_frame_history_lengths"] = torch.full(
            (frame_history.shape[0],),
            frame_history.shape[1],
            dtype=torch.long,
            device=frame_history.device,
        )

    def _update_rgb_frame_history(self, final_obs, reset_obs, dones):
        if getattr(self.cfg, "model_type", None) != "uninavid":
            return

        done_mask = torch.as_tensor(
            dones,
            dtype=torch.bool,
            device=final_obs["rgb_frame_history"].device,
        )
        if done_mask.any() and "wrist_images" in reset_obs:
            reset_frames = reset_obs["wrist_images"].to(
                device=final_obs["rgb_frame_history"].device
            )
            history = final_obs["rgb_frame_history"]
            history[done_mask] = (
                reset_frames[done_mask]
                .unsqueeze(1)
                .expand(
                    -1,
                    history.shape[1],
                    *history.shape[2:],
                )
            )
            final_obs["rgb_frame_history_lengths"][done_mask] = 1
        reset_obs["rgb_frame_history"] = final_obs["rgb_frame_history"]
        reset_obs["rgb_frame_history_lengths"] = final_obs["rgb_frame_history_lengths"]

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(
        self,
        episode,
        first_done_reward_mask,
        is_stop=None,
        truncations=None,
        valid_reward_mask=None,
        infos=None,
    ):
        if self.reward_mode == "subgoal_progress":
            return self._calc_subgoal_progress_reward(
                is_stop=is_stop,
                truncations=truncations,
                valid_reward_mask=valid_reward_mask,
                infos=infos,
            )
        return self._calc_weighted_success_ndtw_reward(
            episode,
            first_done_reward_mask,
        )

    def _calc_weighted_success_ndtw_reward(self, episode, first_done_reward_mask):
        device = episode["success"].device
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        first_done_reward_mask = torch.as_tensor(
            first_done_reward_mask,
            dtype=torch.bool,
            device=device,
        )
        if not first_done_reward_mask.any():
            return reward

        success = episode["success"].to(dtype=torch.float32)
        distance_to_goal = episode["distance_to_goal"].to(dtype=torch.float32)
        ndtw = episode["ndtw"].to(dtype=torch.float32)
        success_distance = float(
            self.env_config.task.measurements.success.success_distance
        )

        success_scale = 1.0 - torch.minimum(
            distance_to_goal / success_distance,
            torch.ones_like(distance_to_goal),
        )
        success_reward = (
            success * float(self.cfg.success_reward_coef) * (1 + success_scale)
        )
        ndtw_reward = ndtw * float(self.cfg.ndtw_reward_coef)
        reward[first_done_reward_mask] = (
            success_reward[first_done_reward_mask] + ndtw_reward[first_done_reward_mask]
        )
        return reward

    def _calc_subgoal_progress_reward(
        self,
        is_stop,
        truncations,
        valid_reward_mask,
        infos,
    ):
        if self.subgoal_reward is None:
            raise RuntimeError("subgoal_progress reward mode requires subgoal_reward.")
        if is_stop is None:
            raise RuntimeError("subgoal_progress reward mode requires is_stop.")
        if valid_reward_mask is None:
            raise RuntimeError(
                "subgoal_progress reward mode requires valid_reward_mask."
            )

        metadata = self.env.get_current_episode_subgoal_distances()
        reward, components = self.subgoal_reward.compute_step(
            distances_to_subgoals=metadata["distances_to_subgoals"],
            distances_to_final_goal=metadata["distance_to_final_goal"],
            is_stop=is_stop,
            is_truncated=truncations,
            valid_mask=valid_reward_mask,
        )
        if infos is not None:
            self._attach_subgoal_reward_metrics(infos, components, valid_reward_mask)
        return to_tensor(reward)

    def _attach_subgoal_reward_metrics(self, infos, components, valid_reward_mask):
        valid_reward_mask = np.asarray(valid_reward_mask, dtype=bool)
        if self.episode_info is None:
            self.episode_info = {}
        for key, value in components.items():
            tensor_value = to_tensor(value)
            if key not in self.episode_info:
                self.episode_info[key] = torch.zeros_like(tensor_value)
            mask = torch.as_tensor(
                valid_reward_mask,
                dtype=torch.bool,
                device=self.episode_info[key].device,
            )
            self.episode_info[key][mask] = tensor_value.to(
                device=self.episode_info[key].device
            )[mask]
        infos["episode"] = {k: v.clone() for k, v in self.episode_info.items()}

    def _record_metrics(self, infos, terminations, first_done_mask):
        episode_info = {}
        dist_threshold = self.env_config.task.measurements.success.success_distance
        terminations = np.array(terminations, dtype=bool, copy=True)
        first_done_mask = np.asarray(first_done_mask, dtype=bool)

        episode_info["distance_to_goal"] = np.asarray(
            infos["distance_to_goal"], dtype=np.float32
        )

        episode_info["ndtw"] = np.asarray(infos["ndtw"], dtype=np.float32)

        needs_seed = np.isnan(self.initial_distance_to_goal)
        if needs_seed.any():
            self.initial_distance_to_goal[needs_seed] = episode_info[
                "distance_to_goal"
            ][needs_seed].copy()

        episode_info["success"] = (
            terminations & (episode_info["distance_to_goal"] < dist_threshold)
        ).astype(np.float32)

        episode_info["trajectory_Length"] = np.asarray(
            infos["trajectory_Length"], dtype=np.float32
        )

        episode_info["spl"] = episode_info["success"] * (
            self.initial_distance_to_goal
            / np.maximum(
                episode_info["trajectory_Length"], self.initial_distance_to_goal
            )
        )

        episode_info["oracle_success"] = np.asarray(
            infos["oracle_success"], dtype=np.float32
        )

        episode_info["oracle_navigation_error"] = np.asarray(
            infos["oracle_navigation_error"], dtype=np.float32
        )

        latest_episode = to_tensor(episode_info)
        if self.episode_info is None:
            self.episode_info = {
                k: torch.zeros_like(v) for k, v in latest_episode.items()
            }

        update_mask = torch.as_tensor(~self.first_done_cached_mask, dtype=torch.bool)
        for k, v in latest_episode.items():
            mask = update_mask.to(device=v.device)
            self.episode_info[k][mask] = v[mask]

        infos["episode"] = {k: v.clone() for k, v in self.episode_info.items()}

        return infos

    def _write_first_done_metrics(self, episode, first_done_mask):
        episode_ids = self.env.get_current_episode_metadata()["episode_id"]
        for i in range(len(first_done_mask)):
            if not first_done_mask[i]:
                continue

            episode_id = episode_ids[i]
            metrics_dict = {}
            for k, v in episode.items():
                if torch.is_tensor(v):
                    if v.dim() == 1:
                        metrics_dict[k] = v[i].item()
                    else:
                        metrics_dict[k] = v[i].cpu().numpy().tolist()
                elif isinstance(v, (list, np.ndarray)):
                    metrics_dict[k] = v[i]
                else:
                    metrics_dict[k] = v
            metrics_dict["elapsed_steps"] = int(self._elapsed_steps[i])
            metrics_file = os.path.join(
                self.metrics_cfg.metrics_base_dir, f"episode_{episode_id}.json"
            )
            os.makedirs(self.metrics_cfg.metrics_base_dir, exist_ok=True)
            if os.path.exists(metrics_file):
                continue
            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    def _init_env(self):
        env_fns = self._get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def _get_env_fns(self):
        env_fn_params = self._get_env_fn_params()
        env_fns = []

        for param in env_fn_params:

            def env_fn(p=param):
                return make_habitat_env(p)

            env_fns.append(env_fn)

        return env_fns

    def _get_env_fn_params(self):
        env_fn_params = []
        global_plan = getattr(self.cfg, "global_plan", None)
        if global_plan is None:
            global_plan = build_habitat_global_plan(
                self.cfg,
                num_group=self.num_group,
                total_num_processes=self.total_num_processes,
                max_episode_steps=self.max_episode_steps,
            )

        config_path = global_plan["config_path"]
        overrides = list(global_plan["overrides"])
        episode_sequences = global_plan["episode_sequences"]
        process_group_episode_ids = episode_sequences[self.seed_offset]
        env_episode_ids = self._expand_group_episode_ids(process_group_episode_ids)

        for env_id, assigned_ids in enumerate(env_episode_ids):
            env_fn_params.append(
                {
                    "config_path": config_path,
                    "overrides": overrides,
                    "episode_ids": list(assigned_ids),
                    "seed": self.seed + env_id,
                }
            )

        return env_fn_params

    def _expand_group_episode_ids(self, process_group_episode_ids):
        return [
            list(process_group_episode_ids[env_id // self.group_size])
            for env_id in range(self.num_envs)
        ]
