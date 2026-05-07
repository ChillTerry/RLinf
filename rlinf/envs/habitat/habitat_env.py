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
import json
import logging
import os
from typing import Optional, Union

import cv2
import gym
import habitat
import numpy as np
import torch
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat_baselines.config.default import get_config
from hydra.core.global_hydra import GlobalHydra

from rlinf.envs.habitat.extensions import measures
from rlinf.envs.habitat.extensions.allocator import vram_balance_episode_ids
from rlinf.envs.habitat.extensions.utils import observations_to_image
from rlinf.envs.habitat.venv import HabitatRLEnv, ReconfigureSubprocEnv
from rlinf.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)

measures.pass_format_check()

logger = logging.getLogger(__name__)


def _clone_habitat_chunk_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _clone_habitat_chunk_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_clone_habitat_chunk_value(v) for v in value)
    return copy.deepcopy(value)


def _masked_update_habitat_chunk_value(dst, src, mask):
    if src is None:
        return dst

    if isinstance(src, torch.Tensor):
        src = src.clone()
        if dst is None:
            return src
        dst = dst.clone()
        if src.ndim > 0 and dst.ndim > 0 and src.shape[0] == mask.shape[0]:
            dst[mask] = src[mask]
            return dst
        return src

    if isinstance(src, np.ndarray):
        src = src.copy()
        if dst is None:
            return src
        dst = np.array(dst, copy=True)
        mask_np = mask.detach().cpu().numpy()
        if src.ndim > 0 and dst.ndim > 0 and src.shape[0] == mask_np.shape[0]:
            dst[mask_np] = src[mask_np]
            return dst
        return src

    if isinstance(src, dict):
        dst_dict = {} if not isinstance(dst, dict) else _clone_habitat_chunk_value(dst)
        for key, value in src.items():
            dst_dict[key] = _masked_update_habitat_chunk_value(
                dst_dict.get(key), value, mask
            )
        return dst_dict

    if isinstance(src, (list, tuple)):
        src_seq = [_clone_habitat_chunk_value(v) for v in src]
        if len(src_seq) == mask.shape[0]:
            if isinstance(dst, (list, tuple)) and len(dst) == len(src_seq):
                dst_seq = [_clone_habitat_chunk_value(v) for v in dst]
            else:
                dst_seq = [_clone_habitat_chunk_value(v) for v in src_seq]
            mask_np = mask.detach().cpu().numpy()
            for idx, value in enumerate(src_seq):
                if mask_np[idx]:
                    dst_seq[idx] = value
            return type(src)(dst_seq)

        if dst is None:
            return type(src)(src_seq)
        dst_seq = list(dst)
        for idx, value in enumerate(src_seq):
            if idx >= len(dst_seq):
                dst_seq.append(value)
            else:
                dst_seq[idx] = _masked_update_habitat_chunk_value(
                    dst_seq[idx], value, mask
                )
        return type(src)(dst_seq)

    return _clone_habitat_chunk_value(src)


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
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.auto_reset = cfg.auto_reset
        self.max_episode_steps = cfg.max_episode_steps
        self.ignore_terminations = cfg.ignore_terminations
        self.dones_once = np.zeros(self.num_envs, dtype=bool)
        self.first_done_infos = None

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self._init_env()

        self.metrics_cfg = cfg.metrics_cfg
        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None

        self.env_config = self.env.get_env_attr("config")[0]
        self.initial_distance_to_goal = np.full(self.num_envs, np.nan, dtype=np.float32)

        self.action_map = {
            0: "stop",
            1: "move_forward",
            2: "turn_left",
            3: "turn_right",
            4: "no_op",
        }

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _habitat_model_type(self):
        return getattr(getattr(self, "cfg", None), "model_type", None)

    def _uninavid_use_raw_rgb_enabled(self):
        return self._habitat_model_type() == "uninavid" and bool(
            getattr(self.cfg, "uninavid_use_raw_rgb", False)
        )

    def _attach_uninavid_chunk_history(self, obs_list):
        if self._habitat_model_type() != "uninavid":
            return
        if not obs_list:
            return
        if any("wrist_images" not in obs for obs in obs_list):
            return
        obs_list[-1]["wrist_images"] = torch.stack(
            [obs["wrist_images"] for obs in obs_list],
            dim=1,
        )

    @staticmethod
    def _format_habitat_actions(actions):
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

    @staticmethod
    def _squeeze_singleton_action_dim(actions):
        actions = np.asarray(actions)
        if actions.ndim > 1 and actions.shape[-1] == 1:
            return np.squeeze(actions, axis=-1)
        return actions

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_actions = np.vectorize(lambda x: self.action_map[x])(chunk_actions)
        chunk_actions = self._squeeze_singleton_action_dim(chunk_actions)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        aggregated_final_info = None
        aggregated_final_obs = None
        for i in range(chunk_size):
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i].copy()
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

            step_dones = torch.logical_or(terminations, truncations)
            if step_dones.any() and self.auto_reset:
                aggregated_final_info = _masked_update_habitat_chunk_value(
                    aggregated_final_info,
                    infos.get("final_info", infos),
                    step_dones,
                )
                aggregated_final_obs = _masked_update_habitat_chunk_value(
                    aggregated_final_obs,
                    infos.get("final_observation", extracted_obs),
                    step_dones,
                )

        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
        if past_dones.any() and self.auto_reset:
            infos_list[-1] = dict(infos_list[-1])
            final_info = _clone_habitat_chunk_value(
                infos_list[-1].get("final_info", infos_list[-1])
            )
            if aggregated_final_info is not None:
                final_info = _masked_update_habitat_chunk_value(
                    final_info, aggregated_final_info, past_dones
                )

            final_observation = _clone_habitat_chunk_value(obs_list[-1])
            if aggregated_final_obs is not None:
                final_observation = _masked_update_habitat_chunk_value(
                    final_observation, aggregated_final_obs, past_dones
                )

            infos_list[-1]["final_info"] = final_info
            infos_list[-1]["final_observation"] = final_observation
            infos_list[-1]["_final_info"] = past_dones.clone()
            infos_list[-1]["_final_observation"] = past_dones.clone()
            infos_list[-1]["_elapsed_steps"] = past_dones.clone()

        self._attach_uninavid_chunk_history(obs_list)

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

    def step(self, actions=None):
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = self._squeeze_singleton_action_dim(actions)
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

        terminations[is_stop] = True
        # TODO: what if termination means failure? (e.g. robot falling down)
        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(infos, terminations)
        step_reward = self._calc_step_reward(infos["episode"]["success"])

        truncations = self.elapsed_steps >= self.max_episode_steps
        dones_for_metric_save = terminations | truncations
        # Only save episode metrics once: at the first time an env becomes done.
        metric_save_masks = dones_for_metric_save & (~self.dones_once)
        if metric_save_masks.any():
            self._save_metrics(infos, metric_save_masks)

        self._overlay_first_done_episode_metrics(infos)

        self.current_raw_obs = raw_obs
        obs = self._wrap_obs(raw_obs, info_lists)

        if self.ignore_terminations:
            terminations[:] = False

        dones = terminations | truncations
        if dones.any() and self.auto_reset:
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
        self.prev_step_reward[env_idx] = 0.0
        self.dones_once[env_idx] = False
        self.initial_distance_to_goal[env_idx] = np.nan
        current_metrics = self.env.get_current_metrics(env_idx)
        distance_to_goal = current_metrics.get("distance_to_goal", None)
        if distance_to_goal is not None:
            distance_to_goal = np.asarray(distance_to_goal, dtype=np.float32)
            self.initial_distance_to_goal[env_idx] = distance_to_goal
        if self.first_done_infos is not None and "episode" in self.first_done_infos:
            episode = self.first_done_infos["episode"]
            device = next(iter(episode.values())).device
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            mask[env_idx] = True
            for v in episode.values():
                v[mask] = 0
        infos = {}

        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs

        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]
        obs = self._wrap_obs(self.current_raw_obs)

        return obs, infos

    def update_reset_state_ids(self):
        pass

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
            obs["depth"] = depth
            raw_obs[env_idx] = obs

    def _wrap_obs(self, obs_list, info_lists=None):
        image_list = []
        raw_rgb_list = []
        task_descs = []
        token_list = []
        should_render_video = info_lists is not None and self.cfg.video_cfg.save_video
        for i in range(len(obs_list)):
            obs = obs_list[i]
            info = info_lists[i] if info_lists is not None else None
            if should_render_video:
                images = observations_to_image(obs, info)
                if "top_down_map" in images:
                    image_size = (images["rgb"].shape[1], images["rgb"].shape[0])
                    images["top_down_map"] = cv2.resize(
                        images["top_down_map"],
                        dsize=image_size,
                        interpolation=cv2.INTER_LINEAR,
                    )
                    images["concat"] = np.concatenate(
                        (images["rgb"], images["top_down_map"]), axis=1
                    )
                else:
                    images["concat"] = images["rgb"]
            else:
                images = observations_to_image(obs)
            inst = str(obs["instruction"].get("text", ""))
            # token is used for CMA algorithm, please refer to
            # https://github.com/jacobkrantz/VLN-CE for more details.
            token = obs["instruction"].get("tokens", [])
            image_list.append(images)
            raw_rgb_list.append(obs["rgb"])
            task_descs.append(inst)
            token_list.append(token)
        image_tensor = to_tensor(list_of_dict_to_dict_of_list(image_list))
        raw_rgb_tensor = None
        if self._uninavid_use_raw_rgb_enabled():
            raw_rgb_tensor = to_tensor(raw_rgb_list)

        episode_ids = self.env.get_current_episode_metadata()["episode_id"]

        obs = {}
        if should_render_video:
            obs["main_images"] = image_tensor["concat"].clone()  # [N_ENV, H, W, C]
        if raw_rgb_tensor is not None:
            obs["wrist_images"] = raw_rgb_tensor.clone()
        else:
            obs["wrist_images"] = image_tensor[
                "rgb"
            ].clone()  # Temporarily use wrist_images to store rgb images
        if "depth" in image_tensor:
            depth_tensor = image_tensor["depth"].clone()
            obs["extra_view_images"] = depth_tensor.unsqueeze(1)  # [N_ENV, 1, H, W, C]
        if self._habitat_model_type() == "cma":
            obs["task_descriptions"] = token_list
        else:
            obs["task_descriptions"] = task_descs
        obs["states"] = torch.tensor([int(episode_id) for episode_id in episode_ids])

        return obs

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

    def _calc_step_reward(self, success):
        reward = self.cfg.reward_coef * success
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _record_metrics(self, infos, terminations):
        episode_info = {}
        dist_threshold = self.env_config.task.measurements.success.success_distance
        terminations = np.array(terminations, dtype=bool, copy=True)

        episode_info["distance_to_goal"] = np.array(
            infos["distance_to_goal"], dtype=np.float32
        ).copy()

        # Record initial distance to goal at the first step of each episode
        is_first_step = self._elapsed_steps == 1
        needs_seed = is_first_step & np.isnan(self.initial_distance_to_goal)
        if needs_seed.any():
            self.initial_distance_to_goal[needs_seed] = episode_info[
                "distance_to_goal"
            ][needs_seed].copy()

        episode_info["success"] = (
            terminations & (episode_info["distance_to_goal"] < dist_threshold)
        ).astype(np.float32)

        episode_info["trajectory_Length"] = np.array(
            infos["trajectory_Length"], dtype=np.float32
        ).copy()

        episode_info["spl"] = episode_info["success"] * (
            self.initial_distance_to_goal
            / np.maximum(
                episode_info["trajectory_Length"], self.initial_distance_to_goal
            )
        )

        episode_info["oracle_success"] = infos["oracle_success"].copy()

        episode_info["oracle_navigation_error"] = infos[
            "oracle_navigation_error"
        ].copy()

        infos["episode"] = to_tensor(episode_info)

        return infos

    def _save_metrics(self, infos, metric_save_masks):
        """Save metrics by episode_id when env first done."""
        mask = torch.from_numpy(metric_save_masks)  # [num_envs]
        self.dones_once[metric_save_masks] = True
        episode = infos["episode"]

        if self.first_done_infos is None:
            self.first_done_infos = {
                "episode": {k: torch.zeros_like(v) for k, v in episode.items()}
            }

        # Update the envs that become done in this step
        for k, v in episode.items():
            cached_v = self.first_done_infos["episode"][k]
            m = mask.to(v.device)
            cached_v[m] = v[m]

        # Save metrics by episode_id when env first done
        if self.metrics_cfg.save_metrics:
            episode_ids = self.env.get_current_episode_metadata()["episode_id"]
            for i in range(len(metric_save_masks)):
                if metric_save_masks[i]:
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
                    metrics_file = os.path.join(
                        self.metrics_cfg.metrics_base_dir, f"episode_{episode_id}.json"
                    )
                    os.makedirs(self.metrics_cfg.metrics_base_dir, exist_ok=True)
                    if os.path.exists(metrics_file):
                        continue
                    with open(metrics_file, "w") as f:
                        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    def _overlay_first_done_episode_metrics(self, infos):
        """Overlay cached first-done values onto live episode metrics"""
        if (
            not self.dones_once.any()
            or not isinstance(infos, dict)
            or not isinstance(self.first_done_infos, dict)
        ):
            return

        live_episode = infos.get("episode")
        cached_episode = self.first_done_infos.get("episode")
        if not isinstance(live_episode, dict) or not isinstance(cached_episode, dict):
            return

        done_once_mask = torch.from_numpy(self.dones_once)
        shared_keys = set(live_episode.keys()) & set(cached_episode.keys())
        for key in shared_keys:
            live_v = live_episode[key]
            cached_v = cached_episode[key]
            if not (torch.is_tensor(live_v) and torch.is_tensor(cached_v)):
                continue
            if live_v.shape[0] != self.num_envs or cached_v.shape[0] != self.num_envs:
                continue

            mask = done_once_mask.to(device=live_v.device)
            cached_v = cached_v.to(device=live_v.device, dtype=live_v.dtype)
            live_v[mask] = cached_v[mask]

    def _init_env(self):
        env_fns = self._get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def _get_env_fns(self):
        env_fn_params = self._get_env_fn_params()
        env_fns = []

        for param in env_fn_params:

            def env_fn(p=param):
                config_path = p["config_path"]
                overrides = p["overrides"]
                episode_ids = p["episode_ids"]
                seed = p["seed"]

                config = get_config(config_path, overrides=overrides)

                dataset = habitat.datasets.make_dataset(
                    config.habitat.dataset.type,
                    config=config.habitat.dataset,
                )

                episodes_by_id = {
                    episode.episode_id: episode for episode in dataset.episodes
                }
                dataset.episodes = [
                    episodes_by_id[episode_id] for episode_id in episode_ids
                ]

                env = HabitatRLEnv(config=config, dataset=dataset)
                env.seed(seed)
                return env

            env_fns.append(env_fn)

        return env_fns

    def _get_env_fn_params(self):
        env_fn_params = []

        # Habitat uses hydra to load the config,
        # but the hydra maybe initialized somewhere else,
        # so we need to clear it to avoid conflicts
        hydra_initialized = GlobalHydra.instance().is_initialized()
        if hydra_initialized:
            GlobalHydra.instance().clear()

        config_path = self.cfg.init_params.config_path
        overrides = [
            f"habitat.dataset.split={self.cfg.split}",
            f"habitat.dataset.data_path={self.cfg.data_path}",
            f"habitat.dataset.scenes_dir={self.cfg.scenes_dir}",
            f"habitat.environment.max_episode_steps={self.max_episode_steps}",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.group_by_scene=False",
        ]
        habitat_config = get_config(config_path, overrides=overrides)

        habitat_dataset = habitat.datasets.make_dataset(
            habitat_config.habitat.dataset.type,
            config=habitat_config.habitat.dataset,
        )

        self._sample_habitat_dataset_scenes(
            habitat_dataset,
            getattr(self.cfg, "sample_num_scenes", None),
            self.cfg.seed,
        )

        # Load episodes to GPUs in a balanced way according to scene vram profile
        process_group_episode_ids = vram_balance_episode_ids(
            habitat_dataset.episodes,
            auto_reset=self.auto_reset,
            total_num_processes=self.total_num_processes,
            num_group=self.num_group,
            total_num_envs=self.cfg.total_num_envs,
            max_steps_per_rollout_epoch=self.cfg.max_steps_per_rollout_epoch,
            max_episode_steps=self.max_episode_steps,
            seed_offset=self.seed_offset,
        )

        for env_id in range(self.num_envs):
            group_id = env_id // self.group_size
            assigned_ids = list(process_group_episode_ids[group_id])

            env_fn_params.append(
                {
                    "config_path": config_path,
                    "overrides": overrides,
                    "episode_ids": assigned_ids,
                    "seed": self.seed + env_id,
                }
            )

        return env_fn_params

    def _sample_habitat_dataset_scenes(
        self,
        habitat_dataset,
        sample_num_scenes: Optional[int],
        seed: int,
    ) -> None:
        """Subsample episodes to those belonging to a random subset of scenes."""
        if sample_num_scenes is None:
            return
        scene_ids = list(dict.fromkeys(ep.scene_id for ep in habitat_dataset.episodes))
        if sample_num_scenes > len(scene_ids):
            raise ValueError(
                f"sample_num_scenes={sample_num_scenes} exceeds available scenes={len(scene_ids)}"
            )
        scene_rng = np.random.default_rng(seed)
        sampled_scene_ids = set(
            scene_rng.choice(scene_ids, size=sample_num_scenes, replace=False).tolist()
        )
        habitat_dataset.episodes = [
            ep for ep in habitat_dataset.episodes if ep.scene_id in sampled_scene_ids
        ]
        logger.info(
            f"[HabitatEnv] sampled {sample_num_scenes}/{len(scene_ids)} scenes "
            f"with seed={seed}, kept {len(habitat_dataset.episodes)} episodes"
        )
