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
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("habitat")
pytest.importorskip("habitat_sim")
pytest.importorskip("habitat_baselines")

from rlinf.envs.habitat.habitat_env import HabitatEnv

VLN_CE_ROOT = Path("/data/vln-dependency/VLN-CE")
R2R_ROOT = VLN_CE_ROOT / "datasets/r2r"
SCENES_DIR = VLN_CE_ROOT / "scene_dataset"
PNG_OUTPUT_DIR = Path("/tmp/rlinf-habitat-cma-depth-test/png")


def _skip_if_habitat_assets_missing():
    required_paths = [
        R2R_ROOT / "val_seen/val_seen.json.gz",
        R2R_ROOT / "val_seen/val_seen_gt.json.gz",
        SCENES_DIR / "mp3d",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        pytest.skip("Missing VLN-CE Habitat test assets: " + ", ".join(missing_paths))


def _make_habitat_cfg(
    *,
    model_type="cma",
    config_path="rlinf/envs/habitat/extensions/config/vlnce_r2r_cma.yaml",
    num_envs=1,
    group_size=1,
):
    return SimpleNamespace(
        seed=123,
        group_size=group_size,
        total_num_envs=num_envs,
        auto_reset=False,
        ignore_terminations=False,
        max_steps_per_rollout_epoch=1,
        max_episode_steps=8,
        model_type=model_type,
        success_reward_coef=10.0,
        ndtw_reward_coef=5.0,
        split="val_seen",
        data_path=str(R2R_ROOT / "val_seen/val_seen.json.gz"),
        ndtw_gt_path=str(R2R_ROOT / "val_seen/val_seen_gt.json.gz"),
        scenes_dir=str(SCENES_DIR),
        init_params=SimpleNamespace(
            config_path=config_path,
        ),
        metrics_cfg=SimpleNamespace(
            save_metrics=False,
            metrics_base_dir="/tmp/rlinf-habitat-cma-depth-test/metrics",
        ),
        video_cfg=SimpleNamespace(
            save_video=False,
            video_base_dir="/tmp/rlinf-habitat-cma-depth-test/video",
        ),
    )


def _make_cma_habitat_cfg():
    return _make_habitat_cfg()


def _shape_summary(obs_list, key):
    return [tuple(obs[key].shape) for obs in obs_list if key in obs]


def _to_numpy_image(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        tensor_or_array = tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def _save_rgb_png(path, image):
    image = _to_numpy_image(image)
    image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _save_depth_png(path, depth):
    depth = _to_numpy_image(depth).squeeze()
    depth = np.clip(depth, 0.0, 1.0)
    depth = (depth * 255.0).astype(np.uint8)
    Image.fromarray(depth, mode="L").save(path)


def _save_observation_pngs(prefix, wrapped_obs):
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_rgb_png(PNG_OUTPUT_DIR / f"{prefix}_rgb.png", wrapped_obs["wrist_images"][0])
    _save_depth_png(
        PNG_OUTPUT_DIR / f"{prefix}_depth.png",
        wrapped_obs["extra_view_images"][0, 0],
    )
    if "main_images" in wrapped_obs:
        _save_rgb_png(
            PNG_OUTPUT_DIR / f"{prefix}_main_images.png",
            wrapped_obs["main_images"][0],
        )


def test_cma_extra_view_images_use_raw_habitat_depth_from_real_rendered_obs():
    _skip_if_habitat_assets_missing()

    env = HabitatEnv(
        cfg=_make_cma_habitat_cfg(),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
    )
    try:
        wrapped_obs, _ = env.reset()
        _save_observation_pngs("reset", wrapped_obs)

        raw_depth = env.current_raw_obs[0]["depth"]
        if raw_depth.ndim == 2:
            raw_depth = raw_depth[..., None]
        raw_depth = torch.as_tensor(raw_depth, dtype=torch.float32)
        wrapped_depth = wrapped_obs["extra_view_images"][0, 0].to(torch.float32)

        assert wrapped_obs["extra_view_images"].shape == (1, 1, 256, 256, 1)
        assert wrapped_depth.shape == raw_depth.shape
        assert torch.allclose(wrapped_depth, raw_depth)
        assert torch.is_floating_point(wrapped_obs["extra_view_images"])
        assert float(wrapped_depth.min()) >= 0.0
        assert float(wrapped_depth.max()) <= 1.0 + 1e-5
        assert not np.issubdtype(raw_depth.numpy().dtype, np.uint8)

        env.cfg.video_cfg.save_video = True
        step_obs, _, _, _, _ = env.step(np.array(["no_op"]))
        _save_observation_pngs("step", step_obs)
        assert "main_images" in step_obs
        assert (PNG_OUTPUT_DIR / "reset_rgb.png").exists()
        assert (PNG_OUTPUT_DIR / "reset_depth.png").exists()
        assert (PNG_OUTPUT_DIR / "step_rgb.png").exists()
        assert (PNG_OUTPUT_DIR / "step_depth.png").exists()
        assert (PNG_OUTPUT_DIR / "step_main_images.png").exists()
    finally:
        env.env.close()


def test_raw_habitat_depth_shapes_are_consistent_after_no_op_normalization():
    _skip_if_habitat_assets_missing()

    env = HabitatEnv(
        cfg=_make_habitat_cfg(num_envs=2, group_size=1),
        num_envs=2,
        seed_offset=0,
        total_num_processes=1,
    )
    try:
        env.reset()

        env_actions = np.array(["no_op", "move_forward"])
        raw_obs, _, _, _ = env.env.step(env._format_habitat_actions(env_actions))
        raw_depth_shapes = _shape_summary(raw_obs, "depth")

        env._normalize_depth(env_actions, raw_obs)
        normalized_depth_shapes = _shape_summary(raw_obs, "depth")

        assert raw_depth_shapes, (
            "Expected Habitat R2R raw observations to include depth"
        )
        assert len(set(normalized_depth_shapes)) == 1, (
            f"raw depth shapes: {raw_depth_shapes}; "
            f"normalized depth shapes: {normalized_depth_shapes}"
        )
        assert normalized_depth_shapes[0] == (256, 256, 1)
    finally:
        env.env.close()


def test_uninavid_ignores_inherited_habitat_depth_sensor_in_wrapped_obs():
    _skip_if_habitat_assets_missing()

    env = HabitatEnv(
        cfg=_make_habitat_cfg(
            model_type="uninavid",
            config_path="rlinf/envs/habitat/extensions/config/vlnce_r2r_uninavid.yaml",
            num_envs=2,
            group_size=1,
        ),
        num_envs=2,
        seed_offset=0,
        total_num_processes=1,
    )
    try:
        env.reset()
        wrapped_obs, _, _, _, _ = env.step(np.array(["no_op", "move_forward"]))

        assert wrapped_obs["wrist_images"].shape == (2, 480, 640, 3)
        assert "extra_view_images" not in wrapped_obs
    finally:
        env.env.close()
