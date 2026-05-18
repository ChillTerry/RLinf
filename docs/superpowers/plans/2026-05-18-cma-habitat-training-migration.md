# CMA Habitat Training Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port CMA Habitat R2R PPO/GRPO training support from `feature/vln-train-cma` while leaving `rlinf/envs/habitat` unchanged.

**Architecture:** Add the CMA model package and register it in the current model registry. Keep Habitat compatibility at non-Habitat boundaries: `rlinf/envs/action_utils.py` for action shape normalization and CMA configs that use current Habitat reward fields. Tests lock the registry, action adapter, no-GT-prefix contract, and no-Habitat-edit constraint.

**Tech Stack:** Python, PyTorch, OmegaConf/Hydra config, pytest, Ruff, Git.

---

## File Structure

Create:

- `rlinf/models/embodiment/cma/__init__.py`: CMA model entry point.
- `rlinf/models/embodiment/cma/cma_action_model.py`: CMA policy, rollout action prediction, training forward.
- `rlinf/models/embodiment/cma/modules/__init__.py`: CMA module exports.
- `rlinf/models/embodiment/cma/modules/instruction_encoder.py`: instruction token encoder.
- `rlinf/models/embodiment/cma/modules/policy.py`: CMA policy wrapper and categorical distribution.
- `rlinf/models/embodiment/cma/modules/resnet_encoders.py`: RGB/depth encoders.
- `rlinf/models/embodiment/cma/modules/rnn_state_encoder.py`: recurrent state encoder.
- `rlinf/models/embodiment/cma/modules/utils.py`: CMA module helpers.
- `examples/embodiment/config/model/cma.yaml`: base CMA model config.
- `examples/embodiment/config/habitat_r2r_ppo_cma.yaml`: Habitat R2R PPO CMA config using current Habitat fields.
- `examples/embodiment/config/habitat_r2r_grpo_cma.yaml`: Habitat R2R GRPO CMA config using current Habitat fields.
- `tests/unit_tests/test_cma_habitat_migration.py`: focused migration contract tests.

Modify:

- `rlinf/models/__init__.py`: register `SupportedModel.CMA_POLICY.value`.
- `rlinf/envs/action_utils.py`: add Habitat action shape adapter outside `rlinf/envs/habitat`.

Must not modify:

- `rlinf/envs/habitat/**`

---

### Task 1: Add Failing Migration Contract Tests

**Files:**

- Create: `tests/unit_tests/test_cma_habitat_migration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit_tests/test_cma_habitat_migration.py` with:

```python
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

"""Tests for CMA Habitat migration contracts."""

from pathlib import Path

import numpy as np

from rlinf.config import SupportedModel
from rlinf.envs.action_utils import prepare_actions
from rlinf.models import _MODEL_REGISTRY


def test_cma_policy_is_registered_in_model_registry():
    assert SupportedModel.CMA_POLICY.value == "cma"
    assert SupportedModel.CMA_POLICY.value in _MODEL_REGISTRY


def test_habitat_action_adapter_squeezes_cma_singleton_action_dim():
    raw_actions = np.array(
        [
            [[0], [1], [2]],
            [[3], [2], [1]],
        ],
        dtype=np.int64,
    )

    prepared_actions = prepare_actions(
        raw_chunk_actions=raw_actions,
        env_type="habitat",
        model_type="cma",
        num_action_chunks=3,
        action_dim=1,
    )

    assert prepared_actions.shape == (2, 3)
    np.testing.assert_array_equal(
        prepared_actions,
        np.array([[0, 1, 2], [3, 2, 1]], dtype=np.int64),
    )


def test_cma_migration_does_not_depend_on_gt_prefix_or_gt_data_path():
    feature_paths = (
        Path("examples/embodiment/config/model/cma.yaml"),
        Path("examples/embodiment/config/habitat_r2r_ppo_cma.yaml"),
        Path("examples/embodiment/config/habitat_r2r_grpo_cma.yaml"),
        Path("rlinf/models/embodiment/cma/cma_action_model.py"),
    )

    missing_paths = [str(path) for path in feature_paths if not path.exists()]
    assert missing_paths == []

    matches = []
    for path in feature_paths:
        text = path.read_text()
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if (
                "gt_prefix" in lowered
                or "gt prefix" in lowered
                or "gt-prefix" in lowered
                or "gt_data_path" in lowered
            ):
                matches.append(f"{path}:{line_number}: {line.strip()}")

    assert matches == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/unit_tests/test_cma_habitat_migration.py -v
```

Expected:

- `test_cma_policy_is_registered_in_model_registry` fails because `cma` is not in `_MODEL_REGISTRY`.
- `test_habitat_action_adapter_squeezes_cma_singleton_action_dim` fails because Habitat actions remain shape `(2, 3, 1)`.
- `test_cma_migration_does_not_depend_on_gt_prefix_or_gt_data_path` fails because CMA files/configs do not exist yet.

- [ ] **Step 3: Commit failing tests**

Run:

```bash
git add tests/unit_tests/test_cma_habitat_migration.py
git commit -s -m "test: cover cma habitat migration contracts"
```

Expected: commit succeeds.

---

### Task 2: Add Habitat Action Adapter Outside Habitat Package

**Files:**

- Modify: `rlinf/envs/action_utils.py`
- Test: `tests/unit_tests/test_cma_habitat_migration.py`

- [ ] **Step 1: Add the Habitat adapter function**

In `rlinf/envs/action_utils.py`, add this function after `prepare_actions_for_mujoco`:

```python
def prepare_actions_for_habitat(raw_chunk_actions) -> np.ndarray:
    chunk_actions = np.asarray(raw_chunk_actions)
    if chunk_actions.ndim == 3 and chunk_actions.shape[-1] == 1:
        return np.squeeze(chunk_actions, axis=-1)
    return chunk_actions
```

- [ ] **Step 2: Route Habitat through the adapter**

In `prepare_actions`, add this branch before the final `else`:

```python
    elif env_type == SupportedEnvType.HABITAT:
        chunk_actions = prepare_actions_for_habitat(
            raw_chunk_actions=raw_chunk_actions,
        )
```

- [ ] **Step 3: Run the action adapter test**

Run:

```bash
pytest tests/unit_tests/test_cma_habitat_migration.py::test_habitat_action_adapter_squeezes_cma_singleton_action_dim -v
```

Expected: PASS.

- [ ] **Step 4: Commit the adapter**

Run:

```bash
git add rlinf/envs/action_utils.py tests/unit_tests/test_cma_habitat_migration.py
git commit -s -m "feat: adapt cma actions for habitat"
```

Expected: commit succeeds.

---

### Task 3: Port CMA Model Package From Reference Branch

**Files:**

- Create: `rlinf/models/embodiment/cma/__init__.py`
- Create: `rlinf/models/embodiment/cma/cma_action_model.py`
- Create: `rlinf/models/embodiment/cma/modules/__init__.py`
- Create: `rlinf/models/embodiment/cma/modules/instruction_encoder.py`
- Create: `rlinf/models/embodiment/cma/modules/policy.py`
- Create: `rlinf/models/embodiment/cma/modules/resnet_encoders.py`
- Create: `rlinf/models/embodiment/cma/modules/rnn_state_encoder.py`
- Create: `rlinf/models/embodiment/cma/modules/utils.py`
- Test: `tests/unit_tests/test_cma_habitat_migration.py`

- [ ] **Step 1: Confirm the destination package is absent or only contains current-task edits**

Run:

```bash
if [ -e rlinf/models/embodiment/cma ]; then find rlinf/models/embodiment/cma -maxdepth 3 -type f | sort; else echo "cma package absent"; fi
```

Expected before this task: `cma package absent`.

- [ ] **Step 2: Restore the CMA package from the reference branch**

Run:

```bash
git restore --source=feature/vln-train-cma -- rlinf/models/embodiment/cma
```

Expected: the eight CMA package files listed above are created.

- [ ] **Step 3: Verify the restored package did not touch Habitat**

Run:

```bash
git status --short rlinf/envs/habitat
```

Expected: no output.

- [ ] **Step 4: Verify CMA package has no GT prefix or `gt_data_path` dependency**

Run:

```bash
rg -n "gt_prefix|GT prefix|GT-prefix|gt_data_path" rlinf/models/embodiment/cma examples/embodiment/config/model/cma.yaml examples/embodiment/config/habitat_r2r_ppo_cma.yaml examples/embodiment/config/habitat_r2r_grpo_cma.yaml
```

Expected at this point: command may report missing config files because configs are added in Task 5, but it must not report matches under `rlinf/models/embodiment/cma`.

- [ ] **Step 5: Run import smoke check**

Run:

```bash
python - <<'PY'
from rlinf.models.embodiment.cma import CMAConfig, CMAPolicy, get_model

print(CMAConfig.__name__, CMAPolicy.__name__, callable(get_model))
PY
```

Expected output:

```text
CMAConfig CMAPolicy True
```

- [ ] **Step 6: Commit the CMA package**

Run:

```bash
git add rlinf/models/embodiment/cma
git commit -s -m "feat: add cma embodiment model"
```

Expected: commit succeeds.

---

### Task 4: Register CMA in the Current Model Registry

**Files:**

- Modify: `rlinf/models/__init__.py`
- Test: `tests/unit_tests/test_cma_habitat_migration.py`

- [ ] **Step 1: Add the CMA builder**

In `rlinf/models/__init__.py`, inside `_register_builtin_models`, add this builder after `_build_flow_policy`:

```python
    def _build_cma_policy(cfg: DictConfig, torch_dtype):
        from rlinf.models.embodiment.cma import get_model

        return get_model(cfg, torch_dtype)
```

- [ ] **Step 2: Register the CMA builder**

In the same function, add this registration after the `SupportedModel.FLOW_POLICY.value` registration:

```python
    register_model(
        SupportedModel.CMA_POLICY.value,
        _build_cma_policy,
        category="embodied",
        force=True,
    )
```

- [ ] **Step 3: Run registry test**

Run:

```bash
pytest tests/unit_tests/test_cma_habitat_migration.py::test_cma_policy_is_registered_in_model_registry -v
```

Expected: PASS.

- [ ] **Step 4: Run existing custom model registry tests**

Run:

```bash
pytest tests/unit_tests/test_custom_model_registration.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit registry change**

Run:

```bash
git add rlinf/models/__init__.py tests/unit_tests/test_cma_habitat_migration.py
git commit -s -m "feat: register cma policy model"
```

Expected: commit succeeds.

---

### Task 5: Add CMA Habitat PPO and GRPO Configs Using Current Habitat Semantics

**Files:**

- Create: `examples/embodiment/config/model/cma.yaml`
- Create: `examples/embodiment/config/habitat_r2r_ppo_cma.yaml`
- Create: `examples/embodiment/config/habitat_r2r_grpo_cma.yaml`
- Test: `tests/unit_tests/test_cma_habitat_migration.py`

- [ ] **Step 1: Add base CMA model config**

Create `examples/embodiment/config/model/cma.yaml` with:

```yaml
# CMA Policy Configuration

model_type: "cma"
precision: "fp32"
load_to_device: True

action_dim: 1
num_action_classes: 4
num_action_chunks: 1

model_path: ""

image_size: [256, 256, 3]

add_value_head: True
add_q_head: False
q_head_type: "default"
num_q_heads: 2

independent_std: True
action_scale: null
final_tanh: False
std_range: null
logstd_range: null

instruction_encoder_config:
  sensor_uuid: "instruction"
  vocab_size: 2504
  use_pretrained_embeddings: True
  embedding_file: ""
  fine_tune_embeddings: True
  embedding_size: 50
  hidden_size: 128
  rnn_type: "LSTM"
  final_state_only: False
  bidirectional: True

depth_encoder_config:
  cnn_type: "VlnResnetDepthEncoder"
  output_size: 128
  backbone: "resnet50"
  ddppo_checkpoint: ""
  trainable: True

rgb_encoder_config:
  cnn_type: "TorchVisionResNet50"
  output_size: 256
  trainable: True

state_encoder_config:
  hidden_size: 512
  rnn_type: "GRU"

hidden_size: 512
normalize_rgb: False
ablate_instruction: False
ablate_depth: False
ablate_rgb: False

use_progress_monitor: False
progress_monitor_alpha: 1.0

is_lora: False
```

- [ ] **Step 2: Add PPO config**

Create `examples/embodiment/config/habitat_r2r_ppo_cma.yaml` with:

```yaml
defaults:
  - env/habitat_r2r@env.train
  - env/habitat_r2r@env.eval
  - model/cma@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: embodied
  logger:
    log_path: "logs"
    project_name: rlinf
    experiment_name: "habitat_r2r_ppo_cma"
    logger_backends: ["tensorboard"]
  max_epochs: 1000
  max_steps: -1
  only_eval: False
  eval_policy_path: null
  val_check_interval: -1
  save_interval: 10
  resume_dir: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  update_epoch: 1
  rollout_epoch: 4
  eval_rollout_epoch: 1
  group_size: 8
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.28
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  success_reward_coef: 10.0
  ndtw_reward_coef: 5.0
  filter_rewards: True
  rewards_lower_bound: 0.5
  rewards_upper_bound: 12.0
  sampling_params:
    do_sample: True
    temperature_train: 1.6
    temperature_eval: 1.6
    top_k: -1
    top_p: 1.0
    repetition_penalty: 1.0
  length_params:
    max_new_token: null
    max_length: 1024
    min_length: 1

env:
  group_name: "EnvGroup"
  enable_offload: False
  data_path_dir: "VLN-CE/datasets/r2r"
  scenes_dir: "VLN-CE/scene_dataset"
  ndtw_gt_path: ""
  train:
    total_num_envs: 256
    max_steps_per_rollout_epoch: 120
    max_episode_steps: 120
    sample_num_scenes: null
    group_size: 1
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
    ndtw_gt_path: ${env.ndtw_gt_path}
    model_type: ${actor.model.model_type}
    split: train
    data_path: ${env.data_path_dir}/${env.train.split}/${env.train.split}.json.gz
    scenes_dir: ${env.scenes_dir}
    video_cfg:
      fps: 5
      save_video: False
      async_save: False
      video_base_dir: ${runner.logger.log_path}/video/train
    metrics_cfg:
      save_metrics: True
      metrics_base_dir: ${runner.logger.log_path}/metrics/train
  eval:
    total_num_envs: 608
    max_episode_steps: 120
    max_steps_per_rollout_epoch: 360
    group_size: 1
    auto_reset: True
    ignore_terminations: True
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
    ndtw_gt_path: ${env.ndtw_gt_path}
    model_type: ${actor.model.model_type}
    split: val_unseen
    data_path: ${env.data_path_dir}/${env.eval.split}/${env.eval.split}.json.gz
    scenes_dir: ${env.scenes_dir}
    video_cfg:
      fps: 5
      save_video: False
      async_save: False
      video_base_dir: ${runner.logger.log_path}/video/eval
    metrics_cfg:
      save_metrics: True
      metrics_base_dir: ${runner.logger.log_path}/metrics/eval

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    instruction_encoder_config:
      embedding_file: ${actor.model.instruction_encoder_config.embedding_file}
    depth_encoder_config:
      ddppo_checkpoint: ${actor.model.depth_encoder_config.ddppo_checkpoint}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 128
  global_batch_size: 1024
  seed: 1234
  enable_offload: True
  model:
    model_path: "VLN-CE/models/cma_weights/ckpt/best_ckpt_r2r.pth"
    instruction_encoder_config:
      embedding_file: "VLN-CE/models/cma_weights/instruction_encoder_embedding/embeddings.json.gz"
    depth_encoder_config:
      ddppo_checkpoint: "VLN-CE/models/cma_weights/ddppo-models/gibson-2plus-resnet50.pth"
  optim:
    lr: 1.0e-5
    value_lr: 3.0e-3
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-05
    weight_decay: 0.01
    clip_grad: 1.0
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

- [ ] **Step 3: Add GRPO config**

Create `examples/embodiment/config/habitat_r2r_grpo_cma.yaml` with:

```yaml
defaults:
  - env/habitat_r2r@env.train
  - env/habitat_r2r@env.eval
  - model/cma@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: all

runner:
  task_type: embodied
  logger:
    log_path: "logs"
    project_name: rlinf
    experiment_name: "habitat_r2r_grpo_cma"
    logger_backends: ["tensorboard"]
  max_epochs: 500
  max_steps: -1
  only_eval: False
  eval_policy_path: null
  val_check_interval: -1
  save_interval: 10
  resume_dir: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  update_epoch: 1
  rollout_epoch: 1
  eval_rollout_epoch: 1
  group_size: 8
  reward_type: action_level
  logprob_type: token_level
  entropy_type: token_level
  adv_type: grpo
  loss_type: actor
  loss_agg_func: "token-mean"
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.28
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0
  gamma: 0.99
  gae_lambda: 0.95
  success_reward_coef: 10.0
  ndtw_reward_coef: 5.0
  filter_rewards: False
  rewards_lower_bound: 1.0
  rewards_upper_bound: 12.0
  sampling_params:
    do_sample: True
    temperature_train: 1.0
    temperature_eval: 0.2
    top_k: -1
    top_p: 1.0
    repetition_penalty: 1.0
  length_params:
    max_new_token: null
    max_length: 1024
    min_length: 1

env:
  group_name: "EnvGroup"
  enable_offload: False
  data_path_dir: "VLN-CE/datasets/r2r"
  scenes_dir: "VLN-CE/scene_dataset"
  ndtw_gt_path: ""
  train:
    total_num_envs: 128
    max_episode_steps: 120
    max_steps_per_rollout_epoch: 120
    sample_num_scenes: 20
    group_size: ${algorithm.group_size}
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
    ndtw_gt_path: ${env.ndtw_gt_path}
    model_type: ${actor.model.model_type}
    split: train
    data_path: ${env.data_path_dir}/${env.train.split}/${env.train.split}.json.gz
    scenes_dir: ${env.scenes_dir}
    video_cfg:
      fps: 5
      save_video: False
      async_save: False
      video_base_dir: ${runner.logger.log_path}/video/train
    metrics_cfg:
      save_metrics: True
      metrics_base_dir: ${runner.logger.log_path}/metrics/train
  eval:
    total_num_envs: 228
    max_episode_steps: 120
    max_steps_per_rollout_epoch: 960
    group_size: 1
    auto_reset: True
    ignore_terminations: True
    success_reward_coef: ${algorithm.success_reward_coef}
    ndtw_reward_coef: ${algorithm.ndtw_reward_coef}
    ndtw_gt_path: ${env.ndtw_gt_path}
    model_type: ${actor.model.model_type}
    split: val_unseen
    data_path: ${env.data_path_dir}/${env.eval.split}/${env.eval.split}.json.gz
    scenes_dir: ${env.scenes_dir}
    video_cfg:
      fps: 5
      save_video: False
      async_save: False
      video_base_dir: ${runner.logger.log_path}/video/eval
    metrics_cfg:
      save_metrics: True
      metrics_base_dir: ${runner.logger.log_path}/metrics/eval

rollout:
  group_name: "RolloutGroup"
  generation_backend: "huggingface"
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    instruction_encoder_config:
      embedding_file: ${actor.model.instruction_encoder_config.embedding_file}
    depth_encoder_config:
      ddppo_checkpoint: ${actor.model.depth_encoder_config.ddppo_checkpoint}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 120
  global_batch_size: 3840
  seed: 1234
  enable_offload: True
  model:
    model_path: "VLN-CE/models/cma_weights/ckpt/best_ckpt_r2r.pth"
    instruction_encoder_config:
      embedding_file: "VLN-CE/models/cma_weights/instruction_encoder_embedding/embeddings.json.gz"
    depth_encoder_config:
      ddppo_checkpoint: "VLN-CE/models/cma_weights/ddppo-models/gibson-2plus-resnet50.pth"
  optim:
    lr: 1.0e-6
    value_lr: 3.0e-3
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

- [ ] **Step 4: Run no-GT contract test**

Run:

```bash
pytest tests/unit_tests/test_cma_habitat_migration.py::test_cma_migration_does_not_depend_on_gt_prefix_or_gt_data_path -v
```

Expected: PASS.

- [ ] **Step 5: Compose PPO and GRPO Hydra configs**

Run:

```bash
EMBODIED_PATH=/data/RLinf/examples/embodiment python - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

config_dir = Path("examples/embodiment/config").resolve()
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    for name in ("habitat_r2r_ppo_cma", "habitat_r2r_grpo_cma"):
        cfg = compose(config_name=name)
        assert cfg.actor.model.model_type == "cma"
        assert cfg.env.train.model_type == "cma"
        assert cfg.env.eval.model_type == "cma"
        assert "gt_data_path" not in cfg.env.train
        assert "gt_data_path" not in cfg.env.eval
        assert "reward_coef" not in cfg.env.train
        assert "reward_coef" not in cfg.env.eval
        assert cfg.env.train.success_reward_coef == cfg.algorithm.success_reward_coef
        assert cfg.env.train.ndtw_reward_coef == cfg.algorithm.ndtw_reward_coef
        print(name, "ok")
PY
```

Expected output:

```text
habitat_r2r_ppo_cma ok
habitat_r2r_grpo_cma ok
```

- [ ] **Step 6: Commit configs**

Run:

```bash
git add examples/embodiment/config/model/cma.yaml examples/embodiment/config/habitat_r2r_ppo_cma.yaml examples/embodiment/config/habitat_r2r_grpo_cma.yaml tests/unit_tests/test_cma_habitat_migration.py
git commit -s -m "feat: add cma habitat training configs"
```

Expected: commit succeeds.

---

### Task 6: Run Focused Verification and Guard Habitat Directory

**Files:**

- Verify only; no file edits expected.

- [ ] **Step 1: Run migration tests**

Run:

```bash
pytest tests/unit_tests/test_cma_habitat_migration.py -v
```

Expected: all tests PASS.

- [ ] **Step 2: Run registry tests**

Run:

```bash
pytest tests/unit_tests/test_custom_model_registration.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run formatter/linter on touched Python files**

Run:

```bash
ruff format rlinf/envs/action_utils.py rlinf/models/__init__.py rlinf/models/embodiment/cma tests/unit_tests/test_cma_habitat_migration.py
ruff check rlinf/envs/action_utils.py rlinf/models/__init__.py rlinf/models/embodiment/cma tests/unit_tests/test_cma_habitat_migration.py
```

Expected: both commands complete successfully.

- [ ] **Step 4: Verify `rlinf/envs/habitat` has no changes**

Run:

```bash
git diff --name-only HEAD -- rlinf/envs/habitat
git status --short rlinf/envs/habitat
```

Expected: both commands produce no output.

- [ ] **Step 5: Verify reference-branch Habitat files were not restored**

Run:

```bash
git status --short rlinf/envs/habitat/extensions/config/vlnce_r2r_cma.yaml rlinf/envs/habitat/extensions/config/scene_vram_profile.json
```

Expected: no output.

- [ ] **Step 6: Record runtime validation gap**

Do not run full Habitat R2R training unless local Habitat data, scenes, CMA weights, and GPU dependencies are available. In the final report, include this exact statement when full e2e was not run:

```text
Full Habitat R2R PPO/GRPO training was not run because local Habitat assets,
CMA checkpoints, and GPU runtime dependencies were not validated in this
session.
```

- [ ] **Step 7: Commit verification-only formatting changes if Ruff changed files**

Run:

```bash
git status --short
```

If Ruff modified touched files, run:

```bash
git add rlinf/envs/action_utils.py rlinf/models/__init__.py rlinf/models/embodiment/cma tests/unit_tests/test_cma_habitat_migration.py
git commit -s -m "style: format cma habitat migration"
```

Expected: commit succeeds only if files changed.

---

## Self-Review

Spec coverage:

- CMA model package: Task 3.
- Model registry: Task 4.
- Habitat action adapter outside `rlinf/envs/habitat`: Task 2.
- PPO/GRPO configs with current Habitat fields: Task 5.
- Focused tests: Tasks 1, 2, 4, 5, 6.
- No Habitat edits: Tasks 3 and 6.
- Final report requirements: Task 6.

No placeholder scan:

- The plan uses concrete paths, commands, expected outputs, and code snippets.
- It does not require GT prefix, `gt_data_path`, or Habitat package edits.

Type consistency:

- `SupportedModel.CMA_POLICY.value` matches current `rlinf/config.py`.
- `prepare_actions_for_habitat(raw_chunk_actions)` returns `np.ndarray`.
- Config fields use current `env/habitat_r2r.yaml` names: `success_reward_coef`, `ndtw_reward_coef`, and `ndtw_gt_path`.

