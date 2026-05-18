# CMA Habitat R2R Training Migration Design

## Goal

Port CMA model support from `feature/vln-train-cma` into the current branch so
Habitat R2R PPO and GRPO training can run with `model_type: cma`.

The migration must not modify files under `rlinf/envs/habitat`. CMA must adapt
to the current Habitat environment interface and current reward semantics.

## Scope

In scope:

- Port `rlinf/models/embodiment/cma/**` from `feature/vln-train-cma`.
- Register the CMA model in the current model registry.
- Add Habitat R2R CMA PPO and GRPO configs.
- Add a Habitat action adapter outside `rlinf/envs/habitat` if needed.
- Add focused tests for registry and action-shape compatibility.
- Report any Habitat-side behavior from the reference branch that is not
  migrated.

Out of scope:

- No changes under `rlinf/envs/habitat`.
- No GT prefix or `gt_data_path` behavior.
- No attempt to reproduce the reference branch's old Habitat reward logic.
- No broad refactor of actor, rollout, data, or config code unless required by
  CMA compatibility.

## Current Context

The current branch already registers `SupportedModel.CMA_POLICY` with value
`"cma"`, but does not include the CMA model package or CMA example configs.

The reference branch includes CMA model files and configs, but also modifies
`rlinf/envs/habitat`. Those Habitat changes include action formatting,
episode allocation, reward calculation, metrics, and config files. They are
treated as behavioral reference only, not direct migration targets.

The current Habitat wrapper already exposes the observation fields CMA needs
when configured with `model_type: cma`:

- `wrist_images` for RGB observations.
- `extra_view_images` for depth observations.
- `task_descriptions` as instruction tokens.
- `states` as episode ids.

## Architecture

### Model

Add `rlinf/models/embodiment/cma/` from the reference branch, preserving the
existing CMA implementation structure:

- `cma_action_model.py`
- `modules/instruction_encoder.py`
- `modules/policy.py`
- `modules/resnet_encoders.py`
- `modules/rnn_state_encoder.py`
- `modules/utils.py`

The public model entry point remains `get_model(cfg, torch_dtype)`. It builds
`CMAConfig`, instantiates `CMAPolicy`, optionally loads `cfg.model_path`, and
casts to the configured dtype.

`CMAPolicy.predict_action_batch` is responsible for converting current Habitat
observations into CMA inputs. It stores rollout-time `forward_inputs` required
for training-time logprob, entropy, and value recomputation.

### Registry

Extend `rlinf/models/__init__.py` with a CMA builder:

```python
def _build_cma_policy(cfg: DictConfig, torch_dtype):
    from rlinf.models.embodiment.cma import get_model

    return get_model(cfg, torch_dtype)
```

Register it with `SupportedModel.CMA_POLICY.value`.

No new model enum is needed because the current branch already defines
`SupportedModel.CMA_POLICY`.

### Action Adaptation

CMA emits discrete Habitat navigation actions. The current Habitat environment
expects action chunks compatible with its existing discrete action path.

Add a small adapter in `rlinf/envs/action_utils.py`:

- For `env_type == SupportedEnvType.HABITAT`, convert tensors to numpy as the
  existing function already does.
- If action shape is `[batch, chunks, 1]`, squeeze the singleton action
  dimension to `[batch, chunks]`.
- Otherwise pass actions through unchanged.

This keeps Habitat-specific shape normalization outside `rlinf/envs/habitat`.

### Configs

Add:

- `examples/embodiment/config/model/cma.yaml`
- `examples/embodiment/config/habitat_r2r_ppo_cma.yaml`
- `examples/embodiment/config/habitat_r2r_grpo_cma.yaml`

The configs follow the current branch's Habitat semantics:

- Use `success_reward_coef`.
- Use `ndtw_reward_coef`.
- Use `ndtw_gt_path` when NDTW ground truth is required.
- Set `model_type: ${actor.model.model_type}` under `env.train` and `env.eval`.
- Do not use `reward_coef` or `gt_data_path`.

The model path fields remain explicit user-editable paths for CMA checkpoints,
instruction embeddings, and DDPPO depth weights.

### Training Path

The intended training flow is:

1. Habitat env returns current observation dict.
2. Rollout worker calls `CMAPolicy.predict_action_batch`.
3. CMA returns discrete actions plus:
   - `prev_logprobs`
   - `prev_values`
   - `forward_inputs`
4. `prepare_actions` normalizes CMA action chunk shape for Habitat.
5. Actor receives trajectories and uses existing PPO/GRPO loss code.
6. During training, `CMAPolicy.forward` recomputes logprobs, entropy, and
   values from `forward_inputs`.

If validation exposes a logprob or entropy shape mismatch, the fix should be
localized to CMA output formatting or the smallest necessary non-Habitat
training boundary.

## Testing

Add focused tests:

- Model registry resolves `model_type: cma` to a builder rather than returning
  `None`.
- Habitat action adapter converts `[B, T, 1]` discrete CMA actions to `[B, T]`.
- CMA observation preprocessing contract does not require `gt_data_path`.

Full Habitat R2R e2e training is not required for this migration unless the
local machine has Habitat assets and dependencies available. If it cannot be
run, the final report must state that limitation.

## Migration Report Requirements

The final implementation report must explicitly state:

- `rlinf/envs/habitat` was not modified, if true.
- Which CMA files and configs were added.
- Which reference-branch Habitat changes were intentionally not migrated.
- Which tests were run and their results.
- Any remaining runtime validation gaps, especially missing Habitat assets or
  unavailable GPU/Habitat dependencies.

