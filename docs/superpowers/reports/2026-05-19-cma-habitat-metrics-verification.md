# CMA Habitat Metrics Verification Report

## Baseline

Command:

```bash
python .vscode/compare_habitat_metrics.py \
  logs/20260519-03:43:47-habitat_r2r_eval_cma/metrics/eval \
  logs/20260519-01:32:39/metrics/eval
```

Observed:

| Metric | cma2 saved JSON | legacy saved JSON | Difference |
| --- | ---: | ---: | ---: |
| success | 0.248904 | 0.277412 | -0.028509 |
| spl | 0.232475 | 0.261990 | -0.029515 |
| oracle_success | 0.299890 | 0.333882 | -0.033991 |

## Hypothesis 1: Habitat config composition

Status: blocked

Earlier requested command:

```bash
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_legacy_config"
mkdir -p "${LOG_DIR}"
python examples/embodiment/eval_embodied_agent.py \
  --config-path examples/embodiment/config/ \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.init_params.config_path=rlinf/envs/habitat/extensions/config/vlnce_r2r_cma_legacy.yaml \
  env.eval.data_path=VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz \
  env.eval.ndtw_gt_path=null \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Earlier requested command result:

- `LOG_DIR`: `logs/20260519-04:35:50-habitat_r2r_eval_cma_legacy_config`
- Exit status: `1`
- Episode JSON files: `0`
- Error summary: Hydra resolved `--config-path examples/embodiment/config/` relative to the script-local config path and reported missing config directory `/data/RLinf/examples/embodiment/examples/embodiment/config`.

Equivalent rerun command, using the script-local Hydra config path:

```bash
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_legacy_config"
mkdir -p "${LOG_DIR}"
python examples/embodiment/eval_embodied_agent.py \
  --config-path config \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.init_params.config_path=rlinf/envs/habitat/extensions/config/vlnce_r2r_cma_legacy.yaml \
  env.eval.data_path=VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz \
  env.eval.ndtw_gt_path=null \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Equivalent rerun result:

- `LOG_DIR`: `logs/20260519-04:36:42-habitat_r2r_eval_cma_legacy_config`
- Exit status: `1`
- Episode JSON files: `0`
- Error summary: config interpolation failed because environment variable `EMBODIED_PATH` is not set.

These two failures were invocation environment errors, not Habitat evaluation results.

Corrected command:

```bash
export EMBODIED_PATH="$(cd examples/embodiment && pwd)"
export REPO_PATH="$(pwd)"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_legacy_config"
mkdir -p "${LOG_DIR}"
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.init_params.config_path=rlinf/envs/habitat/extensions/config/vlnce_r2r_cma_legacy.yaml \
  env.eval.data_path=VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz \
  env.eval.ndtw_gt_path=null \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Corrected command result:

- `LOG_DIR`: `logs/20260519-04:39:39-habitat_r2r_eval_cma_legacy_config`
- Exit status: `255`
- Episode JSON files: `0`
- GPU placement: rollout/env workers were placed on GPU ranks `4`, `5`, `6`, and `7`.
- Error summary: evaluation reached Habitat worker execution, then failed in `HabitatEnv._record_metrics` with `KeyError: 'ndtw'`.

Interpretation:

- The corrected command resolved the prior invocation environment failures.
- The exact legacy config composition omits the `ndtw` measurement.
- Current cma2 metrics recording unconditionally expects `infos["ndtw"]`.
- Therefore, Cause 1 cannot be measured in isolation with this exact command because the legacy config composition exposes a metrics-recorder contract mismatch before any episode JSON files are saved.

Comparison with legacy branch metrics was not run because the corrected eval did not produce `metrics/eval` episode JSON files.

| Metric | cma2 legacy-config saved JSON | legacy saved JSON | Difference |
| --- | ---: | ---: | ---: |
| success | blocked | 0.277412 | blocked |
| spl | blocked | 0.261990 | blocked |
| oracle_success | blocked | 0.333882 | blocked |

Conclusion: blocked by current metrics recorder expectations under the exact legacy config composition. Cause 1 is not confirmed, partially confirmed, or rejected because no metrics were produced.

## Hypothesis 2a: first-done metrics recorder

Status: rejected as the primary cause

Implementation:

- Added an opt-in legacy first-done debug recorder in `HabitatEnv`.
- The recorder is disabled by default and only runs when
  `env.eval.metrics_cfg.legacy_metrics_base_dir` is provided.

Initial dual-recorder command:

```bash
export EMBODIED_PATH="$(cd examples/embodiment && pwd)"
export REPO_PATH="$(pwd)"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_dual_recorder"
mkdir -p "${LOG_DIR}"
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  +env.eval.metrics_cfg.legacy_metrics_base_dir="${LOG_DIR}/metrics/eval_legacy_recorder" \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Initial result:

- `LOG_DIR`: `logs/20260519-04:58:49-habitat_r2r_eval_cma_dual_recorder`
- Exit status: `0` from the shell pipeline, but the eval process reported worker failure and exited main execution early.
- Current recorder episode JSON files: `0`
- Legacy debug recorder episode JSON files: `0`
- Failure point: `EnvWorker.init_worker`, before any episode rollout or metric JSON output.
- Error summary:

```text
ValueError: record count must be divisible by total_num_processes * num_group
Exiting main process due to a failure upon worker execution.
```

Root cause of initial blocked run:

- The command did not override `env.eval.data_path` and `env.eval.ndtw_gt_path`.
- Hydra therefore used the default `tiny_val_unseen.json.gz` from `examples/embodiment/config/habitat_r2r_eval_cma.yaml`.
- The tiny dataset episode count does not satisfy the current allocator constraint `total_num_processes * num_group`.
- This failure happened before rollout and was not caused by the opt-in recorder.

Corrected full-val command:

```bash
export EMBODIED_PATH="$(cd examples/embodiment && pwd)"
export REPO_PATH="$(pwd)"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_dual_recorder"
mkdir -p "${LOG_DIR}"
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.data_path=VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz \
  env.eval.ndtw_gt_path=VLN-CE/datasets/r2r/val_unseen/val_unseen_gt.json.gz \
  +env.eval.metrics_cfg.legacy_metrics_base_dir="${LOG_DIR}/metrics/eval_legacy_recorder" \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Corrected full-val result:

- `LOG_DIR`: `logs/20260519-05:03:51-habitat_r2r_eval_cma_dual_recorder`
- Current recorder episode JSON files: `1824`
- Legacy debug recorder episode JSON files: `1824`

Same-run comparison:

```bash
python .vscode/compare_habitat_metrics.py \
  logs/20260519-05:03:51-habitat_r2r_eval_cma_dual_recorder/metrics/eval \
  logs/20260519-05:03:51-habitat_r2r_eval_cma_dual_recorder/metrics/eval_legacy_recorder
```

| Metric | current recorder | legacy debug recorder | Difference |
| --- | ---: | ---: | ---: |
| success | 0.256579 | 0.256579 | 0.000000 |
| spl | 0.242229 | 0.242062 | 0.000167 |
| oracle_success | 0.302083 | 0.302083 | 0.000000 |

Additional same-run facts:

- `common_count`: `1824`
- `success.changed`: `0`
- `oracle_success.changed`: `0`
- `spl.changed`: `10`

Legacy debug recorder vs legacy branch:

```bash
python .vscode/compare_habitat_metrics.py \
  logs/20260519-05:03:51-habitat_r2r_eval_cma_dual_recorder/metrics/eval_legacy_recorder \
  logs/20260519-01:32:39/metrics/eval
```

| Metric | legacy debug recorder | legacy branch | Difference |
| --- | ---: | ---: | ---: |
| success | 0.256579 | 0.277412 | -0.020833 |
| spl | 0.242062 | 0.261990 | -0.019928 |
| oracle_success | 0.302083 | 0.333882 | -0.031798 |

Additional legacy-branch comparison facts:

- `common_count`: `1824`
- `success.changed`: `494`
- `spl.changed`: `626`
- `oracle_success.changed`: `488`

Conclusion:

Rejected as the primary explanation for the metrics gap. On the same corrected
full-val trajectory, the current recorder and legacy first-done debug recorder
produce identical `success` and `oracle_success`, and only a very small `spl`
mean difference (`0.000167`). The legacy debug recorder still remains below the
legacy branch by `success=-0.020833`, `spl=-0.019928`, and
`oracle_success=-0.031798`, so the first-done recorder alone does not explain
the observed gap.

## Hypothesis 2b: full legacy Habitat step semantics

Status: suggestive, not controlled

Implementation:

- Added an explicit opt-in `env.eval.use_legacy_cma_step_semantics` flag in
  `HabitatEnv`.
- Default `chunk_step` behavior remains unchanged when the flag is absent or
  false.

Syntax check:

```bash
python -m py_compile rlinf/envs/habitat/habitat_env.py
```

Syntax check result: pass.

Command:

```bash
export EMBODIED_PATH="$(cd examples/embodiment && pwd)"
export REPO_PATH="$(pwd)"
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH}"
export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
LOG_DIR="logs/$(date +'%Y%m%d-%H:%M:%S')-habitat_r2r_eval_cma_legacy_step"
mkdir -p "${LOG_DIR}"
python "${SRC_FILE}" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name habitat_r2r_eval_cma \
  runner.logger.log_path="${LOG_DIR}" \
  env.eval.data_path=VLN-CE/datasets/r2r/val_unseen/val_unseen.json.gz \
  env.eval.ndtw_gt_path=VLN-CE/datasets/r2r/val_unseen/val_unseen_gt.json.gz \
  +env.eval.use_legacy_cma_step_semantics=True \
  2>&1 | tee "${LOG_DIR}/eval_embodiment.log"
```

Result:

- `LOG_DIR`: `logs/20260519-05:31:36-habitat_r2r_eval_cma_legacy_step`
- Eval process exit status: `0`
- Episode JSON files: `1824`
- GPU placement: rollout/env workers were placed on GPU ranks `4`, `5`, `6`,
  and `7`.

Legacy-step run vs legacy branch:

```bash
python .vscode/compare_habitat_metrics.py \
  logs/20260519-05:31:36-habitat_r2r_eval_cma_legacy_step/metrics/eval \
  logs/20260519-01:32:39/metrics/eval
```

| Metric | legacy-step cma2 run | legacy branch | Difference |
| --- | ---: | ---: | ---: |
| success | 0.259320 | 0.277412 | -0.018092 |
| spl | 0.242417 | 0.261990 | -0.019573 |
| oracle_success | 0.303180 | 0.333882 | -0.030702 |

Additional legacy-branch comparison facts:

- `common_count`: `1824`
- `success.changed`: `479`
- `spl.changed`: `623`
- `oracle_success.changed`: `496`
- `success_flips.left_success_right_fail`: `223`
- `success_flips.left_fail_right_success`: `256`

Legacy-step run vs cma2 baseline:

```bash
python .vscode/compare_habitat_metrics.py \
  logs/20260519-05:31:36-habitat_r2r_eval_cma_legacy_step/metrics/eval \
  logs/20260519-03:43:47-habitat_r2r_eval_cma/metrics/eval
```

| Metric | legacy-step cma2 run | cma2 baseline | Difference |
| --- | ---: | ---: | ---: |
| success | 0.259320 | 0.248904 | 0.010417 |
| spl | 0.242417 | 0.232475 | 0.009942 |
| oracle_success | 0.303180 | 0.299890 | 0.003289 |

Additional cma2-baseline comparison facts:

- `common_count`: `1824`
- `success.changed`: `383`
- `spl.changed`: `538`
- `oracle_success.changed`: `376`
- `success_flips.left_success_right_fail`: `201`
- `success_flips.left_fail_right_success`: `182`

Conclusion:

Suggestive, but not an isolated causal estimate. The legacy-step run is closer
to the legacy branch than the original cma2 baseline on `success` and `spl`,
with the original baseline gaps changing from `success=-0.028509` to
`success=-0.018092` and from `spl=-0.029515` to `spl=-0.019573`. However, these
comparisons are across independent runs. A corrected default CMA run in
Hypothesis 2a produced `success=0.256579`, while the original cma2 baseline was
`success=0.248904`, so run-to-run variation is large enough that this result
cannot be treated as a controlled same-run estimate of the legacy step effect.
The legacy-step run also remains materially below the legacy branch, especially
for `oracle_success=-0.030702`.

## Conclusion

Status: unresolved with suggestive evidence.

- Hypothesis 1: blocked because the corrected eval reaches Habitat execution
  but fails before metric JSON output with missing `ndtw` in `infos`.
- Hypothesis 2a: rejected as the primary cause because the same-run current and
  legacy first-done recorders match on `success` and `oracle_success`, and the
  remaining `spl` delta is far smaller than the baseline gap.
- Hypothesis 2b: suggestive but not controlled. The legacy-step run is closer
  to the legacy branch than the original cma2 baseline on `success` and `spl`,
  but no paired default rerun was performed, and observed default CMA run
  variation prevents attributing the difference causally to step semantics
  alone.
