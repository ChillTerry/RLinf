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

Command requested:

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

Requested command result:

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

Comparison with legacy branch metrics was not run because the eval did not produce `metrics/eval` episode JSON files.

| Metric | cma2 legacy-config saved JSON | legacy saved JSON | Difference |
| --- | ---: | ---: | ---: |
| success | blocked | 0.277412 | blocked |
| spl | blocked | 0.261990 | blocked |
| oracle_success | blocked | 0.333882 | blocked |

Conclusion: blocked. Cause 1 is not confirmed, partially confirmed, or rejected because no metrics were produced.

## Hypothesis 2a: first-done metrics recorder

Status: not run

## Hypothesis 2b: full legacy Habitat step semantics

Status: not run

## Conclusion

Status: blocked on Hypothesis 1 environment setup.
