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

Status: not run

## Hypothesis 2a: first-done metrics recorder

Status: not run

## Hypothesis 2b: full legacy Habitat step semantics

Status: not run

## Conclusion

Status: pending
