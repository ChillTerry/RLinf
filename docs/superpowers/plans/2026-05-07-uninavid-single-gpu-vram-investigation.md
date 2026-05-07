# UniNaVid Single-GPU VRAM Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify why UniNaVid Habitat eval shows large single-GPU VRAM swings over time.

**Architecture:** This is an evidence-first investigation. Each run uses a temporary Hydra config copied outside the tracked source tree, external `nvidia-smi` sampling, and controlled one-variable changes so the memory source can be classified before any code changes are proposed.

**Tech Stack:** RLinf embodied eval runner, Hydra/OmegaConf config composition, Ray workers, Habitat simulator, UniNaVid rollout model, Bash, Python standard library, `nvidia-smi`.

---

## File Structure

No source files are modified during the no-code investigation.

Runtime artifacts are created under:

- `results/uninavid_single_gpu_vram/<RUN_ID>/`: investigation root for one complete matrix.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/hydra_config/habitat_r2r_eval_uninavid_single_gpu.yaml`: temporary primary Hydra config with single-GPU placement.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/resolved_config.yaml`: resolved Hydra config for the case.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/gpu_samples.csv`: one-second selected-GPU samples.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/process_samples.csv`: one-second process-level GPU memory samples.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/process_tree_samples.csv`: process command samples for mapping Ray worker roles to PIDs.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/eval.log`: RLinf eval stdout/stderr.
- `results/uninavid_single_gpu_vram/<RUN_ID>/<CASE_NAME>/summary.json`: per-case memory summary.
- `results/uninavid_single_gpu_vram/<RUN_ID>/matrix_summary.md`: final evidence table and root-cause classification.

If the no-code matrix is ambiguous, stop and request approval for a separate opt-in instrumentation plan scoped only to:

- `rlinf/models/embodiment/uninavid/`
- `rlinf/envs/habitat/`

## Shared Shell Setup

Use these variables for all tasks in one shell session:

```bash
export REPO=/data/RLinf
export GPU_ID="${GPU_ID:-4}"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-single-gpu-vram"
export RUN_ROOT="${REPO}/results/uninavid_single_gpu_vram/${RUN_ID}"
export EMBODIED_PATH="${REPO}/examples/embodiment"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
mkdir -p "${RUN_ROOT}"
```

Expected:

```text
The directory ${RUN_ROOT} exists, and no files under examples/ or rlinf/ are modified.
```

## Shared Case Runner

Each eval case uses this command pattern. Substitute the case-specific values from
the tasks below.

```bash
CASE_NAME="baseline_env1"
TOTAL_ENVS="1"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"

CASE_DIR="${RUN_ROOT}/${CASE_NAME}"
CONFIG_DIR="${CASE_DIR}/hydra_config"
CONFIG_FILE="${CONFIG_DIR}/habitat_r2r_eval_uninavid_single_gpu.yaml"
mkdir -p "${CONFIG_DIR}"
cp "${REPO}/examples/embodiment/config/habitat_r2r_eval_uninavid.yaml" "${CONFIG_FILE}"
sed -i -E "s/^([[:space:]]*)actor,env,rollout: .*/\\1actor,env,rollout: ${GPU_ID}-${GPU_ID}/" "${CONFIG_FILE}"

python - "${CONFIG_DIR}" "${CASE_DIR}/resolved_config.yaml" "${CASE_DIR}" "${TOTAL_ENVS}" "${MAX_STEPS}" "${ROLLOUT_MODE}" "${MAX_NEW_TOKEN}" <<'PY'
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir, output_path, case_dir, total_envs, max_steps, rollout_mode, max_new_token = sys.argv[1:]
os.environ["EMBODIED_PATH"] = os.environ.get(
    "EMBODIED_PATH",
    "/data/RLinf/examples/embodiment",
)
with initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = compose(
        config_name="habitat_r2r_eval_uninavid_single_gpu",
        overrides=[
            f"runner.logger.log_path={case_dir}/rlinf",
            f"env.eval.total_num_envs={total_envs}",
            "env.eval.max_episode_steps=512",
            f"env.eval.max_steps_per_rollout_epoch={max_steps}",
            "env.eval.video_cfg.save_video=false",
            f"actor.model.rollout_mode={rollout_mode}",
            f"algorithm.length_params.max_new_token={max_new_token}",
        ],
    )
Path(output_path).write_text(OmegaConf.to_yaml(cfg, resolve=True))
PY

nvidia-smi \
  --id="${GPU_ID}" \
  --query-gpu=timestamp,index,memory.used,memory.free,memory.total,utilization.gpu \
  --format=csv,nounits \
  -l 1 > "${CASE_DIR}/gpu_samples.csv" &
GPU_SAMPLER_PID=$!

(
  printf 'timestamp,gpu_uuid,pid,process_name,used_memory_mib\n'
  while true; do
    ts="$(date -Iseconds)"
    nvidia-smi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits |
      awk -v ts="${ts}" 'BEGIN { FS=", "; OFS="," } NF >= 4 { print ts, $1, $2, $3, $4 }'
    sleep 1
  done
) > "${CASE_DIR}/process_samples.csv" &
PROCESS_SAMPLER_PID=$!

(
  printf 'timestamp,pid,ppid,cmd\n'
  while true; do
    ts="$(date -Iseconds)"
    ps -eo pid=,ppid=,cmd= |
      awk -v ts="${ts}" '
        index($0, "ray::") || index($0, "eval_embodied_agent") || index($0, "habitat") || index($0, "uninavid") {
          pid=$1
          ppid=$2
          $1=""
          $2=""
          sub(/^[[:space:]]+/, "", $0)
          gsub(/,/, " ", $0)
          print ts "," pid "," ppid "," $0
        }'
    sleep 1
  done
) > "${CASE_DIR}/process_tree_samples.csv" &
PROCESS_TREE_SAMPLER_PID=$!

set +e
python "${EMBODIED_PATH}/eval_embodied_agent.py" \
  --config-path "${CONFIG_DIR}" \
  --config-name habitat_r2r_eval_uninavid_single_gpu \
  "runner.logger.log_path=${CASE_DIR}/rlinf" \
  "env.eval.total_num_envs=${TOTAL_ENVS}" \
  "env.eval.max_episode_steps=512" \
  "env.eval.max_steps_per_rollout_epoch=${MAX_STEPS}" \
  "env.eval.video_cfg.save_video=false" \
  "actor.model.rollout_mode=${ROLLOUT_MODE}" \
  "algorithm.length_params.max_new_token=${MAX_NEW_TOKEN}" \
  2>&1 | tee "${CASE_DIR}/eval.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

kill "${GPU_SAMPLER_PID}" "${PROCESS_SAMPLER_PID}" "${PROCESS_TREE_SAMPLER_PID}" 2>/dev/null || true
wait "${GPU_SAMPLER_PID}" "${PROCESS_SAMPLER_PID}" "${PROCESS_TREE_SAMPLER_PID}" 2>/dev/null || true
if [ "${RUN_STATUS}" -ne 0 ]; then
  echo "case ${CASE_NAME} failed with status ${RUN_STATUS}" >&2
fi
test "${RUN_STATUS}" -eq 0
```

Expected:

```text
For a successful case, eval.log ends without Traceback or CUDA out of memory.
resolved_config.yaml, gpu_samples.csv, process_samples.csv, and
process_tree_samples.csv exist and contain multiple sample rows.
```

## Shared Case Summary Command

Run this after each case. Substitute `CASE_NAME` with the completed case.

```bash
CASE_NAME="baseline_env1"
CASE_DIR="${RUN_ROOT}/${CASE_NAME}"
python - "${CASE_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
gpu_csv = case_dir / "gpu_samples.csv"
process_csv = case_dir / "process_samples.csv"
eval_log = case_dir / "eval.log"
summary_json = case_dir / "summary.json"

def find_column(fieldnames, needle):
    for name in fieldnames:
        if needle in name:
            return name
    raise KeyError(f"missing column containing {needle!r}: {fieldnames}")

gpu_rows = []
with gpu_csv.open(newline="") as file_obj:
    reader = csv.DictReader(file_obj)
    used_col = find_column(reader.fieldnames, "memory.used")
    free_col = find_column(reader.fieldnames, "memory.free")
    util_col = find_column(reader.fieldnames, "utilization.gpu")
    for row in reader:
        try:
            gpu_rows.append(
                {
                    "used": float(row[used_col]),
                    "free": float(row[free_col]),
                    "util": float(row[util_col]),
                }
            )
        except ValueError:
            continue

process_peaks = {}
with process_csv.open(newline="") as file_obj:
    reader = csv.DictReader(file_obj)
    for row in reader:
        pid = row.get("pid", "").strip()
        process_name = row.get("process_name", "").strip()
        used_text = row.get("used_memory_mib", "").strip()
        if not pid or not used_text:
            continue
        try:
            used = float(used_text)
        except ValueError:
            continue
        current = process_peaks.get(pid)
        if current is None or used > current["peak_used_mib"]:
            process_peaks[pid] = {
                "pid": pid,
                "process_name": process_name,
                "peak_used_mib": used,
            }

log_text = eval_log.read_text(errors="replace") if eval_log.exists() else ""
summary = {
    "case": case_dir.name,
    "sample_count": len(gpu_rows),
    "gpu_used_min_mib": min((row["used"] for row in gpu_rows), default=None),
    "gpu_used_max_mib": max((row["used"] for row in gpu_rows), default=None),
    "gpu_used_delta_mib": (
        max(row["used"] for row in gpu_rows) - min(row["used"] for row in gpu_rows)
        if gpu_rows
        else None
    ),
    "gpu_util_max_percent": max((row["util"] for row in gpu_rows), default=None),
    "process_peaks": sorted(
        process_peaks.values(),
        key=lambda item: item["peak_used_mib"],
        reverse=True,
    )[:20],
    "has_traceback": "Traceback" in log_text,
    "has_cuda_oom": "CUDA out of memory" in log_text,
}
summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
```

Expected:

```text
summary.json contains gpu_used_min_mib, gpu_used_max_mib, gpu_used_delta_mib,
has_traceback, and has_cuda_oom.
```

---

### Task 1: Prepare Investigation Root

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/`
- Modify: none
- Test: shell checks only

- [ ] **Step 1: Export shared environment variables**

```bash
export REPO=/data/RLinf
export GPU_ID="${GPU_ID:-4}"
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-single-gpu-vram"
export RUN_ROOT="${REPO}/results/uninavid_single_gpu_vram/${RUN_ID}"
export EMBODIED_PATH="${REPO}/examples/embodiment"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export HYDRA_FULL_ERROR=1
mkdir -p "${RUN_ROOT}"
```

- [ ] **Step 2: Verify selected GPU is visible**

```bash
nvidia-smi --id="${GPU_ID}" --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Expected:

```text
One row is printed for the selected GPU, and memory.free is high enough for a UniNaVid eval run.
```

- [ ] **Step 3: Record repository state**

```bash
git status --short > "${RUN_ROOT}/git_status_before.txt"
git rev-parse HEAD > "${RUN_ROOT}/git_head.txt"
```

Expected:

```text
git_status_before.txt and git_head.txt exist under ${RUN_ROOT}.
```

- [ ] **Step 4: Commit runtime artifact directory is untracked**

```bash
test -d "${RUN_ROOT}" && test -f "${RUN_ROOT}/git_head.txt"
```

Expected:

```text
Command exits with status 0.
```

### Task 2: Verify Temporary Hydra Config Composition

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/compose_check/hydra_config/habitat_r2r_eval_uninavid_single_gpu.yaml`
- Modify: none
- Test: Hydra compose check

- [ ] **Step 1: Create temporary primary config**

```bash
CASE_DIR="${RUN_ROOT}/compose_check"
CONFIG_DIR="${CASE_DIR}/hydra_config"
CONFIG_FILE="${CONFIG_DIR}/habitat_r2r_eval_uninavid_single_gpu.yaml"
mkdir -p "${CONFIG_DIR}"
cp "${REPO}/examples/embodiment/config/habitat_r2r_eval_uninavid.yaml" "${CONFIG_FILE}"
sed -i -E "s/^([[:space:]]*)actor,env,rollout: .*/\\1actor,env,rollout: ${GPU_ID}-${GPU_ID}/" "${CONFIG_FILE}"
```

- [ ] **Step 2: Compose without launching RLinf**

```bash
python - "${CONFIG_DIR}" <<'PY'
import os
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

os.environ["EMBODIED_PATH"] = os.environ.get(
    "EMBODIED_PATH",
    "/data/RLinf/examples/embodiment",
)
config_dir = sys.argv[1]
with initialize_config_dir(version_base="1.1", config_dir=config_dir):
    cfg = compose(
        config_name="habitat_r2r_eval_uninavid_single_gpu",
        overrides=[
            "env.eval.total_num_envs=1",
            "env.eval.max_steps_per_rollout_epoch=512",
            "actor.model.rollout_mode=batched_feature_cache",
            "algorithm.length_params.max_new_token=1024",
        ],
    )
print(OmegaConf.select(cfg, "cluster.component_placement"))
print(OmegaConf.select(cfg, "env.eval.total_num_envs"))
print(OmegaConf.select(cfg, "actor.model.rollout_mode"))
PY
```

Expected:

```text
The output includes actor,env,rollout mapped to the selected single GPU,
env.eval.total_num_envs equals 1, and rollout_mode equals batched_feature_cache.
```

### Task 3: Run Minimal Baseline

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/baseline_env1/`
- Modify: none
- Test: external GPU samples and RLinf eval log

- [ ] **Step 1: Run baseline case**

Use the shared case runner with:

```bash
CASE_NAME="baseline_env1"
TOTAL_ENVS="1"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

Expected:

```text
baseline_env1/eval.log, baseline_env1/gpu_samples.csv, and
baseline_env1/process_samples.csv are created.
```

- [ ] **Step 2: Summarize baseline case**

Use the shared case summary command with:

```bash
CASE_NAME="baseline_env1"
```

Expected:

```text
baseline_env1/summary.json exists and has has_traceback=false unless the baseline hit a real runtime failure.
```

- [ ] **Step 3: Classify baseline**

```bash
python - "${RUN_ROOT}/baseline_env1/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
delta = summary["gpu_used_delta_mib"]
print(f"baseline_delta_mib={delta}")
if summary["has_cuda_oom"]:
    print("classification=baseline_oom")
elif delta is not None and delta >= 8192:
    print("classification=large_swing_at_env1")
else:
    print("classification=env1_not_enough_to_reproduce")
PY
```

Expected:

```text
The output is one of baseline_oom, large_swing_at_env1, or env1_not_enough_to_reproduce.
```

### Task 4: Run Batch Scaling Cases

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/batch_env2/`
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/batch_env4/`
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/batch_env8/`
- Optional create: `results/uninavid_single_gpu_vram/<RUN_ID>/batch_env16/`
- Modify: none
- Test: external GPU samples and summary JSON files

- [ ] **Step 1: Run env count 2**

Use the shared case runner with:

```bash
CASE_NAME="batch_env2"
TOTAL_ENVS="2"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 2: Summarize env count 2**

Use the shared case summary command with:

```bash
CASE_NAME="batch_env2"
```

Expected:

```text
batch_env2/summary.json exists.
```

- [ ] **Step 3: Run env count 4**

Use the shared case runner with:

```bash
CASE_NAME="batch_env4"
TOTAL_ENVS="4"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 4: Summarize env count 4**

Use the shared case summary command with:

```bash
CASE_NAME="batch_env4"
```

Expected:

```text
batch_env4/summary.json exists.
```

- [ ] **Step 5: Run env count 8**

Use the shared case runner with:

```bash
CASE_NAME="batch_env8"
TOTAL_ENVS="8"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 6: Summarize env count 8**

Use the shared case summary command with:

```bash
CASE_NAME="batch_env8"
```

Expected:

```text
batch_env8/summary.json exists.
```

- [ ] **Step 7: Run env count 16 only if env count 8 stayed below 85 percent of GPU memory**

```bash
python - "${RUN_ROOT}/batch_env8/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
peak = float(summary["gpu_used_max_mib"])
print(f"batch_env8_peak_mib={peak}")
print("run_env16=yes" if peak < 85000 else "run_env16=no")
PY
```

Expected:

```text
If run_env16=no is printed, skip batch_env16 and continue to rollout mode contrast.
```

- [ ] **Step 8: Run env count 16 when safe**

Use the shared case runner with:

```bash
CASE_NAME="batch_env16"
TOTAL_ENVS="16"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 9: Summarize env count 16 when it was run**

Use the shared case summary command with:

```bash
CASE_NAME="batch_env16"
```

Expected:

```text
batch_env16/summary.json exists when batch_env16 was run.
```

- [ ] **Step 10: Compare batch scaling**

```bash
python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
case_names = ["baseline_env1", "batch_env2", "batch_env4", "batch_env8", "batch_env16"]
rows = []
for case_name in case_names:
    path = run_root / case_name / "summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    env_count = 1 if case_name == "baseline_env1" else int(case_name.replace("batch_env", ""))
    rows.append(
        {
            "case": case_name,
            "env_count": env_count,
            "peak": data["gpu_used_max_mib"],
            "delta": data["gpu_used_delta_mib"],
            "oom": data["has_cuda_oom"],
        }
    )
for row in rows:
    print(f'{row["case"]}: envs={row["env_count"]} peak_mib={row["peak"]} delta_mib={row["delta"]} oom={row["oom"]}')
if len(rows) >= 3:
    first = rows[0]["peak"]
    last = rows[-1]["peak"]
    if first is not None and last is not None and last - first >= 8192:
        print("classification=batch_scaled_memory")
    else:
        print("classification=weak_batch_dependence")
PY
```

Expected:

```text
The output prints per-case peak and delta, followed by batch_scaled_memory or weak_batch_dependence.
```

### Task 5: Run Rollout Mode Contrast

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/sequential_env4/`
- Modify: none
- Test: compare `sequential_cache` with `batch_env4`

- [ ] **Step 1: Run sequential cache at env count 4**

Use the shared case runner with:

```bash
CASE_NAME="sequential_env4"
TOTAL_ENVS="4"
MAX_STEPS="512"
ROLLOUT_MODE="sequential_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 2: Summarize sequential cache**

Use the shared case summary command with:

```bash
CASE_NAME="sequential_env4"
```

Expected:

```text
sequential_env4/summary.json exists.
```

- [ ] **Step 3: Compare rollout modes**

```bash
python - "${RUN_ROOT}/batch_env4/summary.json" "${RUN_ROOT}/sequential_env4/summary.json" <<'PY'
import json
import sys
from pathlib import Path

batched = json.loads(Path(sys.argv[1]).read_text())
sequential = json.loads(Path(sys.argv[2]).read_text())
batched_peak = batched["gpu_used_max_mib"]
sequential_peak = sequential["gpu_used_max_mib"]
batched_delta = batched["gpu_used_delta_mib"]
sequential_delta = sequential["gpu_used_delta_mib"]
print(f"batched_peak_mib={batched_peak}")
print(f"sequential_peak_mib={sequential_peak}")
print(f"batched_delta_mib={batched_delta}")
print(f"sequential_delta_mib={sequential_delta}")
if (
    batched_peak is not None
    and sequential_peak is not None
    and batched_peak - sequential_peak >= 4096
):
    print("classification=batched_generation_dominant")
else:
    print("classification=not_explained_by_rollout_mode")
PY
```

Expected:

```text
The output identifies batched_generation_dominant or not_explained_by_rollout_mode.
```

### Task 6: Run Decode Length Contrast

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/decode32_env4/`
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/decode128_env4/`
- Modify: none
- Test: compare decode length effect with `batch_env4`

- [ ] **Step 1: Run decode length 32**

Use the shared case runner with:

```bash
CASE_NAME="decode32_env4"
TOTAL_ENVS="4"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="32"
```

- [ ] **Step 2: Summarize decode length 32**

Use the shared case summary command with:

```bash
CASE_NAME="decode32_env4"
```

Expected:

```text
decode32_env4/summary.json exists.
```

- [ ] **Step 3: Run decode length 128**

Use the shared case runner with:

```bash
CASE_NAME="decode128_env4"
TOTAL_ENVS="4"
MAX_STEPS="512"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="128"
```

- [ ] **Step 4: Summarize decode length 128**

Use the shared case summary command with:

```bash
CASE_NAME="decode128_env4"
```

Expected:

```text
decode128_env4/summary.json exists.
```

- [ ] **Step 5: Compare decode length against default 1024**

```bash
python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
cases = [
    ("decode32_env4", 32),
    ("decode128_env4", 128),
    ("batch_env4", 1024),
]
rows = []
for case_name, max_new_token in cases:
    data = json.loads((run_root / case_name / "summary.json").read_text())
    rows.append((case_name, max_new_token, data["gpu_used_max_mib"], data["gpu_used_delta_mib"]))
for case_name, max_new_token, peak, delta in rows:
    print(f"{case_name}: max_new_token={max_new_token} peak_mib={peak} delta_mib={delta}")
peak32 = rows[0][2]
peak1024 = rows[-1][2]
if peak32 is not None and peak1024 is not None and peak1024 - peak32 >= 4096:
    print("classification=decode_kv_cache_dominant")
else:
    print("classification=weak_decode_length_dependence")
PY
```

Expected:

```text
The output identifies decode_kv_cache_dominant or weak_decode_length_dependence.
```

### Task 7: Run Episode Step and History Contrast

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/steps1024_env4/`
- Modify: none
- Test: compare longer episode rollout with `batch_env4`

- [ ] **Step 1: Run longer max steps**

Use the shared case runner with:

```bash
CASE_NAME="steps1024_env4"
TOTAL_ENVS="4"
MAX_STEPS="1024"
ROLLOUT_MODE="batched_feature_cache"
MAX_NEW_TOKEN="1024"
```

- [ ] **Step 2: Summarize longer max steps**

Use the shared case summary command with:

```bash
CASE_NAME="steps1024_env4"
```

Expected:

```text
steps1024_env4/summary.json exists.
```

- [ ] **Step 3: Compare step/history effect**

```bash
python - "${RUN_ROOT}/batch_env4/summary.json" "${RUN_ROOT}/steps1024_env4/summary.json" <<'PY'
import json
import sys
from pathlib import Path

steps512 = json.loads(Path(sys.argv[1]).read_text())
steps1024 = json.loads(Path(sys.argv[2]).read_text())
print(f'steps512_peak_mib={steps512["gpu_used_max_mib"]}')
print(f'steps1024_peak_mib={steps1024["gpu_used_max_mib"]}')
print(f'steps512_delta_mib={steps512["gpu_used_delta_mib"]}')
print(f'steps1024_delta_mib={steps1024["gpu_used_delta_mib"]}')
if (
    steps512["gpu_used_max_mib"] is not None
    and steps1024["gpu_used_max_mib"] is not None
    and steps1024["gpu_used_max_mib"] - steps512["gpu_used_max_mib"] >= 4096
):
    print("classification=history_or_long_rollout_growth")
else:
    print("classification=weak_step_count_dependence")
PY
```

Expected:

```text
The output identifies history_or_long_rollout_growth or weak_step_count_dependence.
```

### Task 8: Write Matrix Summary

**Files:**
- Create: `results/uninavid_single_gpu_vram/<RUN_ID>/matrix_summary.md`
- Modify: none
- Test: summary file contains all completed cases

- [ ] **Step 1: Generate matrix summary**

```bash
python - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
case_order = [
    "baseline_env1",
    "batch_env2",
    "batch_env4",
    "batch_env8",
    "batch_env16",
    "sequential_env4",
    "decode32_env4",
    "decode128_env4",
    "steps1024_env4",
]
rows = []
for case_name in case_order:
    path = run_root / case_name / "summary.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    rows.append(
        {
            "case": case_name,
            "peak": data["gpu_used_max_mib"],
            "delta": data["gpu_used_delta_mib"],
            "oom": data["has_cuda_oom"],
            "traceback": data["has_traceback"],
        }
    )

lines = [
    "# UniNaVid Single-GPU VRAM Investigation Summary",
    "",
    f"Run root: `{run_root}`",
    "",
    "| Case | Peak MiB | Delta MiB | CUDA OOM | Traceback |",
    "| --- | ---: | ---: | --- | --- |",
]
for row in rows:
    lines.append(
        f'| {row["case"]} | {row["peak"]} | {row["delta"]} | {row["oom"]} | {row["traceback"]} |'
    )

lines.extend(
    [
        "",
        "## Classification Checklist",
        "",
        "- Peak grows with env count: batched visual encoding, inputs_embeds, or generation KV cache.",
        "- sequential_cache much smoother than batched_feature_cache: batched generation path.",
        "- Peak grows with max_new_token: generation KV cache or long generation.",
        "- env1 still swings heavily: Habitat reset/scene behavior or allocator.",
        "- nvidia-smi swings without matching PyTorch allocated data: Habitat/OpenGL or non-PyTorch CUDA.",
        "- allocated drops but reserved remains high after instrumentation: CUDA allocator retention.",
        "",
        "## Provisional Conclusion",
        "",
        "Conclusion pending Task 9 evidence classification.",
    ]
)
(run_root / "matrix_summary.md").write_text("\n".join(lines) + "\n")
print(run_root / "matrix_summary.md")
PY
```

Expected:

```text
matrix_summary.md is printed and contains a row for every completed case.
```

- [ ] **Step 2: Inspect summary**

```bash
sed -n '1,220p' "${RUN_ROOT}/matrix_summary.md"
```

Expected:

```text
The table contains baseline_env1, batch_env2, batch_env4, batch_env8, sequential_env4,
decode32_env4, decode128_env4, and steps1024_env4 if all required cases ran.
```

### Task 9: Decision Checkpoint

**Files:**
- Modify: `results/uninavid_single_gpu_vram/<RUN_ID>/matrix_summary.md`
- Test: no automated test

- [ ] **Step 1: Choose evidence-backed classification**

Read `matrix_summary.md` and choose one:

```text
batched_generation_or_kv_cache
habitat_or_opengl_memory
cuda_allocator_retention
navigation_history_cache_growth
insufficient_evidence_needs_instrumentation
```

- [ ] **Step 2: Update provisional conclusion**

Edit only `results/uninavid_single_gpu_vram/<RUN_ID>/matrix_summary.md` and replace:

```text
Conclusion pending Task 9 evidence classification.
```

with one of:

```text
The dominant source is batched generation or KV cache because peak memory grows with env count, sequential_cache is smoother, or peak memory grows with max_new_token.
```

```text
The dominant source is Habitat/OpenGL memory because env1 already swings heavily and process-level PyTorch evidence does not explain the nvidia-smi movement.
```

```text
The dominant source is CUDA allocator retention because active allocations fall after peaks while reserved memory remains high.
```

```text
The dominant source is UniNaVid navigation history cache growth because memory grows inside longer episode windows and correlates with step count.
```

```text
The no-code matrix is insufficient. The next step is a separate opt-in instrumentation patch scoped to rlinf/models/embodiment/uninavid/ and rlinf/envs/habitat/.
```

- [ ] **Step 3: Report evidence before fixes**

Prepare a concise report with:

```text
Run root
Cases completed
Peak and delta table
Chosen classification
Whether instrumentation is needed
No proposed code fix unless the classification is unambiguous
```

Expected:

```text
The report states a root-cause hypothesis and cites concrete case evidence.
```

### Task 10: Commit Investigation Artifacts Only If Requested

**Files:**
- Optional commit: `results/uninavid_single_gpu_vram/<RUN_ID>/matrix_summary.md`
- Modify: none by default
- Test: git status check

- [ ] **Step 1: Check git status**

```bash
git status --short
```

Expected:

```text
Tracked source files under rlinf/ and examples/ are unchanged by this investigation plan.
```

- [ ] **Step 2: Do not commit bulky runtime artifacts by default**

```bash
find "${RUN_ROOT}" -type f | sort | sed -n '1,120p'
```

Expected:

```text
The runtime artifacts remain under results/ unless the user explicitly requests a commit.
```

- [ ] **Step 3: Commit only a small summary if explicitly requested**

```bash
git add "${RUN_ROOT}/matrix_summary.md"
git commit -m "docs: record uninavid single-gpu vram investigation"
```

Expected:

```text
A commit is created only when the user explicitly requested committing the summary.
```
