#!/usr/bin/env bash

# Allow an accidental `sh script.sh ...` invocation to restart under Bash.
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLINF_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOCAL_SRC="${RLINF_ROOT}/examples/embodiment/go2_vln_ros2/src"

REMOTE_HOST="${GO2_SSH_HOST:-}"
REMOTE_WORKSPACE="${GO2_REMOTE_WORKSPACE:-/home/unitree/qianyx/go2_vln_ws}"
SSH_PORT="${GO2_SSH_PORT:-22}"
DRY_RUN=false
DELETE=false

PACKAGES=(
    go2_vln_interfaces
    go2_vln_executor
    go2_vln_socket_gateway
)

usage() {
    cat <<'EOF'
Usage:
  scripts/sync_go2_to_robot.sh --host SSH_TARGET [options]

Copy the three Go2 VLN ROS2 source packages from this RLinf checkout to the
robot. Robot-side build/, install/, and log/ directories are never copied.

Options:
  --host SSH_TARGET         Robot SSH destination or ~/.ssh/config alias
                            (for example unitree@192.168.3.15 or go2).
  --remote-workspace PATH   Robot colcon workspace.
                            Default: /home/unitree/qianyx/go2_vln_ws
  --port PORT               SSH port. Default: 22
  --dry-run                 Print changes without copying files.
  --delete                  Delete extra files inside the three remote package
                            directories so they exactly match local source.
  -h, --help                Show this help.

Examples:
  scripts/sync_go2_to_robot.sh --host unitree@192.168.3.15 --dry-run
  scripts/sync_go2_to_robot.sh --host unitree@192.168.3.15
EOF
}

while (($#)); do
    case "$1" in
        --host)
            [[ $# -ge 2 ]] || { echo "ERROR: --host requires a value" >&2; exit 2; }
            REMOTE_HOST="$2"
            shift 2
            ;;
        --remote-workspace)
            [[ $# -ge 2 ]] || { echo "ERROR: --remote-workspace requires a value" >&2; exit 2; }
            REMOTE_WORKSPACE="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || { echo "ERROR: --port requires a value" >&2; exit 2; }
            SSH_PORT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --delete)
            DELETE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${REMOTE_HOST}" ]] || { echo "ERROR: use --host SSH_TARGET or set GO2_SSH_HOST" >&2; exit 2; }
[[ "${REMOTE_HOST}" =~ ^([A-Za-z0-9._-]+@)?[A-Za-z0-9._-]+$ ]] || {
    echo "ERROR: --host must be an SSH alias, hostname, or USER@HOST" >&2
    exit 2
}
[[ "${REMOTE_WORKSPACE}" =~ ^/[A-Za-z0-9._/-]+$ ]] || {
    echo "ERROR: --remote-workspace must be an absolute path without spaces" >&2
    exit 2
}
[[ "${SSH_PORT}" =~ ^[0-9]+$ ]] && ((SSH_PORT >= 1 && SSH_PORT <= 65535)) || {
    echo "ERROR: invalid SSH port: ${SSH_PORT}" >&2
    exit 2
}

command -v ssh >/dev/null || { echo "ERROR: ssh is not installed" >&2; exit 1; }
command -v rsync >/dev/null || { echo "ERROR: rsync is not installed locally" >&2; exit 1; }

for package in "${PACKAGES[@]}"; do
    [[ -d "${LOCAL_SRC}/${package}" ]] || {
        echo "ERROR: local package is missing: ${LOCAL_SRC}/${package}" >&2
        exit 1
    }
done

SSH_ARGS=(-p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout=8)
RSYNC_SHELL="ssh -p ${SSH_PORT} -o BatchMode=yes -o ConnectTimeout=8"
RSYNC_ARGS=(
    -az
    --itemize-changes
    --human-readable
    --exclude=.git/
    --exclude=__pycache__/
    --exclude='*.py[cod]'
    --exclude=.pytest_cache/
    --exclude=build/
    --exclude=install/
    --exclude=log/
    --exclude=third_party/
)
${DRY_RUN} && RSYNC_ARGS+=(--dry-run)
${DELETE} && RSYNC_ARGS+=(--delete)

echo "[1/4] Checking passwordless SSH connection to ${REMOTE_HOST}:${SSH_PORT}"
ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" true

echo "[2/4] Checking rsync on the robot"
ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "command -v rsync >/dev/null"

if ${DRY_RUN}; then
    echo "[3/4] Dry run: remote directories will not be created"
else
    echo "[3/4] Creating ${REMOTE_WORKSPACE}/src and package directories"
    ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
        "mkdir -p '${REMOTE_WORKSPACE}/src/go2_vln_interfaces' '${REMOTE_WORKSPACE}/src/go2_vln_executor' '${REMOTE_WORKSPACE}/src/go2_vln_socket_gateway'"
fi

echo "[4/4] Synchronizing local ROS2 packages to the robot"
for package in "${PACKAGES[@]}"; do
    echo "  -> ${package}"
    rsync "${RSYNC_ARGS[@]}" -e "${RSYNC_SHELL}" \
        "${LOCAL_SRC}/${package}/" \
        "${REMOTE_HOST}:${REMOTE_WORKSPACE}/src/${package}/"
done

if ${DRY_RUN}; then
    echo "Dry run complete: no files were changed."
else
    echo "Upload complete. Source code is now under ${REMOTE_WORKSPACE}/src on ${REMOTE_HOST}."
    echo "Next on the robot: source ROS2 and unitree_ros2, then run colcon build."
fi
