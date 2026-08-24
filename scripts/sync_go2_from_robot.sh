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
  scripts/sync_go2_from_robot.sh --host SSH_TARGET [options]

Copy the three Go2 VLN ROS2 source packages from the robot into this RLinf
checkout. Robot-side build/, install/, and log/ directories are never copied.

Options:
  --host SSH_TARGET         Robot SSH source or ~/.ssh/config alias
                            (for example unitree@192.168.3.15 or go2).
  --remote-workspace PATH   Robot colcon workspace.
                            Default: /home/unitree/qianyx/go2_vln_ws
  --port PORT               SSH port. Default: 22
  --dry-run                 Print changes without copying files.
  --delete                  Delete extra local files inside the three package
                            directories so they exactly match robot source.
  -h, --help                Show this help.

Examples:
  scripts/sync_go2_from_robot.sh --host unitree@192.168.3.15 --dry-run
  scripts/sync_go2_from_robot.sh --host unitree@192.168.3.15
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

SSH_ARGS=(-p "${SSH_PORT}" -o BatchMode=yes -o ConnectTimeout=8)
RSYNC_SHELL="ssh -p ${SSH_PORT} -o BatchMode=yes -o ConnectTimeout=8"
RSYNC_ARGS=(
    -az
    --itemize-changes
    --out-format='    %i  %n%L'
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

echo "[2/4] Checking rsync and all three source packages on the robot"
ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" \
    "command -v rsync >/dev/null && test -d '${REMOTE_WORKSPACE}/src/go2_vln_interfaces' && test -d '${REMOTE_WORKSPACE}/src/go2_vln_executor' && test -d '${REMOTE_WORKSPACE}/src/go2_vln_socket_gateway'"

if ${DRY_RUN}; then
    echo "[3/4] Dry run: local directories will not be created"
else
    echo "[3/4] Creating the local ROS2 source directory if needed"
    mkdir -p "${LOCAL_SRC}"
fi

echo "[4/4] Synchronizing robot ROS2 packages into this RLinf checkout"
if ${DRY_RUN}; then
    echo "Files that would be synchronized:"
else
    echo "Files synchronized:"
fi
for package in "${PACKAGES[@]}"; do
    echo "  <- ${package}/"
    rsync "${RSYNC_ARGS[@]}" -e "${RSYNC_SHELL}" \
        "${REMOTE_HOST}:${REMOTE_WORKSPACE}/src/${package}/" \
        "${LOCAL_SRC}/${package}/"
done

if ${DRY_RUN}; then
    echo "Dry run complete: no files were changed."
else
    echo "Download complete. Robot source is now under ${LOCAL_SRC}."
    echo "If a package had no paths listed below it, that package was already synchronized."
    echo "Review the result with: git -C '${RLINF_ROOT}' status --short"
fi
