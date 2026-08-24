#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    echo "socket_setup.sh must be sourced from Bash." >&2
    return 2 2>/dev/null || exit 2
fi

_GO2_SOCKET_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_GO2_SOCKET_RLINF_ROOT="$(cd -- "${_GO2_SOCKET_SCRIPT_DIR}/.." && pwd)"

if ! cd "${_GO2_SOCKET_RLINF_ROOT}"; then
    echo "Cannot enter RLinf root: ${_GO2_SOCKET_RLINF_ROOT}" >&2
    unset _GO2_SOCKET_SCRIPT_DIR _GO2_SOCKET_RLINF_ROOT
    return 1 2>/dev/null || exit 1
fi
if ! source .venv-go2/bin/activate; then
    echo "Cannot activate ${_GO2_SOCKET_RLINF_ROOT}/.venv-go2" >&2
    unset _GO2_SOCKET_SCRIPT_DIR _GO2_SOCKET_RLINF_ROOT
    return 1 2>/dev/null || exit 1
fi

export EMBODIED_PATH="${_GO2_SOCKET_RLINF_ROOT}/examples/embodiment"
export PYTHONPATH="${_GO2_SOCKET_RLINF_ROOT}:${PYTHONPATH:-}"

if [[ -z "${GO2_VLN_TOKEN:-}" ]]; then
    echo "GO2_VLN_TOKEN is not set." >&2
    echo "Set the same secret on the RLinf and Go2 machines." >&2
    unset _GO2_SOCKET_SCRIPT_DIR _GO2_SOCKET_RLINF_ROOT
    return 1 2>/dev/null || exit 1
fi

if ! python -c "import rlinf, torch, ray; print('RLinf socket runtime OK')"; then
    echo "RLinf socket runtime dependency check failed." >&2
    unset _GO2_SOCKET_SCRIPT_DIR _GO2_SOCKET_RLINF_ROOT
    return 1 2>/dev/null || exit 1
fi

unset _GO2_SOCKET_SCRIPT_DIR _GO2_SOCKET_RLINF_ROOT
