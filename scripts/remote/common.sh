#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="${HOME}/.config/knowhere-x86-remote/remote.env"

expand_path() {
    local path="${1:-}"
    if [[ -z "${path}" ]]; then
        return 0
    fi
    if [[ "${path}" == "~" ]]; then
        printf '%s\n' "${HOME}"
    elif [[ "${path}" == ~/* ]]; then
        printf '%s/%s\n' "${HOME}" "${path#~/}"
    else
        printf '%s\n' "${path}"
    fi
}

ensure_local_command() {
    local cmd="${1}"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "missing local command: ${cmd}" >&2
        exit 1
    fi
}

load_remote_config() {
    local env_file="${KNOWHERE_REMOTE_ENV:-${DEFAULT_ENV_FILE}}"
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
    fi

    REMOTE_HOST="${REMOTE_HOST:-}"
    REMOTE_USER="${REMOTE_USER:-root}"
    REMOTE_PORT="${REMOTE_PORT:-22}"
    REMOTE_WORK_ROOT="${REMOTE_WORK_ROOT:-/data/work}"
    REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-${REMOTE_WORK_ROOT}/knowhere-src}"
    REMOTE_BUILD_DIR="${REMOTE_BUILD_DIR:-${REMOTE_WORK_ROOT}/knowhere-build}"
    REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-${REMOTE_WORK_ROOT}/logs}"
    REMOTE_CCACHE_DIR="${REMOTE_CCACHE_DIR:-${REMOTE_WORK_ROOT}/ccache}"
    REMOTE_VENV_DIR="${REMOTE_VENV_DIR:-${REMOTE_WORK_ROOT}/pyenv}"
    DEFAULT_BRANCH="${DEFAULT_BRANCH:-feat/pipnn-diskann}"
    DEFAULT_BUILD_TYPE="${DEFAULT_BUILD_TYPE:-Debug}"
    WITH_UT="${WITH_UT:-True}"
    WITH_DISKANN="${WITH_DISKANN:-True}"
    WITH_PIPNN="${WITH_PIPNN:-True}"
    REMOTE_REPO_URL="${REMOTE_REPO_URL:-$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)}"
    CONANCENTER_REMOTE_URL="${CONANCENTER_REMOTE_URL:-https://center.conan.io}"
    DEFAULT_CONAN_LOCAL_URL="${DEFAULT_CONAN_LOCAL_URL:-https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local}"
    CONANCENTER_REMOTE_USER="${CONANCENTER_REMOTE_USER:-}"
    CONANCENTER_REMOTE_PASSWORD="${CONANCENTER_REMOTE_PASSWORD:-}"
    SSH_IDENTITY_FILE="$(expand_path "${SSH_IDENTITY_FILE:-}")"
}

require_remote_config() {
    local missing=0
    local var
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            echo "missing config: ${var}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        echo "load a config file via KNOWHERE_REMOTE_ENV or ${DEFAULT_ENV_FILE}" >&2
        exit 1
    fi
}

remote_target() {
    printf '%s@%s' "${REMOTE_USER}" "${REMOTE_HOST}"
}

ssh_base_args() {
    SSH_BASE_ARGS=(-o StrictHostKeyChecking=accept-new -p "${REMOTE_PORT}")
    if [[ -n "${SSH_IDENTITY_FILE}" ]]; then
        SSH_BASE_ARGS+=(-i "${SSH_IDENTITY_FILE}" -o IdentitiesOnly=yes)
    fi
}

run_ssh() {
    local target
    target="$(remote_target)"
    ssh_base_args
    ssh "${SSH_BASE_ARGS[@]}" "${target}" "$@"
}

run_remote_script() {
    local target
    target="$(remote_target)"
    ssh_base_args
    ssh "${SSH_BASE_ARGS[@]}" "${target}" bash -s -- "$@"
}

run_rsync() {
    ssh_base_args
    rsync -az -e "ssh ${SSH_BASE_ARGS[*]}" "$@"
}

timestamp_utc() {
    date -u +"%Y%m%dT%H%M%SZ"
}

print_config_summary() {
    cat <<EOF
remote_host=${REMOTE_HOST}
remote_user=${REMOTE_USER}
remote_port=${REMOTE_PORT}
remote_work_root=${REMOTE_WORK_ROOT}
remote_repo_dir=${REMOTE_REPO_DIR}
remote_build_dir=${REMOTE_BUILD_DIR}
remote_log_dir=${REMOTE_LOG_DIR}
remote_ccache_dir=${REMOTE_CCACHE_DIR}
remote_venv_dir=${REMOTE_VENV_DIR}
default_branch=${DEFAULT_BRANCH}
default_build_type=${DEFAULT_BUILD_TYPE}
with_ut=${WITH_UT}
with_diskann=${WITH_DISKANN}
with_pipnn=${WITH_PIPNN}
conancenter_remote_url=${CONANCENTER_REMOTE_URL}
default_conan_local_url=${DEFAULT_CONAN_LOCAL_URL}
conancenter_remote_user=${CONANCENTER_REMOTE_USER:-<unset>}
conancenter_remote_password_configured=$([[ -n "${CONANCENTER_REMOTE_PASSWORD}" ]] && printf yes || printf no)
ssh_identity_file=${SSH_IDENTITY_FILE:-<default>}
EOF
}
