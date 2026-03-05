#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

BUILD_TYPE=""
PREWARM_CMAKE="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-prewarm-cmake)
            PREWARM_CMAKE="false"
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_BUILD_DIR REMOTE_LOG_DIR REMOTE_CCACHE_DIR REMOTE_VENV_DIR

BUILD_TYPE="${BUILD_TYPE:-${DEFAULT_BUILD_TYPE}}"

run_remote_script "${BUILD_TYPE}" "${WITH_UT}" "${WITH_DISKANN}" "${WITH_PIPNN}" "${REMOTE_REPO_DIR}" "${REMOTE_BUILD_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CCACHE_DIR}" "${REMOTE_VENV_DIR}" "${PREWARM_CMAKE}" <<'EOF'
set -euo pipefail

build_type="$1"
with_ut="$2"
with_diskann="$3"
with_pipnn="$4"
repo_dir="$5"
build_dir="$6"
log_dir="$7"
ccache_dir="$8"
venv_dir="$9"
prewarm_cmake="${10}"

mkdir -p "${build_dir}" "${log_dir}" "${ccache_dir}"
log_file="${log_dir}/build_$(date -u +%Y%m%dT%H%M%SZ).log"

export PATH="${venv_dir}/bin:${PATH}"
export CCACHE_DIR="${ccache_dir}"
export CONAN_RETRY=1
export CONAN_RETRY_WAIT=5

clean_corrupted_conan_cache() {
    rm -rf "${HOME}/.conan/data/cmake/3.30.5" 2>/dev/null || true
    find "${HOME}/.conan/data/cmake/3.30.5" -type f -name '*.tgz' -delete 2>/dev/null || true
    find "${HOME}/.conan/data/cmake/3.30.5" -type f -name 'metadata.json' -delete 2>/dev/null || true
}

prewarm_cmake_package() {
    local attempt
    for attempt in 1 2 3; do
        echo "[build] prewarming cmake/3.30.5 attempt=${attempt}"
        clean_corrupted_conan_cache
        if CONAN_RETRY=0 CONAN_RETRY_WAIT=0 conan download cmake/3.30.5@ -r conancenter; then
            return 0
        fi
        sleep 3
    done
    return 1
}

run_conan_install() {
    local attempt
    for attempt in 1 2; do
        echo "[build] conan install attempt=${attempt}"
        clean_corrupted_conan_cache
        if CONAN_RETRY=0 CONAN_RETRY_WAIT=0 conan install "${repo_dir}" --build=missing \
            -o with_ut="${with_ut}" \
            -o with_diskann="${with_diskann}" \
            -o with_pipnn="${with_pipnn}" \
            -s compiler.libcxx=libstdc++11 \
            -s build_type="${build_type}"; then
            return 0
        fi
        sleep 3
    done
    return 1
}

{
    echo "[build] repo_dir=${repo_dir}"
    echo "[build] build_dir=${build_dir}"
    echo "[build] build_type=${build_type}"
    echo "[build] with_pipnn=${with_pipnn}"
    echo "[build] commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    echo "[build] prewarm_cmake=${prewarm_cmake}"
    cd "${build_dir}"

    if [[ "${prewarm_cmake}" == "true" ]]; then
        if ! prewarm_cmake_package; then
            echo "[build] prewarm failed after retries"
        fi
    fi

    run_conan_install

    conan build "${repo_dir}"
} 2>&1 | tee "${log_file}"

printf 'build=ok\n'
printf 'log=%s\n' "${log_file}"
EOF
