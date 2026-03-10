#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <build|test> [options]" >&2
    exit 1
fi

SUBCOMMAND="$1"
shift

BUILD_TYPE=""
FILTER=""
PREWARM_CMAKE="true"
LABEL=""
ENV_VARS=()

sanitize_label() {
    local raw="${1:-}"
    if [[ -z "${raw}" ]]; then
        return 0
    fi
    printf '%s' "${raw}" | tr -cs 'A-Za-z0-9._-' '_'
}

validate_env_assignment() {
    local assignment="${1:-}"
    if [[ ! "${assignment}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
        echo "invalid --env assignment: ${assignment}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --label)
            LABEL="$(sanitize_label "$2")"
            shift 2
            ;;
        --env)
            validate_env_assignment "$2"
            ENV_VARS+=("$2")
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

case "${SUBCOMMAND}" in
    build)
        run_remote_script "${BUILD_TYPE}" "${WITH_UT}" "${WITH_DISKANN}" "${REMOTE_REPO_DIR}" "${REMOTE_BUILD_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CCACHE_DIR}" "${REMOTE_VENV_DIR}" "${PREWARM_CMAKE}" <<'EOF'
set -euo pipefail

build_type="$1"
with_ut="$2"
with_diskann="$3"
repo_dir="$4"
build_dir="$5"
log_dir="$6"
ccache_dir="$7"
venv_dir="$8"
prewarm_cmake="$9"

mkdir -p "${build_dir}" "${log_dir}" "${ccache_dir}"
log_file="${log_dir}/build_bg_$(date -u +%Y%m%dT%H%M%SZ).log"
cmd=$(cat <<CMD
export PATH="${venv_dir}/bin:\$PATH"
export CCACHE_DIR="${ccache_dir}"
export CONAN_RETRY=1
export CONAN_RETRY_WAIT=5
clean_corrupted_conan_cache() {
  rm -rf "\$HOME/.conan/data/cmake/3.30.5" 2>/dev/null || true
  find "\$HOME/.conan/data/cmake/3.30.5" -type f -name '*.tgz' -delete 2>/dev/null || true
  find "\$HOME/.conan/data/cmake/3.30.5" -type f -name 'metadata.json' -delete 2>/dev/null || true
}
prewarm_cmake_package() {
  local attempt
  for attempt in 1 2 3; do
    echo "[build] prewarming cmake/3.30.5 attempt=\${attempt}"
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
    echo "[build] conan install attempt=\${attempt}"
    clean_corrupted_conan_cache
    if CONAN_RETRY=0 CONAN_RETRY_WAIT=0 conan install "${repo_dir}" --build=missing -o with_ut="${with_ut}" -o with_diskann="${with_diskann}" -s compiler.libcxx=libstdc++11 -s build_type="${build_type}"; then
      return 0
    fi
    sleep 3
  done
  return 1
}
cd "${build_dir}"
echo "[build] commit=\$(git -C "${repo_dir}" rev-parse HEAD)"
echo "[build] prewarm_cmake=${prewarm_cmake}"
if [[ "${prewarm_cmake}" == "true" ]]; then
  if ! prewarm_cmake_package; then
    echo "[build] prewarm failed after retries"
  fi
fi
run_conan_install
conan build "${repo_dir}"
CMD
)
nohup bash -lc "${cmd}" >"${log_file}" 2>&1 </dev/null &
pid=$!
printf 'task=build\n'
printf 'pid=%s\n' "${pid}"
printf 'log=%s\n' "${log_file}"
EOF
        ;;
    test)
        run_remote_script "${BUILD_TYPE}" "${FILTER:-[pipnn]}" "${REMOTE_REPO_DIR}" "${REMOTE_LOG_DIR}" "${#ENV_VARS[@]}" "${LABEL:-__none__}" ${ENV_VARS[@]+"${ENV_VARS[@]}"} <<'EOF'
set -euo pipefail

build_type="$1"
filter="$2"
repo_dir="$3"
log_dir="$4"
env_count="$5"
label="${6:-__none__}"
shift 6

for ((i = 0; i < env_count; ++i)); do
    export "$1"
    shift
done

test_bin="${repo_dir}/build/${build_type}/tests/ut/knowhere_tests"
if [[ ! -x "${test_bin}" ]]; then
    echo "missing test binary: ${test_bin}" >&2
    exit 1
fi

runpath=$(readelf -d "${test_bin}" 2>/dev/null | sed -n 's/.*Library runpath: \[\(.*\)\]/\1/p' | head -1)
if [[ -n "${runpath}" ]]; then
    export LD_LIBRARY_PATH="${runpath}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${log_dir}"
log_prefix="test_bg"
if [[ -n "${label}" && "${label}" != "__none__" ]]; then
    log_prefix="test_${label}"
fi
log_file="${log_dir}/${log_prefix}_$(date -u +%Y%m%dT%H%M%SZ).log"
cmd=$(cat <<CMD
export TMPDIR="/data/tmp"
mkdir -p "/data/tmp"
cd "${repo_dir}/build/${build_type}"
"${test_bin}" "${filter}"
CMD
)
nohup bash -lc "${cmd}" >"${log_file}" 2>&1 </dev/null &
pid=$!
printf 'task=test\n'
printf 'pid=%s\n' "${pid}"
printf 'log=%s\n' "${log_file}"
EOF
        ;;
    *)
        echo "unsupported subcommand: ${SUBCOMMAND}" >&2
        exit 1
        ;;
esac
