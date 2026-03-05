#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

FILTER=""
RUN_ALL="false"
BUILD_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER="$2"
            shift 2
            ;;
        --all)
            RUN_ALL="true"
            shift
            ;;
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_BUILD_DIR REMOTE_LOG_DIR

BUILD_TYPE="${BUILD_TYPE:-${DEFAULT_BUILD_TYPE}}"

run_remote_script "${RUN_ALL}" "${FILTER}" "${BUILD_TYPE}" "${REMOTE_REPO_DIR}" "${REMOTE_BUILD_DIR}" "${REMOTE_LOG_DIR}" <<'EOF'
set -euo pipefail

run_all="$1"
filter="$2"
build_type="$3"
repo_dir="$4"
build_dir="$5"
log_dir="$6"

test_bin="${repo_dir}/build/${build_type}/tests/ut/knowhere_tests"
if [[ ! -x "${test_bin}" ]]; then
    echo "missing test binary: ${test_bin}" >&2
    exit 1
fi

# Extract RUNPATH from binary and set LD_LIBRARY_PATH
# This is needed because some conan-built libraries lack RUNPATH
runpath=$(readelf -d "${test_bin}" 2>/dev/null | sed -n 's/.*Library runpath: \[\(.*\)\]/\1/p' | head -1)
if [[ -n "${runpath}" ]]; then
    export LD_LIBRARY_PATH="${runpath}:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${log_dir}"
log_file="${log_dir}/test_$(date -u +%Y%m%dT%H%M%SZ).log"

{
    echo "[test] binary=${test_bin}"
    if [[ "${run_all}" == "true" ]]; then
        "${test_bin}"
    elif [[ -n "${filter}" ]]; then
        "${test_bin}" "${filter}"
    else
        "${test_bin}" "[pipnn]"
    fi
} 2>&1 | tee "${log_file}"

printf 'test=ok
'
printf 'log=%s
' "${log_file}"
EOF
