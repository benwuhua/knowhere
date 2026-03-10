#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

FILTER=""
RUN_ALL="false"
BUILD_TYPE=""
POLL_INTERVAL_SECONDS="5"

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
        --poll-interval)
            POLL_INTERVAL_SECONDS="$2"
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
RUN_ID="$(timestamp_utc)_$$"
FORWARD_BASE_FBIN="${KNOWHERE_PIPNN_BASE_FBIN:-}"
FORWARD_QUERY_FBIN="${KNOWHERE_PIPNN_QUERY_FBIN:-}"
FORWARD_GT_IBIN="${KNOWHERE_PIPNN_GT_IBIN:-}"
FORWARD_DATASET_LABEL="${KNOWHERE_PIPNN_DATASET_LABEL:-}"

REMOTE_METADATA_RAW="$(run_remote_script "${RUN_ALL}" "${FILTER}" "${BUILD_TYPE}" "${REMOTE_REPO_DIR}" "${REMOTE_BUILD_DIR}" "${REMOTE_LOG_DIR}" "${RUN_ID}" "${FORWARD_BASE_FBIN}" "${FORWARD_QUERY_FBIN}" "${FORWARD_GT_IBIN}" "${FORWARD_DATASET_LABEL}" <<'EOF'
set -euo pipefail

run_all="$1"
filter="$2"
build_type="$3"
repo_dir="$4"
build_dir="$5"
log_dir="$6"
run_id="$7"
forward_base_fbin="${8:-}"
forward_query_fbin="${9:-}"
forward_gt_ibin="${10:-}"
forward_dataset_label="${11:-}"

test_bin="${repo_dir}/build/${build_type}/tests/ut/knowhere_tests"
if [[ ! -x "${test_bin}" ]]; then
    echo "missing test binary: ${test_bin}" >&2
    exit 1
fi

runpath=$(readelf -d "${test_bin}" 2>/dev/null | sed -n 's/.*Library runpath: \[\(.*\)\]/\1/p' | head -1)
mkdir -p "${log_dir}"
log_file="${log_dir}/test_${run_id}.log"
status_file="${log_dir}/test_${run_id}.status"
lock_file="${log_dir}/test.${build_type}.lock"

if [[ "${run_all}" == "true" ]]; then
    test_args=()
elif [[ -n "${filter}" ]]; then
    test_args=("${filter}")
else
    test_args=("[pipnn]")
fi

if [[ "${run_all}" == "true" ]]; then
    filter_token="__ALL__"
elif [[ ${#test_args[@]} -gt 0 ]]; then
    filter_token="${test_args[0]}"
else
    filter_token="[pipnn]"
fi

bash_cmd="$(printf '%q ' "${test_bin}" "${test_args[@]}")"
if [[ -n "${runpath}" ]]; then
    bash_cmd="export LD_LIBRARY_PATH=$(printf '%q' "${runpath}:${LD_LIBRARY_PATH:-}"); ${bash_cmd}"
fi
bash_cmd="cd $(printf '%q' "${repo_dir}/build/${build_type}") && ${bash_cmd}"

nohup env \
    TEST_LOCK_FILE="${lock_file}" \
    TEST_FILTER_TOKEN="${filter_token}" \
    TEST_LOG_FILE="${log_file}" \
    TEST_STATUS_FILE="${status_file}" \
    TEST_RUN_ID="${run_id}" \
    TEST_COMMAND="${bash_cmd}" \
    KNOWHERE_PIPNN_BASE_FBIN="${forward_base_fbin}" \
    KNOWHERE_PIPNN_QUERY_FBIN="${forward_query_fbin}" \
    KNOWHERE_PIPNN_GT_IBIN="${forward_gt_ibin}" \
    KNOWHERE_PIPNN_DATASET_LABEL="${forward_dataset_label}" \
    bash -lc '
set -euo pipefail
cleanup() {
    flock -u 9 || true
}
exec 9>"${TEST_LOCK_FILE}"
flock -n 9 || {
    printf "status=conflict\nrun_id=%s\nfilter=%s\nfinished_at=%s\nmessage=another test run is still active for this build type\n" \
        "${TEST_RUN_ID}" "${TEST_FILTER_TOKEN}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${TEST_STATUS_FILE}"
    exit 91
}
trap cleanup EXIT
printf "status=running\nrun_id=%s\nfilter=%s\nstarted_at=%s\n" \
    "${TEST_RUN_ID}" "${TEST_FILTER_TOKEN}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${TEST_STATUS_FILE}"
{
    echo "[test] run_id=${TEST_RUN_ID}"
    echo "[test] filter=${TEST_FILTER_TOKEN}"
    echo "[test] cwd=$(pwd)"
    echo "[test] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    eval "${TEST_COMMAND}"
    rc=$?
    set -e
    printf "status=%s\nrun_id=%s\nfilter=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\n" \
        "$([[ ${rc} -eq 0 ]] && printf ok || printf failed)" \
        "${TEST_RUN_ID}" "${TEST_FILTER_TOKEN}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TEST_LOG_FILE}" >"${TEST_STATUS_FILE}"
    printf "[test] exit_code=%s\n[test] finished_at=%s\n" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${TEST_LOG_FILE}"
    exit "${rc}"
} >"${TEST_LOG_FILE}" 2>&1
' >/dev/null 2>&1 &
pid=$!
printf 'pid=%s\n' "${pid}"
printf 'log=%s\n' "${log_file}"
printf 'status_file=%s\n' "${status_file}"
printf 'filter=%s\n' "${filter_token}"
EOF
)"

PID=""
LOG_FILE=""
STATUS_FILE=""
FILTER_TOKEN=""
while IFS= read -r line; do
    case "${line}" in
        pid=*)
            PID="${line#pid=}"
            ;;
        log=*)
            LOG_FILE="${line#log=}"
            ;;
        status_file=*)
            STATUS_FILE="${line#status_file=}"
            ;;
        filter=*)
            FILTER_TOKEN="${line#filter=}"
            ;;
    esac
done <<<"${REMOTE_METADATA_RAW}"

if [[ -z "${PID}" || -z "${LOG_FILE}" || -z "${STATUS_FILE}" ]]; then
    echo "failed to initialize remote test run" >&2
    exit 1
fi

while true; do
    STATUS_CONTENT="$(run_ssh "if [[ -f $(printf '%q' "${STATUS_FILE}") ]]; then cat $(printf '%q' "${STATUS_FILE}"); fi")"
    if grep -q '^status=ok$' <<<"${STATUS_CONTENT}"; then
        printf 'test=ok\n'
        printf 'pid=%s\n' "${PID}"
        printf 'filter=%s\n' "${FILTER_TOKEN}"
        printf 'log=%s\n' "${LOG_FILE}"
        printf 'status_file=%s\n' "${STATUS_FILE}"
        printf '%s\n' "${STATUS_CONTENT}"
        exit 0
    fi
    if grep -q '^status=failed$' <<<"${STATUS_CONTENT}"; then
        printf 'test=failed\n' >&2
        printf 'pid=%s\n' "${PID}" >&2
        printf 'filter=%s\n' "${FILTER_TOKEN}" >&2
        printf 'log=%s\n' "${LOG_FILE}" >&2
        printf 'status_file=%s\n' "${STATUS_FILE}" >&2
        printf '%s\n' "${STATUS_CONTENT}" >&2
        exit_code="$(awk -F= '/^exit_code=/{print $2; exit}' <<<"${STATUS_CONTENT}")"
        exit "${exit_code:-1}"
    fi
    if grep -q '^status=conflict$' <<<"${STATUS_CONTENT}"; then
        printf 'test=conflict\n' >&2
        printf 'pid=%s\n' "${PID}" >&2
        printf 'filter=%s\n' "${FILTER_TOKEN}" >&2
        printf 'log=%s\n' "${LOG_FILE}" >&2
        printf 'status_file=%s\n' "${STATUS_FILE}" >&2
        printf '%s\n' "${STATUS_CONTENT}" >&2
        exit 91
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done
