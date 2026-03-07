#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

cat >"${TMP_DIR}/remote.env" <<EOF
REMOTE_HOST=fake-host
REMOTE_USER=fake-user
REMOTE_REPO_DIR=/remote/repo
REMOTE_BUILD_DIR=/remote/build
REMOTE_LOG_DIR=/remote/logs
REMOTE_CCACHE_DIR=/remote/ccache
REMOTE_VENV_DIR=/remote/venv
EOF

cat >"${TMP_DIR}/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
capture_dir="${TEST_CAPTURE_DIR:?}"
printf '%s\n' "$@" >"${capture_dir}/ssh_args.txt"
cat >"${capture_dir}/remote_script.sh"
EOF
chmod +x "${TMP_DIR}/ssh"

export KNOWHERE_REMOTE_ENV="${TMP_DIR}/remote.env"
export TEST_CAPTURE_DIR="${TMP_DIR}"
export PATH="${TMP_DIR}:${PATH}"

"${REPO_ROOT}/scripts/remote/run-bg.sh" test \
    --filter "[pipnn_diskann][e2e][recall]" \
    --type Release \
    --label baseline \
    --env KNOWHERE_PIPNN_LEAF_MAX_SIZE=1000 \
    --env KNOWHERE_PIPNN_FANOUT_L1=10 >/dev/null

grep -F -- 'test_bin="${repo_dir}/build/${build_type}/tests/ut/knowhere_tests"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'runpath=$(readelf -d "${test_bin}"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'label="$3"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'env_count="$6"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'for ((i = 0; i < env_count; ++i)); do' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'export "$1"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'log_prefix="test_${label}"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'log_file="${log_dir}/${log_prefix}_$(date -u +%Y%m%dT%H%M%SZ).log"' "${TMP_DIR}/remote_script.sh" >/dev/null
grep -F -- 'KNOWHERE_PIPNN_LEAF_MAX_SIZE=1000' "${TMP_DIR}/ssh_args.txt" >/dev/null
grep -F -- 'KNOWHERE_PIPNN_FANOUT_L1=10' "${TMP_DIR}/ssh_args.txt" >/dev/null
