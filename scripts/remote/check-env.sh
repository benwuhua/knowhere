#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

OUTPUT="text"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --json)
            OUTPUT="json"
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
require_remote_config REMOTE_HOST REMOTE_USER

REMOTE_INFO="$(
    run_remote_script "${REMOTE_VENV_DIR}" "${CONANCENTER_REMOTE_URL}" "${DEFAULT_CONAN_LOCAL_URL}" "${CONANCENTER_REMOTE_USER}" <<'EOF'
set -euo pipefail
venv_dir="$1"
conancenter_remote_url="$2"
default_conan_local_url="$3"
conancenter_remote_user="${4-}"
arch="$(uname -m)"
hostname="$(hostname)"
kernel="$(uname -sr)"
lscpu_out="$(command -v lscpu >/dev/null 2>&1 && lscpu || true)"
cpu_flags="$(printf '%s\n' "${lscpu_out}" | awk -F: '/Flags/ {print $2; exit}')"
if [[ -z "${cpu_flags}" && -r /proc/cpuinfo ]]; then
    cpu_flags="$(grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2-)"
fi
avx2="no"
avx512="no"
if [[ " ${cpu_flags} " == *" avx2 "* ]]; then
    avx2="yes"
fi
if [[ " ${cpu_flags} " == *" avx512f "* ]]; then
    avx512="yes"
fi
gcc_v="$(command -v gcc >/dev/null 2>&1 && gcc --version | head -n1 || echo missing)"
cmake_v="$(command -v cmake >/dev/null 2>&1 && cmake --version | head -n1 || echo missing)"
python_v="$(command -v python3 >/dev/null 2>&1 && python3 --version || echo missing)"
conan_v="$(command -v conan >/dev/null 2>&1 && conan --version || echo missing)"
if [[ "${conan_v}" == "missing" && -x "${venv_dir}/bin/conan" ]]; then
    conan_v="$("${venv_dir}/bin/conan" --version 2>/dev/null || echo missing)"
fi
conan_remotes="missing"
if [[ -x "${venv_dir}/bin/conan" ]]; then
    conan_remotes="$("${venv_dir}/bin/conan" remote list 2>/dev/null | tr '\n' ';' || echo missing)"
fi
printf 'hostname=%s\n' "${hostname}"
printf 'kernel=%s\n' "${kernel}"
printf 'arch=%s\n' "${arch}"
printf 'avx2=%s\n' "${avx2}"
printf 'avx512=%s\n' "${avx512}"
printf 'gcc=%s\n' "${gcc_v}"
printf 'cmake=%s\n' "${cmake_v}"
printf 'python=%s\n' "${python_v}"
printf 'conan=%s\n' "${conan_v}"
printf 'conancenter_remote_expected=%s\n' "${conancenter_remote_url}"
printf 'default_conan_local_expected=%s\n' "${default_conan_local_url}"
printf 'conancenter_remote_user_expected=%s\n' "${conancenter_remote_user:-<unset>}"
printf 'conan_remotes=%s\n' "${conan_remotes}"
EOF
)"

if [[ "${OUTPUT}" == "json" ]]; then
    python3 - "${REMOTE_INFO}" <<'EOF'
import json
import sys

data = {}
for line in sys.argv[1].splitlines():
    if "=" in line:
        key, value = line.split("=", 1)
        data[key] = value
print(json.dumps(data, indent=2, sort_keys=True))
EOF
else
    print_config_summary
    printf '%s\n' "${REMOTE_INFO}"
fi
