#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

REPO_URL=""
BRANCH=""
SKIP_CLONE="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-url)
            REPO_URL="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --skip-clone)
            SKIP_CLONE="true"
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
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_WORK_ROOT REMOTE_REPO_DIR REMOTE_BUILD_DIR REMOTE_LOG_DIR REMOTE_CCACHE_DIR REMOTE_VENV_DIR CONANCENTER_REMOTE_URL DEFAULT_CONAN_LOCAL_URL

REPO_URL="${REPO_URL:-${REMOTE_REPO_URL}}"
BRANCH="${BRANCH:-${DEFAULT_BRANCH}}"

run_remote_script "${REPO_URL}" "${BRANCH}" "${SKIP_CLONE}" "${REMOTE_WORK_ROOT}" "${REMOTE_REPO_DIR}" "${REMOTE_BUILD_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CCACHE_DIR}" "${REMOTE_VENV_DIR}" "${CONANCENTER_REMOTE_URL}" "${DEFAULT_CONAN_LOCAL_URL}" "${CONANCENTER_REMOTE_USER}" "${CONANCENTER_REMOTE_PASSWORD}" <<'EOF'
set -euo pipefail

repo_url="$1"
branch="$2"
skip_clone="$3"
work_root="$4"
repo_dir="$5"
build_dir="$6"
log_dir="$7"
ccache_dir="$8"
venv_dir="$9"
conancenter_remote_url="${10}"
default_conan_local_url="${11}"
conancenter_remote_user="${12-}"
conancenter_remote_password="${13-}"

if [[ "$(id -u)" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

export DEBIAN_FRONTEND=noninteractive

${SUDO} apt-get update
${SUDO} apt-get install -y \
    build-essential \
    ccache \
    clang-format \
    clang-tidy \
    cmake \
    gdb \
    git \
    libaio-dev \
    libboost-program-options-dev \
    libopenblas-openmp-dev \
    ninja-build \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    rsync

mkdir -p "${work_root}" "${repo_dir}" "${build_dir}" "${log_dir}" "${ccache_dir}"

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/pip" install --upgrade pip setuptools wheel
"${venv_dir}/bin/pip" install "conan==1.61.0"

"${venv_dir}/bin/python" - <<'PY'
import pathlib
import sysconfig

site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
shim = site_packages / "imp.py"
content = """from importlib import machinery, reload as reload, util
from types import ModuleType

PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
PKG_DIRECTORY = 5
C_BUILTIN = 6
PY_FROZEN = 7

def new_module(name):
    return ModuleType(name)

def _load_with_loader(loader, name):
    spec = util.spec_from_loader(name, loader)
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module

def load_source(name, pathname, file=None):
    loader = machinery.SourceFileLoader(name, pathname)
    return _load_with_loader(loader, name)

def load_compiled(name, pathname, file=None):
    loader = machinery.SourcelessFileLoader(name, pathname)
    return _load_with_loader(loader, name)

def find_module(name, path=None):
    spec = machinery.PathFinder.find_spec(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(name)
    filename = None
    if hasattr(spec.loader, "get_filename"):
        filename = spec.loader.get_filename(name)
    if filename and filename.endswith(tuple(machinery.BYTECODE_SUFFIXES)):
        return open(filename, "rb"), filename, ("", "rb", PY_COMPILED)
    if filename and filename.endswith(tuple(machinery.SOURCE_SUFFIXES)):
        return open(filename, "r"), filename, ("", "r", PY_SOURCE)
    return None, filename, ("", "", PKG_DIRECTORY)

def acquire_lock():
    return None

def release_lock():
    return None

def lock_held():
    return False
"""
shim.write_text(content)
PY

"${venv_dir}/bin/conan" remote add conancenter "${conancenter_remote_url}" --force >/dev/null 2>&1 || true
"${venv_dir}/bin/conan" remote update conancenter "${conancenter_remote_url}" >/dev/null 2>&1 || true
"${venv_dir}/bin/conan" remote add default-conan-local "${default_conan_local_url}" --force >/dev/null 2>&1 || true
"${venv_dir}/bin/conan" remote update default-conan-local "${default_conan_local_url}" >/dev/null 2>&1 || true
if [[ -n "${conancenter_remote_user}" && -n "${conancenter_remote_password}" ]]; then
    "${venv_dir}/bin/conan" user "${conancenter_remote_user}" -p "${conancenter_remote_password}" -r conancenter >/dev/null
fi

if [[ "${skip_clone}" != "true" && -n "${repo_url}" && ! -d "${repo_dir}/.git" ]]; then
    rm -rf "${repo_dir}"
    git clone "${repo_url}" "${repo_dir}"
fi

if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" fetch origin "${branch}" || true
fi

cat <<OUT
bootstrap=ok
repo_dir=${repo_dir}
build_dir=${build_dir}
log_dir=${log_dir}
venv_dir=${venv_dir}
conancenter_remote_url=${conancenter_remote_url}
default_conan_local_url=${default_conan_local_url}
conancenter_remote_user=${conancenter_remote_user:-<unset>}
OUT
EOF
