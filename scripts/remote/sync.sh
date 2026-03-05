#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

MODE="git"
REF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
ensure_local_command git
ensure_local_command rsync
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR

REF="${REF:-${DEFAULT_BRANCH}}"

case "${MODE}" in
    git)
        run_remote_script "${REMOTE_REPO_DIR}" "${REF}" <<'EOF'
set -euo pipefail
repo_dir="$1"
ref="$2"

if [[ ! -d "${repo_dir}/.git" ]]; then
    echo "remote repo missing: ${repo_dir}" >&2
    exit 1
fi

git -C "${repo_dir}" fetch --all --prune
if git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${ref}"; then
    git -C "${repo_dir}" checkout -B "${ref}" "origin/${ref}"
    git -C "${repo_dir}" reset --hard "origin/${ref}"
else
    git -C "${repo_dir}" checkout --detach "${ref}"
fi

printf 'sync_mode=git\n'
printf 'commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
printf 'branch=%s\n' "$(git -C "${repo_dir}" rev-parse --abbrev-ref HEAD)"
EOF
        ;;
    rsync)
        run_remote_script "${REMOTE_REPO_DIR}" <<'EOF'
set -euo pipefail
repo_dir="$1"
mkdir -p "${repo_dir}"
EOF
        run_rsync \
            --delete \
            --exclude=.git \
            --exclude=build \
            --exclude=.cache \
            "${REPO_ROOT}/" "$(remote_target):${REMOTE_REPO_DIR}/"
        echo "sync_mode=rsync"
        ;;
    *)
        echo "unsupported mode: ${MODE}" >&2
        exit 1
        ;;
esac
