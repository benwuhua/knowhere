#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

DEST="${TMPDIR:-/tmp}/knowhere-remote-logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)
            DEST="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command rsync
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_LOG_DIR

mkdir -p "${DEST}"
run_rsync "$(remote_target):${REMOTE_LOG_DIR}/" "${DEST}/"
printf 'logs=%s\n' "${DEST}"
