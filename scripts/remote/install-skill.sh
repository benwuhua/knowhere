#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="${REPO_ROOT}/tools/skills/knowhere-x86-remote"
INSTALL_CODEX="true"
INSTALL_CLAUDE="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --codex-only)
            INSTALL_CLAUDE="false"
            shift
            ;;
        --claude-only)
            INSTALL_CODEX="false"
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

install_skill() {
    local target_root="$1"
    local target_dir="${target_root}/knowhere-x86-remote"
    mkdir -p "${target_root}"
    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"
    cp -R "${SOURCE_DIR}/." "${target_dir}/"
}

if [[ "${INSTALL_CODEX}" == "true" ]]; then
    install_skill "${HOME}/.codex/skills"
fi

if [[ "${INSTALL_CLAUDE}" == "true" ]]; then
    install_skill "${HOME}/.claude/skills"
fi

echo "installed knowhere-x86-remote skill"
