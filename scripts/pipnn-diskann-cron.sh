#!/bin/bash
# PipNN-DiskANN Cron Job - Build and Verify
# This script is executed by macOS launchd daily

set -euo

# Configuration
REPO_DIR="/Users/ryan/Code/knowhere"
BRANCH="feat/pipnn-diskann"
CONTAINER_NAME="knowhere-x86-builder"
BUILD_TYPE="Debug"
LOG_DIR="/Users/ryan/Code/knowhere/logs/pipnn-cron"
NOTIFY_SCRIPT="/Users/ryan/Code/knowhere/scripts/notify.sh"

# Create log directory
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H%M%S")
LOG_FILE="$LOG_DIR/build_$TIMESTAMP.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error_notify() {
    local msg="$1"
    log "ERROR: $msg"
    if [ -f "$NOTIFY_SCRIPT" ]; then
        "$NOTIFY_SCRIPT" "pipnn-diskann cron FAILED" "$msg"
    fi
}

success_notify() {
    local msg="$1"
    log "SUCCESS: $msg"
    if [ -f "$NOTIFY_SCRIPT" ]; then
        "$NOTIFY_SCRIPT" "pipnn-diskann cron OK" "$msg"
    fi
}

cleanup() {
    # Remove any stale containers
    docker ps -aq -f name="$CONTAINER_NAME" | xargs -r docker rm -f || true
}
trap cleanup EXIT

log "=========================================="
log "PiPNN-DiskANN Cron Job Started"
log "=========================================="
log "Repo: $REPO_DIR"
log "Branch: $BRANCH"
log "Container: $CONTAINER_NAME"
log "Build Type: $BUILD_TYPE"
log "=========================================="

cd "$REPO_DIR"

# Step 1: Pull latest changes
log "Step 1: Pulling latest changes..."
if ! git -C "$REPO_DIR" fetch origin "$BRANCH" 2>&1 | tee -a "$LOG_FILE"; then
    error_notify "Failed to fetch from origin/$BRANCH"
    exit 1
fi

CURRENT_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
FETCH_COMMIT=$(git -C "$REPO_DIR" rev-parse origin/"$BRANCH")

log "Current commit: $CURRENT_COMMIT"
log "Latest remote commit: $FETCH_COMMIT"

if [ "$CURRENT_COMMIT" != "$FETCH_COMMIT" ]; then
    log "New commit detected, proceeding with build..."
else
    log "Already at latest commit, skipping build."
    success_notify "Already up to date"
    exit 0
fi

# Step 2: Build with x86 container
log "Step 2: Building with x86 container..."
BUILD_START=$(date +%s)

docker run --platform linux/amd64 --rm \
    -v "$REPO_DIR:/workspace \
    -v "$HOME/.cache/conan:/root/.conan" \
    -w /workspace/build \
    "$CONTAINER_NAME" \
    bash -c "
        set -ex
        rm -rf build/Debug

        echo '=== Running conan install ===' >&2
        conan install .. \
          --build=missing \
          -o with_diskann=True \
          -o with_ut=True \
          -s compiler.libcxx=libstdc++11 \
          -s build_type=Debug \
          2>&1 | tee -a /workspace/build/conan_install.log

        echo '' >&2
        echo '=== Running conan build ===' >&2
        conan build .. 2>&1 | tee -a /workspace/build/conan_build.log

        if [ \$? -ne 0 ]; then
            echo 'BUILD FAILED' >&2
            exit \$?
        fi

        echo '=== Build Complete ===' >&2
        echo 'libknowhere.so size:' \$(du -h lib/Debug/libknowhere.so)
    " 2>&1 | tee -a "$LOG_FILE"

BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

if [ $? -ne 0 ]; then
    error_notify "Docker build failed (exit code: $?)"
    exit 1
fi

log "Build completed in ${BUILD_DURATION}s"

# Step 3: Run unit tests
log "Step 3: Running PiPNN-DiskANN tests..."

TEST_START=$(date +%s)

docker run --platform linux/amd64 --rm \
    -v "$REPO_DIR:/workspace \
    -v "$HOME/.cache/conan:/root/.conan" \
    -w /workspace/build/Debug \
    "$CONTAINER_NAME" \
    ./tests/ut/knowhere_tests '[pipnn]' \
    2>&1 | tee -a "$LOG_FILE"

TEST_END=$(date +%s)
TEST_DURATION=$((TEST_END - TEST_START))
TEST_EXIT=$?

log "Tests completed in ${TEST_DURATION}s, exit code: $TEST_EXIT"

# Step 4: Code quality checks
log "Step 4: Running code quality checks..."

cd "$REPO_DIR/src/index/diskann/impl"

# Check clang-format
if command -v clang-format &>/dev/null; then
    log "Checking clang-format..."
    for file in pipnn_diskann.cc pipnn_diskann_config.h; do
        if [ -f "$file" ]; then
            FORMATTED=$(clang-format --dry-run --Werror "$file" 2>/dev/null)
            if [ -n "$FORMATTED" ]; then
                log "  ERROR: $file not formatted properly"
                ERROR_FILES+=("$file")
            else
                log "  OK: $file"
            fi
        fi
    done
else
    log "clang-format not found, skipping format check"
fi

# Summary
log "=========================================="
log "Cron Job Summary"
log "=========================================="
log "Build duration: ${BUILD_DURATION}s"
log "Test duration: ${TEST_DURATION}s"
log "Test exit code: $TEST_EXIT"

if [ "$TEST_EXIT" -eq 0 ] && [ ${#ERROR_FILES[@]} -eq 0 ]; then
    success_notify "Build: OK, Tests: PASSED, Format: OK"
    log "Status: ALL CHECKS PASSED"
    exit 0
else
    ERROR_MSG="Build/Tests FAILED"
    if [ "$TEST_EXIT" -ne 0 ]; then
        ERROR_MSG="$ERROR_MSG (exit code: $TEST_EXIT)"
    fi
    if [ ${#ERROR_FILES[@]} -gt 0 ]; then
        ERROR_MSG="$ERROR_MSG (format issues: ${ERROR_FILES[*]})"
    fi
    error_notify "$ERROR_MSG"
    log "Status: FAILED"
    exit 1
fi
