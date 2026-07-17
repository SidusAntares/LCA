#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

REMOTE_USER="${REMOTE_USER:-user}"
REMOTE_HOST="${REMOTE_HOST:-10.150.10.38}"
REMOTE_BASE_DIR="${REMOTE_BASE_DIR:-/data/user}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR:-/data/user/LCA}"

if [[ "$PROJECT_NAME" != "LCA" ]]; then
    echo "[ERROR] Expected the local repository directory to be named LCA, got: ${PROJECT_NAME}" >&2
    exit 2
fi

ARCHIVE_PATH="$(mktemp "/tmp/${PROJECT_NAME}_sync_XXXXXX.tar.gz")"
ARCHIVE_NAME="$(basename "$ARCHIVE_PATH")"

cleanup() {
    rm -f "$ARCHIVE_PATH"
}
trap cleanup EXIT

echo "[INFO] Project directory: ${PROJECT_DIR}"
echo "[INFO] Remote target: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
echo "[INFO] Server datasets and generated experiment artifacts will be preserved."

cd "$(dirname "$PROJECT_DIR")"

tar \
    --exclude="${PROJECT_NAME}/.git" \
    --exclude="${PROJECT_NAME}/.idea" \
    --exclude="${PROJECT_NAME}/.pytest_cache" \
    --exclude="${PROJECT_NAME}/**/.pytest_cache" \
    --exclude="${PROJECT_NAME}/.ruff_cache" \
    --exclude="${PROJECT_NAME}/**/.ruff_cache" \
    --exclude="${PROJECT_NAME}/__pycache__" \
    --exclude="${PROJECT_NAME}/**/__pycache__" \
    --exclude="${PROJECT_NAME}/**/*.pyc" \
    --exclude="${PROJECT_NAME}/TSClassif/dataset" \
    --exclude="${PROJECT_NAME}/TSClassif/LCA_*result" \
    --exclude="${PROJECT_NAME}/TSClassif/test_logs" \
    --exclude="${PROJECT_NAME}/outputs" \
    --exclude="${PROJECT_NAME}/runs" \
    --exclude="${PROJECT_NAME}/logs" \
    --exclude="${PROJECT_NAME}/result" \
    --exclude="${PROJECT_NAME}/server_artifacts" \
    --exclude="${PROJECT_NAME}/**/checkpoint.pt" \
    --exclude="${PROJECT_NAME}/**/failed_runs.jsonl" \
    --exclude="${PROJECT_NAME}/**/*.log" \
    --exclude="${PROJECT_NAME}/**/*.out" \
    --exclude="${PROJECT_NAME}/**/*.err" \
    --exclude="${PROJECT_NAME}/**/*.tmp" \
    -czf "$ARCHIVE_PATH" \
    "$PROJECT_NAME"

echo "[INFO] Archive created: ${ARCHIVE_PATH}"

scp "$ARCHIVE_PATH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE_DIR}/"

ssh "${REMOTE_USER}@${REMOTE_HOST}" "
    set -euo pipefail
    mkdir -p '${REMOTE_BASE_DIR}' '${REMOTE_PROJECT_DIR}'
    tar -xzf '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}' -C '${REMOTE_BASE_DIR}'
    rm -f '${REMOTE_BASE_DIR}/${ARCHIVE_NAME}'
    test -f '${REMOTE_PROJECT_DIR}/TSClassif/tools/check_environment.py'
    test -f '${REMOTE_PROJECT_DIR}/TSClassif/tools/check_har_dataset.py'
    test -f '${REMOTE_PROJECT_DIR}/TSClassif/tools/smoke_test_lca.py'
"

echo "[SUCCESS] Code synced to ${REMOTE_PROJECT_DIR}"
echo "[NEXT] ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "[NEXT] cd ${REMOTE_PROJECT_DIR}/TSClassif"

