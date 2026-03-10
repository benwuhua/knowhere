#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

MODE="probe"
DEST_DIR="/data/work/datasets/openai-arxiv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dest)
            DEST_DIR="$2"
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
require_remote_config REMOTE_HOST REMOTE_USER

run_remote_script "${MODE}" "${DEST_DIR}" <<'EOF'
set -euo pipefail

mode="$1"
dest_dir="$2"
base_url="https://comp21storage.z5.web.core.windows.net/arxiv-openaiv2-2M"
base_src="${base_url}/openai_base.bin"
query_src="${base_url}/openai_query.bin"
gt_src="${base_url}/openai-2M"

mkdir -p "${dest_dir}"

probe_file() {
    local src="$1"
    python3 - "$src" <<'PY'
import sys, urllib.request
url = sys.argv[1]
req = urllib.request.Request(url, method='HEAD')
with urllib.request.urlopen(req, timeout=30) as resp:
    print(f"url={url}")
    print(f"status={getattr(resp, 'status', 'unknown')}")
    print(f"length={resp.headers.get('Content-Length', '')}")
    print(f"etag={resp.headers.get('ETag', '')}")
PY
}

if [[ "${mode}" == "probe" ]]; then
    echo "dest_dir=${dest_dir}"
    echo "base_path=${dest_dir}/base.fbin"
    echo "query_path=${dest_dir}/query.fbin"
    echo "gt_path=${dest_dir}/gt.ibin"
    echo "--- remote_existing ---"
    ls -lah "${dest_dir}" || true
    echo "--- remote_source_probe:base ---"
    probe_file "${base_src}"
    echo "--- remote_source_probe:query ---"
    probe_file "${query_src}"
    echo "--- remote_source_probe:gt ---"
    probe_file "${gt_src}"
    exit 0
fi

if [[ "${mode}" != "download" ]]; then
    echo "unsupported mode: ${mode}" >&2
    exit 1
fi

python3 - "${dest_dir}" "${base_src}" "${query_src}" "${gt_src}" <<'PY'
import os, sys, urllib.request, shutil

dest_dir, base_src, query_src, gt_src = sys.argv[1:5]
os.makedirs(dest_dir, exist_ok=True)

def download(url, dst):
    tmp = dst + '.tmp'
    if os.path.exists(dst) and os.path.getsize(dst) > 8:
        print(f'skip_existing={dst}')
        return
    print(f'download={url} -> {dst}')
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, 'wb') as out:
        shutil.copyfileobj(resp, out, length=1024 * 1024)
    os.replace(tmp, dst)
    print(f'done={dst} size={os.path.getsize(dst)}')

# big-ann competition files already use the same [rows, dim, payload] header layout
# expected by knowhere's external recall loader, so we only normalize names.
download(base_src, os.path.join(dest_dir, 'base.fbin'))
download(query_src, os.path.join(dest_dir, 'query.fbin'))
download(gt_src, os.path.join(dest_dir, 'gt.ibin'))
PY

python3 - "${dest_dir}" <<'PY'
import os, struct, sys
root = sys.argv[1]
for name in ('base.fbin', 'query.fbin', 'gt.ibin'):
    path = os.path.join(root, name)
    with open(path, 'rb') as f:
        rows = struct.unpack('<I', f.read(4))[0]
        cols = struct.unpack('<I', f.read(4))[0]
    print(f'header:{name}:rows={rows}:cols={cols}:bytes={os.path.getsize(path)}')
PY
EOF
