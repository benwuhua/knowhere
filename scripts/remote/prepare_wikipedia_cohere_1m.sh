#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=scripts/remote/common.sh
source "${SCRIPT_DIR}/common.sh"

MODE="probe"
DEST_DIR="/data/work/datasets/wikipedia-cohere-1m"
BASE_ROWS="1000000"
BASE_DIM="768"

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

run_remote_script "${MODE}" "${DEST_DIR}" "${BASE_ROWS}" "${BASE_DIM}" <<'EOF'
set -euo pipefail

mode="$1"
dest_dir="$2"
base_rows="$3"
base_dim="$4"
base_url="https://comp21storage.z5.web.core.windows.net/wiki-cohere-35M"
base_src="${base_url}/wikipedia_base.bin"
query_src="${base_url}/wikipedia_query.bin"
gt_src="${base_url}/wikipedia-1M"
base_dst="${dest_dir}/base.fbin"
query_dst="${dest_dir}/query.fbin"
gt_dst="${dest_dir}/gt.ibin"
base_crop_bytes=$((8 + base_rows * base_dim * 4))

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
    print(f"accept_ranges={resp.headers.get('Accept-Ranges', '')}")
    print(f"etag={resp.headers.get('ETag', '')}")
PY
}

if [[ "${mode}" == "probe" ]]; then
    echo "dataset_label=wikipedia-cohere-1m-ip"
    echo "dest_dir=${dest_dir}"
    echo "base_path=${base_dst}"
    echo "query_path=${query_dst}"
    echo "gt_path=${gt_dst}"
    echo "base_rows=${base_rows}"
    echo "base_dim=${base_dim}"
    echo "base_crop_bytes=${base_crop_bytes}"
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

python3 - "${base_src}" "${base_dst}" "${query_src}" "${query_dst}" "${gt_src}" "${gt_dst}" "${base_crop_bytes}" "${base_rows}" "${base_dim}" <<'PY'
import math, os, struct, sys, time, urllib.request

base_src, base_dst, query_src, query_dst, gt_src, gt_dst, base_crop_bytes, base_rows, base_dim = sys.argv[1:10]
base_crop_bytes = int(base_crop_bytes)
base_rows = int(base_rows)
base_dim = int(base_dim)
CHUNK = 8 * 1024 * 1024
HEARTBEAT_SECS = 15


def format_bytes(num):
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.1f}{unit}'
        value /= 1024.0


def stream_download(url, dst, expected_size=None, patch_header=None):
    tmp = dst + '.tmp'
    final_exists = os.path.exists(dst)
    final_size = os.path.getsize(dst) if final_exists else -1
    if expected_size is not None and final_exists and final_size == expected_size:
        print(f'skip_existing={dst} size={final_size}')
        return
    if expected_size is None and final_exists and final_size > 8:
        print(f'skip_existing={dst} size={final_size}')
        return

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    resume_from = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if expected_size is not None and resume_from > expected_size:
        print(f'truncate_oversized_tmp={tmp} from={resume_from} to={expected_size}')
        with open(tmp, 'r+b') as f:
            f.truncate(expected_size)
        resume_from = expected_size

    mode = 'ab' if resume_from > 0 else 'wb'
    headers = {}
    if expected_size is not None:
        if resume_from >= expected_size:
            print(f'skip_download_complete_tmp={tmp} size={resume_from}')
        else:
            headers['Range'] = f'bytes={resume_from}-{expected_size - 1}'
            if resume_from > 0:
                print(f'resume={dst} from={resume_from} to={expected_size - 1} expected={expected_size}')
            else:
                print(f'download={url} -> {dst} range=0-{expected_size - 1} expected={expected_size}')
    elif resume_from > 0:
        headers['Range'] = f'bytes={resume_from}-'
        print(f'resume={dst} from={resume_from} expected=unknown')
    else:
        print(f'download={url} -> {dst} expected=unknown')

    written = resume_from
    if expected_size is None or resume_from < expected_size:
        req = urllib.request.Request(url, headers=headers)
        start = time.time()
        last_log = start
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, mode) as out:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                if expected_size is not None:
                    remaining = expected_size - written
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk[:remaining]
                out.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_log >= HEARTBEAT_SECS:
                    elapsed = max(now - start, 1e-6)
                    rate = max(written - resume_from, 0) / elapsed
                    if expected_size is not None and expected_size > 0:
                        pct = 100.0 * written / expected_size
                        print(f'progress={dst} bytes={written}/{expected_size} pct={pct:.2f} rate={format_bytes(rate)}/s')
                    else:
                        print(f'progress={dst} bytes={written} rate={format_bytes(rate)}/s')
                    last_log = now

    size = os.path.getsize(tmp)
    if expected_size is not None and size > expected_size:
        print(f'truncate_post_download={tmp} from={size} to={expected_size}')
        with open(tmp, 'r+b') as f:
            f.truncate(expected_size)
        size = expected_size
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f'unexpected size for {dst}: got {size}, expected {expected_size}')
    if patch_header is not None:
        rows, dim = patch_header
        with open(tmp, 'r+b') as f:
            f.seek(0)
            f.write(struct.pack('<I', rows))
            f.write(struct.pack('<I', dim))
    os.replace(tmp, dst)
    print(f'done={dst} size={os.path.getsize(dst)}')


stream_download(base_src, base_dst, expected_size=base_crop_bytes, patch_header=(base_rows, base_dim))
stream_download(query_src, query_dst)
stream_download(gt_src, gt_dst)
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
