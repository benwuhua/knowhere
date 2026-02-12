#!/bin/bash
# Copyright 2025 The Knowhere Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to download and prepare SIFT1M dataset for JAG benchmark

set -e

DATA_DIR="${1:-./data}"
SIFT_DIR="$DATA_DIR/sift"

echo "========================================"
echo "SIFT1M Data Preparation for JAG Benchmark"
echo "========================================"
echo "Target directory: $DATA_DIR"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# Download SIFT1M if not exists
if [ ! -f "$SIFT_DIR/sift_base.fvecs" ]; then
    echo "Downloading SIFT1M dataset..."
    cd "$DATA_DIR"
    wget -q --show-progress ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz || {
        echo "Failed to download from FTP. Trying HTTP mirror..."
        wget -q --show-progress http://corpus-texmex.irisa.fr/imagenet/sift.tar.gz || {
            echo "ERROR: Failed to download SIFT1M dataset"
            echo "Please download manually from:"
            echo "  http://corpus-texmex.irisa.fr/"
            exit 1
        }
    }
    echo "Extracting..."
    tar -xzf sift.tar.gz
    rm -f sift.tar.gz
    cd - > /dev/null
else
    echo "SIFT1M data already exists at $SIFT_DIR"
fi

# Convert .fvecs to .fbin format
echo ""
echo "Converting to .fbin format..."

python3 - << 'PYTHON_SCRIPT'
import struct
import os
import sys

def convert_fvecs_to_fbin(input_path, output_path):
    """Convert .fvecs format to .fbin format used by JAG benchmark."""
    print(f"Converting {input_path} to {output_path}")

    with open(input_path, 'rb') as fin:
        # Read first vector to get dimension
        dim_bytes = fin.read(4)
        if not dim_bytes:
            print(f"ERROR: Empty file {input_path}")
            return False
        dim = struct.unpack('<i', dim_bytes)[0]
        fin.seek(0, 2)  # Seek to end
        file_size = fin.tell()
        fin.seek(0)

        # Calculate number of vectors
        bytes_per_vec = 4 + dim * 4  # dim field + float values
        n = file_size // bytes_per_vec

        print(f"  Dimension: {dim}, Vectors: {n}")

        # Read all vectors
        data = []
        for i in range(n):
            d = struct.unpack('<i', fin.read(4))[0]
            assert d == dim, f"Dimension mismatch at vector {i}: {d} != {dim}"
            vec = struct.unpack(f'<{dim}f', fin.read(dim * 4))
            data.extend(vec)

        # Write .fbin format
        with open(output_path, 'wb') as fout:
            fout.write(struct.pack('<i', n))      # number of vectors
            fout.write(struct.pack('<i', dim))    # dimension
            fout.write(struct.pack(f'<{n * dim}f', *data))

        print(f"  Written {output_path}")
        return True

data_dir = os.environ.get('DATA_DIR', './data')
sift_dir = os.path.join(data_dir, 'sift')

files = [
    ('sift_base.fvecs', 'sift1m-base.fbin'),
    ('sift_query.fvecs', 'sift1m-query.fbin'),
    ('sift_groundtruth.ivecs', 'sift1m-groundtruth.ibin'),
]

success = True
for input_name, output_name in files:
    input_path = os.path.join(sift_dir, input_name)
    output_path = os.path.join(data_dir, output_name)

    if not os.path.exists(input_path):
        print(f"WARNING: {input_path} not found, skipping")
        continue

    if not convert_fvecs_to_fbin(input_path, output_path):
        success = False

sys.exit(0 if success else 1)
PYTHON_SCRIPT

export DATA_DIR="$DATA_DIR"

# Generate label filter files
echo ""
echo "Generating label filter files..."

python3 - << 'PYTHON_SCRIPT'
import struct
import random
import os

data_dir = os.environ.get('DATA_DIR', './data')
num_labels = 10
seed = 42

# Read base data to get count
base_path = os.path.join(data_dir, 'sift1m-base.fbin')
with open(base_path, 'rb') as f:
    n = struct.unpack('<i', f.read(4))[0]
    dim = struct.unpack('<i', f.read(4))[0]
    print(f"Generating labels for {n} vectors, {num_labels} categories")

# Generate random labels
random.seed(seed)
labels = [random.randint(0, num_labels - 1) for _ in range(n)]

# Write base labels
output_path = os.path.join(data_dir, 'sift1m-base-filters-label.ibin')
with open(output_path, 'wb') as f:
    f.write(struct.pack('<i', n))
    f.write(struct.pack(f'<{n}i', *labels))
print(f"  Written {output_path}")

# Generate query labels (100 queries)
nq = 100
query_labels = [random.randint(0, num_labels - 1) for _ in range(nq)]
output_path = os.path.join(data_dir, 'sift1m-query-filters-label.ibin')
with open(output_path, 'wb') as f:
    f.write(struct.pack('<i', nq))
    f.write(struct.pack(f'<{nq}i', *query_labels))
print(f"  Written {output_path}")

# Print label distribution
from collections import Counter
dist = Counter(labels)
print("\nLabel distribution:")
for label in sorted(dist.keys()):
    pct = dist[label] / n * 100
    print(f"  Label {label}: {dist[label]} ({pct:.1f}%)")
PYTHON_SCRIPT

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "========================================"
echo "Files created in $DATA_DIR:"
ls -la "$DATA_DIR"/*.fbin "$DATA_DIR"/*.ibin 2>/dev/null || echo "  (No .fbin/.ibin files found)"
echo ""
echo "To run the benchmark:"
echo "  export SIFT1M_PATH=$DATA_DIR/sift1m"
echo "  ./knowhere_tests '[jag][benchmark][sift1m]'"
echo "========================================"
