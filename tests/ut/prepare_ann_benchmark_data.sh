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

# Script to download and prepare ANN benchmark datasets for JAG benchmark
# Supports: glove-100, deep-1m, gist, sift1m

set -e

usage() {
    echo "Usage: $0 <dataset> <data_dir> [options]"
    echo ""
    echo "Datasets:"
    echo "  glove-100  GloVe-100 (1.2M vectors, 100D, Angular/IP)"
    echo "  deep-1m    Deep-1M (1M vectors, 96D, L2)"
    echo "  gist       GIST (1M vectors, 960D, L2)"
    echo "  sift1m     SIFT1M (1M vectors, 128D, L2)"
    echo "  all        Download all datasets"
    echo ""
    echo "Options:"
    echo "  --num-labels=N    Number of label categories (default: 10)"
    echo "  --zipf-alpha=A    Zipf distribution alpha parameter (default: 1.5)"
    echo "  --skip-download   Skip download if HDF5 file exists"
    echo ""
    echo "Examples:"
    echo "  $0 glove-100 ./data"
    echo "  $0 deep-1m ./data --num-labels=20"
    echo "  $0 all ./data"
    exit 1
}

# Parse arguments
if [ $# -lt 2 ]; then
    usage
fi

DATASET="$1"
DATA_DIR="$2"
shift 2

# Default options
NUM_LABELS=10
ZIPF_ALPHA=1.5
SKIP_DOWNLOAD=false

# Parse options
for arg in "$@"; do
    case $arg in
        --num-labels=*)
            NUM_LABELS="${arg#*=}"
            shift
            ;;
        --zipf-alpha=*)
            ZIPF_ALPHA="${arg#*=}"
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            usage
            ;;
    esac
done

# Dataset configurations
declare -A DATASET_NAMES
DATASET_NAMES["glove-100"]="GloVe-100"
DATASET_NAMES["deep-1m"]="Deep-1M"
DATASET_NAMES["gist"]="GIST"
DATASET_NAMES["sift1m"]="SIFT1M"

declare -A HDF5_URLS
HDF5_URLS["glove-100"]="http://ann-benchmarks.com/glove-100-angular.hdf5"
HDF5_URLS["deep-1m"]="http://ann-benchmarks.com/deep-image-96-angular.hdf5"
HDF5_URLS["gist"]="http://ann-benchmarks.com/gist-960-euclidean.hdf5"
HDF5_URLS["sift1m"]="http://ann-benchmarks.com/sift-128-euclidean.hdf5"

declare -A METRIC_TYPES
METRIC_TYPES["glove-100"]="IP"     # Angular -> IP after normalization
METRIC_TYPES["deep-1m"]="L2"
METRIC_TYPES["gist"]="L2"
METRIC_TYPES["sift1m"]="L2"

# Function to process a single dataset
process_dataset() {
    local ds="$1"
    local display_name="${DATASET_NAMES[$ds]}"
    local hdf5_url="${HDF5_URLS[$ds]}"
    local metric_type="${METRIC_TYPES[$ds]}"

    echo ""
    echo "========================================"
    echo "$display_name Data Preparation"
    echo "========================================"
    echo "Dataset: $ds"
    echo "Metric: $metric_type"
    echo "Target: $DATA_DIR"
    echo "Labels: $NUM_LABELS categories"
    echo ""

    # Create directories
    mkdir -p "$DATA_DIR"
    local hdf5_path="$DATA_DIR/${ds}.hdf5"

    # Download HDF5 file
    if [ "$SKIP_DOWNLOAD" = true ] && [ -f "$hdf5_path" ]; then
        echo "HDF5 file exists, skipping download: $hdf5_path"
    else
        echo "Downloading $display_name from ann-benchmarks.com..."
        wget -q --show-progress -O "$hdf5_path" "$hdf5_url" || {
            echo "ERROR: Failed to download from $hdf5_url"
            echo "Please check your internet connection or download manually"
            return 1
        }
    fi

    # Convert HDF5 to FBIN/IBIN format
    echo ""
    echo "Converting HDF5 to FBIN/IBIN format..."

    export DATA_DIR
    export NUM_LABELS
    export ZIPF_ALPHA
    export DATASET="$ds"
    export METRIC_TYPE="$metric_type"

    python3 - << 'PYTHON_SCRIPT'
import h5py
import struct
import numpy as np
import os
import sys

def hdf5_to_fbin(hdf5_path, output_dir, dataset_name, metric_type):
    """Convert HDF5 to FBIN format used by JAG benchmark."""
    print(f"Reading {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # List available datasets
        print(f"  Available datasets: {list(f.keys())}")

        # Read datasets (ann-benchmarks standard format)
        train = np.array(f['train'])
        test = np.array(f['test'])
        neighbors = np.array(f['neighbors'])
        distances = np.array(f['distances']) if 'distances' in f else None

        n, dim = train.shape
        nq = test.shape[0]
        k = neighbors.shape[1] if len(neighbors.shape) > 1 else 100

        print(f"  Base vectors: {n}, Dimension: {dim}")
        print(f"  Query vectors: {nq}, Neighbors per query: {k}")

        # For Angular (IP) datasets, normalize vectors
        if metric_type == "IP":
            print("  Normalizing vectors for Angular/IP metric...")
            train = train / np.linalg.norm(train, axis=1, keepdims=True)
            test = test / np.linalg.norm(test, axis=1, keepdims=True)
            # Replace NaN with 0 (for zero vectors)
            train = np.nan_to_num(train)
            test = np.nan_to_num(test)

        # Write base.fbin
        base_path = os.path.join(output_dir, f'{dataset_name}-base.fbin')
        with open(base_path, 'wb') as fout:
            fout.write(struct.pack('<i', n))
            fout.write(struct.pack('<i', dim))
            fout.write(train.astype(np.float32).tobytes())
        print(f"  Written {base_path}")

        # Write query.fbin
        query_path = os.path.join(output_dir, f'{dataset_name}-query.fbin')
        with open(query_path, 'wb') as fout:
            fout.write(struct.pack('<i', nq))
            fout.write(struct.pack('<i', dim))
            fout.write(test.astype(np.float32).tobytes())
        print(f"  Written {query_path}")

        # Write groundtruth.ibin (flat list of neighbor IDs)
        gt_path = os.path.join(output_dir, f'{dataset_name}-groundtruth.ibin')
        with open(gt_path, 'wb') as fout:
            total_neighbors = nq * k
            fout.write(struct.pack('<i', total_neighbors))
            fout.write(neighbors.astype(np.int32).flatten().tobytes())
        print(f"  Written {gt_path}")

        return n, dim, nq

def generate_labels(output_dir, dataset_name, n, nq, num_labels, zipf_alpha):
    """Generate both uniform and Zipf-distributed labels."""
    np.random.seed(42)

    # Uniform distribution
    print(f"\nGenerating uniform labels ({num_labels} categories)...")
    uniform_labels = np.random.randint(0, num_labels, n)
    uniform_path = os.path.join(output_dir, f'{dataset_name}-base-filters-uniform.ibin')
    with open(uniform_path, 'wb') as f:
        f.write(struct.pack('<i', n))
        f.write(uniform_labels.astype(np.int32).tobytes())
    print(f"  Written {uniform_path}")

    # Print uniform distribution
    unique, counts = np.unique(uniform_labels, return_counts=True)
    print("  Distribution:")
    for label, count in zip(unique, counts):
        print(f"    Label {label}: {count} ({count/n*100:.1f}%)")

    # Zipf distribution (for testing varied filter ratios)
    print(f"\nGenerating Zipf labels (alpha={zipf_alpha})...")
    zipf_labels = np.random.zipf(zipf_alpha, n) % num_labels
    zipf_path = os.path.join(output_dir, f'{dataset_name}-base-filters-zipf.ibin')
    with open(zipf_path, 'wb') as f:
        f.write(struct.pack('<i', n))
        f.write(zipf_labels.astype(np.int32).tobytes())
    print(f"  Written {zipf_path}")

    # Print Zipf distribution
    unique, counts = np.unique(zipf_labels, return_counts=True)
    print("  Distribution:")
    for label, count in zip(unique, counts):
        print(f"    Label {label}: {count} ({count/n*100:.1f}%)")

    # Generate query labels (uniform)
    print(f"\nGenerating query labels...")
    query_labels = np.random.randint(0, num_labels, nq)
    query_path = os.path.join(output_dir, f'{dataset_name}-query-filters-label.ibin')
    with open(query_path, 'wb') as f:
        f.write(struct.pack('<i', nq))
        f.write(query_labels.astype(np.int32).tobytes())
    print(f"  Written {query_path}")

# Get parameters from environment
data_dir = os.environ.get('DATA_DIR', './data')
dataset = os.environ.get('DATASET', 'glove-100')
num_labels = int(os.environ.get('NUM_LABELS', '10'))
zipf_alpha = float(os.environ.get('ZIPF_ALPHA', '1.5'))
metric_type = os.environ.get('METRIC_TYPE', 'L2')

hdf5_path = os.path.join(data_dir, f'{dataset}.hdf5')

if not os.path.exists(hdf5_path):
    print(f"ERROR: HDF5 file not found: {hdf5_path}")
    sys.exit(1)

# Convert and generate labels
n, dim, nq = hdf5_to_fbin(hdf5_path, data_dir, dataset, metric_type)
generate_labels(data_dir, dataset, n, nq, num_labels, zipf_alpha)

print("\nConversion complete!")
PYTHON_SCRIPT

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to convert $display_name"
        return 1
    fi

    echo ""
    echo "$display_name preparation complete!"
    echo "Files created:"
    ls -la "$DATA_DIR"/${ds}-*.fbin "$DATA_DIR"/${ds}-*.ibin 2>/dev/null || echo "  (No files found)"

    return 0
}

# Process datasets
mkdir -p "$DATA_DIR"

if [ "$DATASET" = "all" ]; then
    echo "Processing all datasets..."
    for ds in glove-100 deep-1m gist sift1m; do
        process_dataset "$ds" || echo "Warning: $ds failed, continuing..."
    done
else
    if [ -z "${DATASET_NAMES[$DATASET]}" ]; then
        echo "ERROR: Unknown dataset: $DATASET"
        echo "Valid datasets: glove-100, deep-1m, gist, sift1m, all"
        exit 1
    fi
    process_dataset "$DATASET"
fi

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "========================================"
echo ""
echo "To run JAG benchmarks:"
echo "  export ANN_BENCHMARK_PATH=$DATA_DIR"
echo "  ./build/Release/tests/ut/knowhere_tests '[jag][benchmark]'"
echo ""
echo "Dataset-specific environment variables:"
echo "  export GLOVE100_PATH=$DATA_DIR     # For GloVe-100"
echo "  export DEEP1M_PATH=$DATA_DIR       # For Deep-1M"
echo "  export GIST_PATH=$DATA_DIR         # For GIST"
echo "  export SIFT1M_PATH=$DATA_DIR       # For SIFT1M"
echo "========================================"
