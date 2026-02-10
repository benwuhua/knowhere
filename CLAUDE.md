# CLAUDE.md - Knowhere Development Guide

## Project Overview

Knowhere is a high-performance C++ vector similarity search library that serves as the internal core of [Milvus](https://github.com/milvus-io/milvus). It provides approximate nearest neighbor (ANN) search with support for multiple index types, distance metrics, and hardware accelerators (CPU SIMD, GPU via CUVS).

- **Language**: C++17
- **License**: Apache 2.0
- **Build System**: CMake 3.26.4+ with Conan 1.61.0+ for dependency management

## Repository Structure

```
knowhere/
├── include/knowhere/       # Public C++ headers (API surface)
│   ├── index_node.h        # IndexNode base class (Build, Train, Search, Serialize)
│   ├── index.h             # Template Index<T> wrapper
│   ├── index_factory.h     # Factory for creating index instances
│   ├── config.h            # Configuration system with JSON validation
│   ├── dataset.h           # Data container for training/search
│   ├── binaryset.h         # Binary serialization
│   ├── expected.h          # Error handling: expected<T>, Status enum
│   └── comp/index_param.h  # Index type constants, metric types, parameters
├── src/
│   ├── index/              # Index implementations
│   │   ├── flat/           # Brute-force flat index
│   │   ├── ivf/            # IVF (Inverted File) variants
│   │   ├── hnsw/           # HNSW graph-based index
│   │   ├── diskann/        # Disk-based ANN index
│   │   ├── sparse/         # Sparse vector indexes
│   │   ├── minhash/        # MinHash LSH index
│   │   ├── gpu/            # GPU-accelerated indexes (Faiss)
│   │   ├── gpu_cuvs/       # RAPIDS CUVS GPU indexes
│   │   ├── index_factory.cc
│   │   └── index_node.cc
│   ├── common/             # Shared utilities, config, metrics, tracing
│   ├── cluster/            # Clustering operations
│   └── io/                 # Serialization/deserialization
├── tests/
│   ├── ut/                 # C++ unit tests (Catch2)
│   ├── python/             # Python integration tests (pytest)
│   └── faiss/              # Faiss library tests
├── thirdparty/             # Vendored dependencies
│   ├── faiss/              # Meta's FAISS library
│   ├── hnswlib/            # HNSW graph index
│   └── DiskANN/            # Microsoft DiskANN
├── python/                 # Python bindings (SWIG) and wheel building
├── cmake/                  # CMake modules and library integration
│   └── libs/               # libfaiss, libhnsw, libdiskann, libcuvs configs
├── benchmark/              # Benchmarking code
├── scripts/                # Build and utility scripts
└── .github/                # CI workflows and actions
```

## Building

### Prerequisites

```bash
sudo apt install build-essential libopenblas-openmp-dev libaio-dev python3-dev python3-pip
pip3 install conan==1.61.0 --user
```

### Build Commands

```bash
mkdir build && cd build
conan remote add default-conan-local https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local

# Debug with unit tests
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Debug

# Release with unit tests
conan install .. --build=missing -o with_ut=True -s compiler.libcxx=libstdc++11 -s build_type=Release

# With DiskANN support
conan install .. --build=missing -o with_ut=True -o with_diskann=True -s compiler.libcxx=libstdc++11 -s build_type=Release

# Build
conan build ..
```

### Key Build Options (Conan `-o` flags)

| Option | Default | Description |
|--------|---------|-------------|
| `with_ut` | False | Build unit tests (Catch2) |
| `with_diskann` | False | DiskANN index support |
| `with_pageann` | False | PageANN index (requires DiskANN) |
| `with_cuvs` | False | GPU support via RAPIDS CUVS |
| `with_cardinal` | False | Enterprise vector engine (cloud only) |
| `with_benchmark` | False | Build benchmarks |
| `with_coverage` | False | Code coverage instrumentation |
| `with_asan` | False | Address sanitizer |
| `with_light` | False | Lightweight build (reduced index set) |
| `with_faiss_tests` | False | Faiss unit tests |

### Running Tests

```bash
# From the build directory
./Debug/tests/ut/knowhere_tests    # Debug build
./Release/tests/ut/knowhere_tests  # Release build
```

## Code Style and Conventions

### Formatting

Code style is enforced via **clang-format** (Google-based style with customizations):
- 4-space indentation
- 120-character column limit
- Return type always on its own line (`AlwaysBreakAfterReturnType: All`)
- No single-line blocks or functions
- Access modifier offset: -3

Run formatting: `clang-format -style=file -i <file>`

### Static Analysis

**clang-tidy** is configured with all warnings as errors. Key checks:
- Google style rules
- `modernize-*` (C++17 idioms)
- `performance-*` (efficiency)
- `clang-analyzer-*` (static analysis)

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `IndexNode`, `BinarySet`, `DiskANNConfig` |
| Methods/Functions | PascalCase | `GetByName()`, `CheckAndAdjust()`, `Search()` |
| Private members | snake_case with trailing `_` | `index_`, `binary_map_`, `search_pool_` |
| Constants | kCamelCase or UPPER_CASE | `kSeed`, `kEfMinValue` |
| Enum values | UPPER_CASE | `TRAIN`, `SEARCH`, `VECTOR_FLOAT` |
| Macros | UPPER_CASE with prefix | `KNOWHERE_CONFIG_DECLARE_FIELD` |
| Namespaces | lowercase | `knowhere`, `knowhere::meta` |

### Header Files

Both `#pragma once` and traditional include guards (`#ifndef`/`#define`) are used. Either is acceptable.

Every file must include the Apache 2.0 license header:
```cpp
// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); ...
```

### Namespaces

All code lives under `namespace knowhere`. Key sub-namespaces:
- `knowhere::IndexEnum` - Index type string constants
- `knowhere::meta` - Metadata key constants
- `knowhere::indexparam` - Index parameter constants
- `knowhere::metric` - Distance metric constants

### Error Handling

Knowhere uses a **result type pattern** instead of exceptions at the API boundary:

```cpp
// Return expected<T> for operations that can fail
expected<DataSetPtr>
Search(const DataSetPtr dataset, const Json& json, const BitsetView& bitset) override;

// Use Status enum for simpler error returns
Status Train(const DataSetPtr dataset, const Json& json) override;

// Factory methods for creating results
return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");

// Early return macro
RETURN_IF_ERROR(some_operation());
```

Try-catch is used around third-party library calls (Faiss, HNSW) to convert exceptions into `Status` codes.

### Documentation

Use Doxygen-style comments for public API:
```cpp
/**
 * @brief Builds the index using the provided dataset and configuration.
 * @param dataset Dataset to build the index from.
 * @param cfg Configuration
 * @return Status.
 */
```

## Architecture and Key Patterns

### Index Plugin Architecture

New index types follow the **Factory + Template Method** pattern:

1. **Config class** - Extends `BaseConfig`, declares parameters with validation:
   ```cpp
   class MyIndexConfig : public BaseConfig {
       CFG_INT my_param;
       KNOHWERE_DECLARE_CONFIG(MyIndexConfig) {
           KNOWHERE_CONFIG_DECLARE_FIELD(my_param)
               .description("description")
               .set_default(48)
               .set_range(1, 2048)
               .for_train();
       }
   };
   ```

2. **IndexNode subclass** - Implements virtual methods: `Train()`, `Add()`, `Search()`, `RangeSearch()`, `Serialize()`, `Deserialize()`, `HasRawData()`

3. **Factory registration** - Register the index in the factory:
   ```cpp
   IndexFactory::Instance().Register<fp32>("MY_INDEX", creator_func);
   ```

4. **Index constant** - Add to `include/knowhere/comp/index_param.h` in `IndexEnum` namespace

5. **Tests** - Add Catch2 test file in `tests/ut/`

### Conditional Compilation

Feature-gated code uses preprocessor defines:
- `KNOWHERE_WITH_DISKANN` - DiskANN support
- `KNOWHERE_WITH_PAGEANN` - PageANN support
- `KNOWHERE_WITH_CARDINAL` - Enterprise features
- `KNOWHERE_WITH_LIGHT` - Lightweight build

### Thread Pools

Knowhere uses shared thread pools for build and search operations:
- `ThreadPool::GetGlobalSearchThreadPool()`
- `ThreadPool::GetGlobalBuildThreadPool()`

### Supported Index Types

| Category | Indexes |
|----------|---------|
| Flat | FLAT, BIN_FLAT |
| IVF | IVF_FLAT, IVF_FLAT_CC, IVF_PQ, IVF_SQ8, IVF_SQ_CC, SCANN, IVF_RABITQ |
| Graph | HNSW, HNSW_SQ, HNSW_PQ, HNSW_PRQ |
| Disk-based | DISKANN, PAGEANN |
| Sparse | SPARSE_INVERTED_INDEX, SPARSE_WAND, SPARSE_INVERTED_INDEX_CC, SPARSE_WAND_CC |
| MinHash | MINHASH_LSH |
| GPU (CUVS) | GPU_CUVS_BRUTE_FORCE, GPU_CUVS_IVF_FLAT, GPU_CUVS_IVF_PQ, GPU_CUVS_CAGRA |
| GPU (Faiss) | GPU_FAISS_FLAT, GPU_FAISS_IVF_FLAT, GPU_FAISS_IVF_PQ, GPU_FAISS_IVF_SQ8 |

### Supported Metrics

`L2`, `IP` (Inner Product), `COSINE`, `HAMMING`, `JACCARD`, `BM25`, `MAX_SIM`, `DTW`

### Data Types

`VECTOR_FLOAT` (fp32), `VECTOR_FLOAT16`, `VECTOR_BFLOAT16`, `VECTOR_INT8`, `VECTOR_BINARY`, `VECTOR_SPARSE_FLOAT`

## Testing

### Unit Tests (Catch2)

Tests are in `tests/ut/` using the Catch2 framework:

```cpp
#include "catch2/catch_test_macros.hpp"

TEST_CASE("Test Index Search", "[search]") {
    SECTION("basic search") {
        auto result = index.Search(dataset, json, bitset);
        REQUIRE(result.has_value());
    }
}
```

Key test utilities in `tests/ut/utils.h`:
- `GenDataSet(rows, dim, seed)` - Generate random float datasets
- `GenBinDataSet(rows, dim)` - Generate binary datasets
- `GetKNNRecall()` - Calculate recall metrics

### Python Tests (pytest)

Located in `tests/python/`, test the SWIG Python bindings.

## Pre-commit Setup

```bash
pip3 install pre-commit
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Hooks enforce: no large files, no merge conflicts, trailing whitespace cleanup, end-of-file newline, clang-format on C/C++/CUDA files. The `thirdparty/` directory is excluded.

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- **ut.yaml** - Unit tests on push/PR to main
- **analyzer.yaml** - clang-tidy static analysis
- **pre-commit.yml** - Pre-commit validation
- **release.yaml** - Release builds (manual trigger)
- **release-python.yml** - Python wheel publishing

## Key Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Boost | 1.83.0 | Program options, utilities |
| Folly | 2023.10.30.10 | Facebook's C++ library |
| nlohmann_json | 3.11.2 | JSON parsing (config) |
| glog | 0.6.0 | Logging |
| fmt | 9.1.0 | String formatting (header-only) |
| prometheus-cpp | 1.1.0 | Metrics collection |
| simde | 0.8.2 | Portable SIMD intrinsics |
| xxHash | 0.8.3 | Fast hashing |
| OpenTelemetry | 1.8.1.1 | Distributed tracing (not in light builds) |
| Catch2 | 3.3.1 | Unit testing framework |
| protobuf | 3.21.4 | Protocol Buffers serialization |

Third-party vendored libraries: **Faiss** (vector search algorithms), **hnswlib** (HNSW graph index), **DiskANN** (disk-based ANN).
