# PiPNN-DiskANN Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement `PIPNN_DISKANN` IndexNode — PiPNN graph construction (RBC + GEMM + HashPrune) feeding into DiskANN's existing PQ encoding and disk layout, producing a queryable DiskANN-compatible index.

**Architecture:** PiPNN builds a Vamana-format in-memory graph by partitioning data into overlapping subsets (RBC), computing all-pairs distances via Eigen GEMM, and pruning with history-independent HashPrune. The graph is serialized in DiskANN's `save_graph()` binary format, then DiskANN's existing `partition_with_ram_budget` + `create_disk_layout` functions handle PQ encoding and sector-aligned disk writes. Search/Deserialize delegate entirely to `PQFlashIndex` — zero changes to DiskANN thirdparty.

**Tech Stack:** C++17, Eigen 3.4 (GEMM), DiskANN thirdparty (PQ + disk layout), Catch2 (tests), Conan + CMake

---

## Key File Map (read before starting)

| Purpose | File |
|---------|------|
| DiskANN Build() to copy from | `src/index/diskann/diskann.cc:409-486` |
| DiskANN registration pattern | `src/index/diskann/diskann.cc:1066-1069` |
| DiskANN Config to extend | `src/index/diskann/diskann_config.h` |
| Graph binary format (save_graph) | `thirdparty/DiskANN/src/index.cpp:389-416` |
| create_disk_layout signature | `thirdparty/DiskANN/src/aux_utils.cpp:822-825` |
| build_disk_index full flow | `thirdparty/DiskANN/src/aux_utils.cpp:1677-1960` |
| DiskANN cmake template | `cmake/libs/libdiskann.cmake` |
| Test template | `tests/ut/test_diskann.cc:1-80` |
| CMakeLists WITH_DISKANN block | `CMakeLists.txt:178-189` |
| conanfile deps pattern | `conanfile.py:121` (openblas example) |

## Vamana Graph Binary Format (critical — must match exactly)

```
Offset 0:  uint64_t  index_size        (total bytes, written twice: start + end)
Offset 8:  uint32_t  max_degree        (max observed neighbor count)
Offset 12: uint32_t  entry_point       (medoid node id)
Offset 16: uint64_t  num_frozen_pts    (always 0 for PiPNN)
--- Per node i=0..N-1 ---
uint32_t  num_neighbors
uint32_t  neighbor_ids[num_neighbors]
```

Entry point = dataset medoid (centroid nearest neighbor). Compute as:
```cpp
// Find centroid, then nearest point to centroid
std::vector<float> centroid(d, 0);
for (uint32_t i = 0; i < n; i++)
  for (uint32_t j = 0; j < d; j++)
    centroid[j] += data[i*d + j] / n;
// brute-force find nearest to centroid → entry_point
```

---

## Task 1: Add Eigen Dependency

**Files:**
- Modify: `conanfile.py`
- Modify: `conanfile_v2.py`
- Modify: `conanfile_v3.py`
- Create: `cmake/libs/libpipnn_diskann.cmake`
- Modify: `CMakeLists.txt`

### Step 1: Add Eigen to conanfile.py

In `conanfile.py`, find the `requirements()` method or the `self.requires(...)` block (around line 121 where openblas appears). Add:

```python
self.requires("eigen/3.4.0")
```

### Step 2: Same for conanfile_v2.py and conanfile_v3.py

Repeat the same `self.requires("eigen/3.4.0")` addition in both files following the same pattern as the existing requirements.

### Step 3: Create cmake/libs/libpipnn_diskann.cmake

```cmake
# cmake/libs/libpipnn_diskann.cmake
# PiPNN-DiskANN: requires DiskANN to be already included
# Eigen is header-only, no link needed — just include dirs

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

message(STATUS "PiPNN-DiskANN: Eigen3 found at ${EIGEN3_INCLUDE_DIR}")
```

### Step 4: Add WITH_PIPNN to CMakeLists.txt

After the existing `WITH_PAGEANN` block (around line 182-185), add inside the `if(WITH_DISKANN)` block:

```cmake
if(WITH_PIPNN)
  add_definitions(-DKNOWHERE_WITH_PIPNN)
  include(cmake/libs/libpipnn_diskann.cmake)
  message(STATUS "Building with PiPNN-DiskANN construction")
endif()
```

### Step 5: Verify conan can resolve eigen

```bash
# In build/ directory (Linux container)
conan install .. --build=missing -o with_ut=True -o with_diskann=True \
  -s compiler.libcxx=libstdc++11 -s build_type=Release 2>&1 | grep -i eigen
```

Expected: `eigen/3.4.0` appears in dependency resolution.

### Step 6: Commit

```bash
git add conanfile.py conanfile_v2.py conanfile_v3.py \
        cmake/libs/libpipnn_diskann.cmake CMakeLists.txt
git commit -m "build: add Eigen3 dependency and WITH_PIPNN CMake option"
```

---

## Task 2: HashPrune

**Files:**
- Create: `src/index/diskann/impl/hash_prune.h`
- Create: `tests/ut/test_pipnn_diskann.cc` (start the test file)

### Step 1: Create the test file with HashPrune tests

```cpp
// tests/ut/test_pipnn_diskann.cc
// Copyright (C) 2019-2023 Zilliz. All rights reserved.
// (Apache 2.0 header)

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_approx.hpp"
#include "index/diskann/impl/hash_prune.h"

#include <algorithm>
#include <random>
#include <vector>

namespace {
constexpr uint32_t kDim = 32;
constexpr uint32_t kMaxDegree = 16;
constexpr uint32_t kHashBits = 12;
}

// Helper: generate random float vector
std::vector<float> rand_vec(uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

float l2_dist(const float* a, const float* b, uint32_t d) {
    float s = 0;
    for (uint32_t i = 0; i < d; i++) s += (a[i]-b[i])*(a[i]-b[i]);
    return s;
}

TEST_CASE("HashPrune: history-independent insertion", "[pipnn][hash_prune]") {
    std::mt19937 rng(42);
    auto p_vec = rand_vec(kDim, rng);

    // Generate 50 candidate vectors
    std::vector<std::vector<float>> candidates(50);
    for (auto& c : candidates) c = rand_vec(kDim, rng);

    knowhere::pipnn::HashPrune prune_fwd(kDim, kHashBits, kMaxDegree);
    knowhere::pipnn::HashPrune prune_rev(kDim, kHashBits, kMaxDegree);

    // Forward order
    for (uint32_t i = 0; i < candidates.size(); i++) {
        float d = l2_dist(p_vec.data(), candidates[i].data(), kDim);
        prune_fwd.insert(i, p_vec.data(), candidates[i].data(), d);
    }

    // Reverse order
    for (int i = (int)candidates.size()-1; i >= 0; i--) {
        float d = l2_dist(p_vec.data(), candidates[i].data(), kDim);
        prune_rev.insert((uint32_t)i, p_vec.data(), candidates[i].data(), d);
    }

    auto nbrs_fwd = prune_fwd.neighbors();
    auto nbrs_rev = prune_rev.neighbors();

    // Sort both for comparison (order within reservoir may differ)
    std::sort(nbrs_fwd.begin(), nbrs_fwd.end());
    std::sort(nbrs_rev.begin(), nbrs_rev.end());

    REQUIRE(nbrs_fwd == nbrs_rev);
}

TEST_CASE("HashPrune: reservoir never exceeds max_degree", "[pipnn][hash_prune]") {
    std::mt19937 rng(99);
    auto p_vec = rand_vec(kDim, rng);

    knowhere::pipnn::HashPrune prune(kDim, kHashBits, kMaxDegree);

    for (uint32_t i = 0; i < 500; i++) {
        auto c = rand_vec(kDim, rng);
        float d = l2_dist(p_vec.data(), c.data(), kDim);
        prune.insert(i, p_vec.data(), c.data(), d);
    }

    REQUIRE(prune.neighbors().size() <= kMaxDegree);
}

TEST_CASE("HashPrune: closer candidate replaces farther in same bucket",
          "[pipnn][hash_prune]") {
    std::mt19937 rng(7);
    auto p_vec = rand_vec(kDim, rng);

    // Use many hash bits so collisions are rare — we'll force collision
    // by using identical direction vectors scaled differently
    knowhere::pipnn::HashPrune prune(kDim, kHashBits, kMaxDegree);

    // c_far and c_near are in same direction from p (so same hash bucket)
    auto direction = rand_vec(kDim, rng);
    std::vector<float> c_far(kDim), c_near(kDim);
    for (uint32_t j = 0; j < kDim; j++) {
        c_far[j]  = p_vec[j] + direction[j] * 2.0f;
        c_near[j] = p_vec[j] + direction[j] * 0.5f;
    }

    float d_far  = l2_dist(p_vec.data(), c_far.data(), kDim);
    float d_near = l2_dist(p_vec.data(), c_near.data(), kDim);

    prune.insert(0, p_vec.data(), c_far.data(), d_far);
    prune.insert(1, p_vec.data(), c_near.data(), d_near);

    auto nbrs = prune.neighbors();
    // c_near should be in the result, c_far should be replaced
    bool has_near = std::find(nbrs.begin(), nbrs.end(), 1u) != nbrs.end();
    bool has_far  = std::find(nbrs.begin(), nbrs.end(), 0u) != nbrs.end();
    // With high probability (same direction → same hash), near replaces far
    // This test may occasionally pass both if hash doesn't collide — that's ok
    REQUIRE((has_near || has_far));  // At minimum one is present
    if (has_near && has_far) {
        // Both present means no collision — fine
    } else {
        REQUIRE(has_near);  // If one was evicted, it must be c_far
    }
}
```

### Step 2: Run tests — expect compile error (file doesn't exist yet)

```bash
# In container, build directory
cmake -DWITH_UT=ON -DWITH_DISKANN=ON -DWITH_PIPNN=ON .. 2>&1 | tail -5
```

Expected: error about missing `hash_prune.h`.

### Step 3: Implement HashPrune

```cpp
// src/index/diskann/impl/hash_prune.h
// Copyright (C) 2019-2023 Zilliz. All rights reserved.
// Apache 2.0 License

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace knowhere::pipnn {

// HashPrune: history-independent online pruning via residualized LSH.
// Each slot covers one angular cone from point p. Maintains the closest
// candidate per cone, regardless of insertion order (Theorem 3.1, PiPNN paper).
//
// Memory per slot: 8 bytes (4B id + 2B hash + 2B bfloat16 dist)
// Total: 8 * max_degree bytes per point
class HashPrune {
 public:
    HashPrune(uint32_t dim, uint32_t hash_bits, uint32_t max_degree)
        : dim_(dim), m_(hash_bits), max_degree_(max_degree) {
        reservoir_.resize(max_degree_);
        size_ = 0;
        farthest_idx_ = 0;
        farthest_dist_ = 0.0f;

        // Pre-generate m random hyperplanes (m x dim matrix)
        hyperplanes_.resize(m_ * dim_);
        std::mt19937 rng(/*seed=*/12345 + dim_ * 31 + m_);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& h : hyperplanes_) h = normal(rng);
    }

    // compute_sketch: project vec onto m hyperplanes → m floats
    // Caller allocates sketch[m_]
    void compute_sketch(const float* vec, float* sketch) const {
        for (uint32_t i = 0; i < m_; i++) {
            float dot = 0.0f;
            const float* hp = &hyperplanes_[i * dim_];
            for (uint32_t j = 0; j < dim_; j++) dot += hp[j] * vec[j];
            sketch[i] = dot;
        }
    }

    // residual_hash: m-bit hash of (c - p) using sketch difference
    // sketch_p = compute_sketch(p), sketch_c = compute_sketch(c)
    uint16_t residual_hash(const float* sketch_p, const float* sketch_c) const {
        uint16_t h = 0;
        for (uint32_t i = 0; i < m_ && i < 16; i++) {
            // Sign of (sketch_c[i] - sketch_p[i]) == sign of H_i · (c - p)
            if (sketch_c[i] >= sketch_p[i]) h |= (1u << i);
        }
        return h;
    }

    // insert: stream candidate (candidate_id, dist from p) into reservoir.
    // sketch_p = pre-computed sketch of point p
    // sketch_c = pre-computed sketch of candidate c
    // Returns true if candidate was accepted.
    bool insert(uint32_t candidate_id,
                const float* sketch_p,
                const float* sketch_c,
                float dist) {
        uint16_t h = residual_hash(sketch_p, sketch_c);
        uint16_t dist_bf = float_to_bfloat16(dist);

        // Case 1: find collision slot (same hash)
        for (uint32_t i = 0; i < size_; i++) {
            if (reservoir_[i].hash == h) {
                if (dist < bfloat16_to_float(reservoir_[i].dist_bf)) {
                    reservoir_[i].id = candidate_id;
                    reservoir_[i].dist_bf = dist_bf;
                    // Recompute farthest if we replaced it
                    if (i == farthest_idx_) recompute_farthest();
                    return true;
                }
                return false;  // farther than existing in this bucket
            }
        }

        // Case 2: no collision, reservoir not full
        if (size_ < max_degree_) {
            reservoir_[size_] = {candidate_id, h, dist_bf};
            if (dist > farthest_dist_) {
                farthest_dist_ = dist;
                farthest_idx_ = size_;
            }
            size_++;
            return true;
        }

        // Case 3: no collision, reservoir full — replace farthest if closer
        if (dist < farthest_dist_) {
            reservoir_[farthest_idx_] = {candidate_id, h, dist_bf};
            recompute_farthest();
            return true;
        }
        return false;
    }

    // Return final neighbor IDs
    std::vector<uint32_t> neighbors() const {
        std::vector<uint32_t> result(size_);
        for (uint32_t i = 0; i < size_; i++) result[i] = reservoir_[i].id;
        return result;
    }

    uint32_t size() const { return size_; }

 private:
    struct Slot {
        uint32_t id      = 0;
        uint16_t hash    = 0;
        uint16_t dist_bf = 0;  // bfloat16 distance to p
    };

    uint32_t dim_;
    uint32_t m_;           // number of hash bits
    uint32_t max_degree_;  // reservoir capacity

    std::vector<Slot>  reservoir_;
    uint32_t           size_;
    uint32_t           farthest_idx_;
    float              farthest_dist_;

    std::vector<float> hyperplanes_;  // m × dim

    void recompute_farthest() {
        farthest_dist_ = 0.0f;
        farthest_idx_  = 0;
        for (uint32_t i = 0; i < size_; i++) {
            float d = bfloat16_to_float(reservoir_[i].dist_bf);
            if (d > farthest_dist_) {
                farthest_dist_ = d;
                farthest_idx_  = i;
            }
        }
    }

    static uint16_t float_to_bfloat16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    }

    static float bfloat16_to_float(uint16_t bf) {
        uint32_t bits = static_cast<uint32_t>(bf) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

}  // namespace knowhere::pipnn
```

### Step 4: Add hash_prune test to CMakeLists tests/ut/

In `tests/ut/CMakeLists.txt`, find how test files are added. The pattern is typically a glob or explicit list. Add `test_pipnn_diskann.cc` the same way DiskANN tests are included (guarded by `WITH_DISKANN`).

### Step 5: Build and run HashPrune tests

```bash
# In container build directory
cmake -DWITH_UT=ON -DWITH_DISKANN=ON -DWITH_PIPNN=ON \
      -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc) knowhere_tests 2>&1 | tail -20

./Release/tests/ut/knowhere_tests "[hash_prune]" -v
```

Expected:
```
All tests passed (N assertions in 3 test cases)
```

### Step 6: Commit

```bash
git add src/index/diskann/impl/hash_prune.h tests/ut/test_pipnn_diskann.cc \
        tests/ut/CMakeLists.txt
git commit -m "feat: implement HashPrune online pruning algorithm"
```

---

## Task 3: RBC Partitioner

**Files:**
- Create: `src/index/diskann/impl/rbc_partition.h`
- Create: `src/index/diskann/impl/rbc_partition.cc`
- Modify: `tests/ut/test_pipnn_diskann.cc` (add tests)

### Step 1: Write failing tests — add to test_pipnn_diskann.cc

```cpp
#include "index/diskann/impl/rbc_partition.h"

TEST_CASE("RBC: every point appears in at least one leaf", "[pipnn][rbc]") {
    constexpr uint32_t N = 500, D = 16;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> data(N * D);
    for (auto& x : data) x = dist(rng);

    knowhere::pipnn::RBCPartitioner::Config cfg;
    cfg.leaf_max_size = 50;
    cfg.fanout_l1 = 3;
    cfg.fanout_l2 = 2;

    knowhere::pipnn::RBCPartitioner partitioner;
    auto leaves = partitioner.partition(data.data(), N, D, cfg);

    // Every point must appear in at least one leaf
    std::vector<bool> seen(N, false);
    for (auto& leaf : leaves) {
        for (uint32_t id : leaf.point_ids) {
            REQUIRE(id < N);
            seen[id] = true;
        }
    }
    for (uint32_t i = 0; i < N; i++) {
        REQUIRE(seen[i]);
    }
}

TEST_CASE("RBC: all leaf sizes <= leaf_max_size", "[pipnn][rbc]") {
    constexpr uint32_t N = 500, D = 16;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> data(N * D);
    for (auto& x : data) x = dist(rng);

    knowhere::pipnn::RBCPartitioner::Config cfg;
    cfg.leaf_max_size = 50;

    knowhere::pipnn::RBCPartitioner partitioner;
    auto leaves = partitioner.partition(data.data(), N, D, cfg);

    for (auto& leaf : leaves) {
        REQUIRE(leaf.point_ids.size() <= cfg.leaf_max_size);
    }
    REQUIRE(!leaves.empty());
}

TEST_CASE("RBC: overlap factor is reasonable", "[pipnn][rbc]") {
    constexpr uint32_t N = 1000, D = 32;
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> data(N * D);
    for (auto& x : data) x = dist(rng);

    knowhere::pipnn::RBCPartitioner::Config cfg;
    cfg.leaf_max_size = 100;
    cfg.fanout_l1 = 10;
    cfg.fanout_l2 = 3;

    knowhere::pipnn::RBCPartitioner partitioner;
    auto leaves = partitioner.partition(data.data(), N, D, cfg);

    uint64_t total_points = 0;
    for (auto& leaf : leaves) total_points += leaf.point_ids.size();

    float overlap = static_cast<float>(total_points) / N;
    // With fanout_l1=10, fanout_l2=3, overlap should be 2-8x
    REQUIRE(overlap >= 1.5f);
    REQUIRE(overlap <= 30.0f);
}
```

### Step 2: Run — expect compile failure

```bash
./Release/tests/ut/knowhere_tests "[rbc]" -v
```

### Step 3: Implement RBC Partitioner

**`src/index/diskann/impl/rbc_partition.h`:**

```cpp
// src/index/diskann/impl/rbc_partition.h
#pragma once
#include <cstdint>
#include <vector>

namespace knowhere::pipnn {

struct Leaf {
    std::vector<uint32_t> point_ids;
};

class RBCPartitioner {
 public:
    struct Config {
        uint32_t leaf_max_size = 1000;   // C_max
        uint32_t fanout_l1     = 10;     // top-level fanout
        uint32_t fanout_l2     = 3;      // level-2 fanout
        uint32_t fanout_rest   = 1;      // deeper levels
        uint32_t num_leaders   = 0;      // 0 = auto: min(sqrt(n), 1000)
    };

    std::vector<Leaf> partition(
        const float* data, uint32_t n, uint32_t d, const Config& cfg);

 private:
    void partition_recursive(
        const float* data, uint32_t d,
        const std::vector<uint32_t>& point_ids,
        uint32_t depth,
        const Config& cfg,
        std::vector<Leaf>& out_leaves);

    // For each point in point_ids, find its k nearest leaders.
    // Returns assignment[i] = sorted list of up to k leader indices (into leader_ids).
    std::vector<std::vector<uint32_t>> assign_to_k_leaders(
        const float* data, uint32_t d,
        const std::vector<uint32_t>& point_ids,
        const std::vector<uint32_t>& leader_ids,
        uint32_t k);

    float l2sq(const float* a, const float* b, uint32_t d) const;
};

}  // namespace knowhere::pipnn
```

**`src/index/diskann/impl/rbc_partition.cc`:**

```cpp
// src/index/diskann/impl/rbc_partition.cc
#include "index/diskann/impl/rbc_partition.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace knowhere::pipnn {

float RBCPartitioner::l2sq(const float* a, const float* b, uint32_t d) const {
    float s = 0.0f;
    for (uint32_t i = 0; i < d; i++) s += (a[i] - b[i]) * (a[i] - b[i]);
    return s;
}

std::vector<std::vector<uint32_t>> RBCPartitioner::assign_to_k_leaders(
    const float* data, uint32_t d,
    const std::vector<uint32_t>& point_ids,
    const std::vector<uint32_t>& leader_ids,
    uint32_t k) {

    uint32_t k_actual = std::min(k, (uint32_t)leader_ids.size());
    std::vector<std::vector<uint32_t>> assignments(point_ids.size());

    for (uint32_t pi = 0; pi < point_ids.size(); pi++) {
        const float* p = data + point_ids[pi] * d;

        // Compute distance to all leaders, keep top-k closest
        std::vector<std::pair<float, uint32_t>> dists(leader_ids.size());
        for (uint32_t li = 0; li < leader_ids.size(); li++) {
            dists[li] = {l2sq(p, data + leader_ids[li] * d, d), li};
        }
        std::partial_sort(dists.begin(), dists.begin() + k_actual, dists.end());

        assignments[pi].resize(k_actual);
        for (uint32_t ki = 0; ki < k_actual; ki++) {
            assignments[pi][ki] = dists[ki].second;  // leader index (not global id)
        }
    }
    return assignments;
}

void RBCPartitioner::partition_recursive(
    const float* data, uint32_t d,
    const std::vector<uint32_t>& point_ids,
    uint32_t depth,
    const Config& cfg,
    std::vector<Leaf>& out_leaves) {

    if (point_ids.size() <= cfg.leaf_max_size) {
        out_leaves.push_back({point_ids});
        return;
    }

    uint32_t n = point_ids.size();
    uint32_t n_leaders = cfg.num_leaders > 0
        ? cfg.num_leaders
        : std::min((uint32_t)std::sqrt((double)n), 1000u);
    n_leaders = std::max(n_leaders, 2u);
    n_leaders = std::min(n_leaders, n);

    // Select random leaders
    std::mt19937 rng(depth * 1234567 + n);
    std::vector<uint32_t> leader_indices(n);
    std::iota(leader_indices.begin(), leader_indices.end(), 0);
    std::shuffle(leader_indices.begin(), leader_indices.end(), rng);
    leader_indices.resize(n_leaders);

    std::vector<uint32_t> leader_ids(n_leaders);
    for (uint32_t i = 0; i < n_leaders; i++) {
        leader_ids[i] = point_ids[leader_indices[i]];
    }

    // Determine fanout for this depth
    uint32_t fanout = (depth == 0) ? cfg.fanout_l1
                    : (depth == 1) ? cfg.fanout_l2
                    : cfg.fanout_rest;

    // Assign each point to its fanout nearest leaders
    auto assignments = assign_to_k_leaders(data, d, point_ids, leader_ids, fanout);

    // Group points by leader
    std::vector<std::vector<uint32_t>> buckets(n_leaders);
    for (uint32_t pi = 0; pi < point_ids.size(); pi++) {
        for (uint32_t li : assignments[pi]) {
            buckets[li].push_back(point_ids[pi]);
        }
    }

    // Recurse on each non-empty bucket
    for (auto& bucket : buckets) {
        if (bucket.empty()) continue;
        partition_recursive(data, d, bucket, depth + 1, cfg, out_leaves);
    }
}

std::vector<Leaf> RBCPartitioner::partition(
    const float* data, uint32_t n, uint32_t d, const Config& cfg) {
    std::vector<uint32_t> all_ids(n);
    std::iota(all_ids.begin(), all_ids.end(), 0);

    std::vector<Leaf> leaves;
    partition_recursive(data, d, all_ids, 0, cfg, leaves);
    return leaves;
}

}  // namespace knowhere::pipnn
```

### Step 4: Add rbc_partition.cc to the build

In `cmake/libs/libpipnn_diskann.cmake`, add the source:

```cmake
set(PIPNN_SOURCES
    ${CMAKE_SOURCE_DIR}/src/index/diskann/impl/rbc_partition.cc)
# Will be compiled into knowhere target via diskann sources glob
```

Or simply ensure the `src/index/diskann/*.cc` glob in `CMakeLists.txt` picks it up (it won't — it's in `impl/`). Add to CMakeLists.txt inside `WITH_DISKANN` block:

```cmake
if(WITH_PIPNN)
  add_definitions(-DKNOWHERE_WITH_PIPNN)
  include(cmake/libs/libpipnn_diskann.cmake)
  knowhere_file_glob(GLOB_RECURSE KNOWHERE_PIPNN_SRCS
                     src/index/diskann/impl/*.cc)
  list(APPEND KNOWHERE_SRCS ${KNOWHERE_PIPNN_SRCS})
endif()
```

### Step 5: Build and run RBC tests

```bash
make -j$(nproc) knowhere_tests && \
./Release/tests/ut/knowhere_tests "[rbc]" -v
```

Expected: All 3 RBC tests pass.

### Step 6: Commit

```bash
git add src/index/diskann/impl/rbc_partition.h \
        src/index/diskann/impl/rbc_partition.cc \
        tests/ut/test_pipnn_diskann.cc \
        cmake/libs/libpipnn_diskann.cmake CMakeLists.txt
git commit -m "feat: implement RBC partitioner with multi-level fanout"
```

---

## Task 4: GEMM Leaf Builder (PiPNNBuilder)

**Files:**
- Create: `src/index/diskann/impl/pipnn_builder.h`
- Create: `src/index/diskann/impl/pipnn_builder.cc`
- Modify: `tests/ut/test_pipnn_diskann.cc`

### Step 1: Write failing builder test

Add to `tests/ut/test_pipnn_diskann.cc`:

```cpp
#include "index/diskann/impl/pipnn_builder.h"

TEST_CASE("PiPNNBuilder: basic graph construction", "[pipnn][builder]") {
    constexpr uint32_t N = 200, D = 16, K = 10;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> data(N * D);
    for (auto& x : data) x = dist(rng);

    knowhere::pipnn::PiPNNBuilder::Config cfg;
    cfg.k_nn       = 3;
    cfg.hash_bits  = 12;
    cfg.max_degree = 32;
    cfg.final_prune = false;  // skip RobustPrune for this test

    knowhere::pipnn::RBCPartitioner::Config rbc_cfg;
    rbc_cfg.leaf_max_size = 50;
    rbc_cfg.fanout_l1 = 3;

    knowhere::pipnn::PiPNNBuilder builder;
    auto graph = builder.build(data.data(), N, D, cfg, rbc_cfg);

    // Every point has at least 1 neighbor
    REQUIRE(graph.size() == N);
    for (uint32_t i = 0; i < N; i++) {
        REQUIRE(!graph[i].empty());
        REQUIRE(graph[i].size() <= cfg.max_degree);
        for (uint32_t nb : graph[i]) {
            REQUIRE(nb < N);
            REQUIRE(nb != i);  // no self-loops
        }
    }
}

TEST_CASE("PiPNNBuilder: graph connectivity", "[pipnn][builder]") {
    // Build graph and verify all nodes are reachable from entry point via BFS
    constexpr uint32_t N = 300, D = 16;
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<float> data(N * D);
    for (auto& x : data) x = dist(rng);

    knowhere::pipnn::PiPNNBuilder::Config cfg;
    cfg.max_degree = 32;
    cfg.final_prune = false;

    knowhere::pipnn::RBCPartitioner::Config rbc_cfg;
    rbc_cfg.leaf_max_size = 60;
    rbc_cfg.fanout_l1 = 4;
    rbc_cfg.fanout_l2 = 2;

    knowhere::pipnn::PiPNNBuilder builder;
    auto graph = builder.build(data.data(), N, D, cfg, rbc_cfg);

    // BFS from node 0
    std::vector<bool> visited(N, false);
    std::vector<uint32_t> queue = {0};
    visited[0] = true;
    while (!queue.empty()) {
        uint32_t cur = queue.back(); queue.pop_back();
        for (uint32_t nb : graph[cur]) {
            if (!visited[nb]) {
                visited[nb] = true;
                queue.push_back(nb);
            }
        }
    }
    uint32_t reachable = std::count(visited.begin(), visited.end(), true);
    // Expect high connectivity — at least 80% reachable
    REQUIRE(reachable >= N * 0.8f);
}
```

### Step 2: Implement PiPNNBuilder header

```cpp
// src/index/diskann/impl/pipnn_builder.h
#pragma once

#include "index/diskann/impl/hash_prune.h"
#include "index/diskann/impl/rbc_partition.h"

#include <cstdint>
#include <mutex>
#include <vector>

namespace knowhere::pipnn {

class PiPNNBuilder {
 public:
    struct Config {
        uint32_t k_nn        = 3;     // bi-directed k-NN k within leaf
        uint32_t hash_bits   = 12;    // HashPrune m value
        uint32_t max_degree  = 64;    // ℓ_max (DiskANN max_degree)
        float    alpha       = 1.2f;  // RobustPrune alpha (if final_prune=true)
        bool     final_prune = true;  // run RobustPrune pass after HashPrune
        uint32_t num_threads = 0;     // 0 = hardware_concurrency
    };

    // Returns adjacency list: graph[i] = sorted neighbor IDs of point i
    std::vector<std::vector<uint32_t>> build(
        const float* data, uint32_t n, uint32_t d,
        const Config& cfg,
        const RBCPartitioner::Config& rbc_cfg);

 private:
    void process_leaf(
        const float* data, uint32_t d,
        const Leaf& leaf,
        uint32_t k,
        const std::vector<std::vector<float>>& sketches,
        std::vector<HashPrune>& pruners,
        std::vector<std::mutex>& mutexes);

    void robust_prune_pass(
        const float* data, uint32_t n, uint32_t d,
        float alpha, uint32_t max_degree,
        std::vector<std::vector<uint32_t>>& graph);

    float l2sq(const float* a, const float* b, uint32_t d) const;
};

}  // namespace knowhere::pipnn
```

### Step 3: Implement PiPNNBuilder body

```cpp
// src/index/diskann/impl/pipnn_builder.cc
#include "index/diskann/impl/pipnn_builder.h"

#include <Eigen/Dense>
#include <algorithm>
#include <future>
#include <numeric>
#include <thread>

namespace knowhere::pipnn {

float PiPNNBuilder::l2sq(const float* a, const float* b, uint32_t d) const {
    float s = 0.f;
    for (uint32_t i = 0; i < d; i++) s += (a[i]-b[i])*(a[i]-b[i]);
    return s;
}

void PiPNNBuilder::process_leaf(
    const float* data, uint32_t d,
    const Leaf& leaf,
    uint32_t k,
    const std::vector<std::vector<float>>& sketches,
    std::vector<HashPrune>& pruners,
    std::vector<std::mutex>& mutexes) {

    uint32_t n_leaf = leaf.point_ids.size();
    if (n_leaf < 2) return;

    // Step 1: Build Eigen matrix of leaf vectors (n_leaf × d)
    Eigen::MatrixXf X(n_leaf, d);
    for (uint32_t i = 0; i < n_leaf; i++) {
        uint32_t gid = leaf.point_ids[i];
        X.row(i) = Eigen::Map<const Eigen::RowVectorXf>(data + gid * d, d);
    }

    // Step 2: L2 distance matrix via GEMM
    // ||a-b||² = ||a||² + ||b||² - 2·a·bᵀ
    Eigen::VectorXf norms = X.rowwise().squaredNorm();
    Eigen::MatrixXf D_mat = -2.0f * X * X.transpose();
    D_mat.colwise() += norms;
    D_mat.rowwise() += norms.transpose();
    // Zero or negative diagonal due to float precision → set to infinity
    for (uint32_t i = 0; i < n_leaf; i++) D_mat(i, i) = 1e30f;

    // Step 3: For each row, find k nearest (partial sort)
    uint32_t k_actual = std::min(k, n_leaf - 1);
    std::vector<uint32_t> order(n_leaf);

    for (uint32_t i = 0; i < n_leaf; i++) {
        uint32_t gid_i = leaf.point_ids[i];
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + k_actual, order.end(),
            [&](uint32_t a, uint32_t b) { return D_mat(i, a) < D_mat(i, b); });

        // Step 4: Insert bi-directed edges into global HashPrune
        const float* sp_i = sketches[gid_i].data();

        for (uint32_t ki = 0; ki < k_actual; ki++) {
            uint32_t j = order[ki];
            uint32_t gid_j = leaf.point_ids[j];
            float dist = D_mat(i, j);
            if (dist <= 0) continue;

            const float* sp_j = sketches[gid_j].data();

            // Forward edge: i → j
            {
                std::lock_guard<std::mutex> lk(mutexes[gid_i]);
                pruners[gid_i].insert(gid_j, sp_i, sp_j, dist);
            }
            // Backward edge: j → i
            {
                std::lock_guard<std::mutex> lk(mutexes[gid_j]);
                pruners[gid_j].insert(gid_i, sp_j, sp_i, dist);
            }
        }
    }
}

void PiPNNBuilder::robust_prune_pass(
    const float* data, uint32_t n, uint32_t d,
    float alpha, uint32_t max_degree,
    std::vector<std::vector<uint32_t>>& graph) {

    for (uint32_t i = 0; i < n; i++) {
        auto& nbrs = graph[i];
        if (nbrs.size() <= max_degree) continue;

        const float* p = data + i * d;
        // Sort candidates by distance to p
        std::sort(nbrs.begin(), nbrs.end(), [&](uint32_t a, uint32_t b) {
            return l2sq(p, data + a*d, d) < l2sq(p, data + b*d, d);
        });

        std::vector<uint32_t> selected;
        selected.reserve(max_degree);
        for (uint32_t c : nbrs) {
            if (selected.size() >= max_degree) break;
            bool dominated = false;
            const float* pc = data + c * d;
            float dist_pc = l2sq(p, pc, d);
            for (uint32_t s : selected) {
                if (l2sq(data + s*d, pc, d) < alpha * alpha * dist_pc) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) selected.push_back(c);
        }
        nbrs = std::move(selected);
    }
}

std::vector<std::vector<uint32_t>> PiPNNBuilder::build(
    const float* data, uint32_t n, uint32_t d,
    const Config& cfg,
    const RBCPartitioner::Config& rbc_cfg) {

    // Step 1: Pre-compute per-point sketches for HashPrune
    // Sketch dimension = hash_bits (one dot product per hyperplane)
    // We use a temporary HashPrune just to get hyperplanes
    HashPrune proto(d, cfg.hash_bits, cfg.max_degree);
    std::vector<std::vector<float>> sketches(n, std::vector<float>(cfg.hash_bits));
    for (uint32_t i = 0; i < n; i++) {
        proto.compute_sketch(data + i*d, sketches[i].data());
    }

    // Step 2: Per-point HashPrune reservoirs + mutexes
    std::vector<HashPrune> pruners;
    pruners.reserve(n);
    for (uint32_t i = 0; i < n; i++) {
        pruners.emplace_back(d, cfg.hash_bits, cfg.max_degree);
    }
    std::vector<std::mutex> mutexes(n);

    // Step 3: Partition data
    RBCPartitioner partitioner;
    auto leaves = partitioner.partition(data, n, d, rbc_cfg);

    // Step 4: Process leaves in parallel
    uint32_t nthreads = cfg.num_threads > 0
        ? cfg.num_threads
        : std::thread::hardware_concurrency();

    std::vector<std::future<void>> futures;
    futures.reserve(leaves.size());

    // Simple thread pool via async with limited concurrency
    uint32_t active = 0;
    for (auto& leaf : leaves) {
        // Throttle to nthreads concurrent tasks
        if (active >= nthreads) {
            futures.front().wait();
            futures.erase(futures.begin());
            active--;
        }
        futures.push_back(std::async(std::launch::async,
            [&, &leaf_ref = leaf]() {
                process_leaf(data, d, leaf_ref, cfg.k_nn,
                             sketches, pruners, mutexes);
            }));
        active++;
    }
    for (auto& f : futures) f.wait();

    // Step 5: Extract final neighbor lists
    std::vector<std::vector<uint32_t>> graph(n);
    for (uint32_t i = 0; i < n; i++) {
        graph[i] = pruners[i].neighbors();
    }

    // Step 6: Optional RobustPrune pass
    if (cfg.final_prune) {
        robust_prune_pass(data, n, d, cfg.alpha, cfg.max_degree, graph);
    }

    return graph;
}

}  // namespace knowhere::pipnn
```

### Step 4: Build and run builder tests

```bash
make -j$(nproc) knowhere_tests && \
./Release/tests/ut/knowhere_tests "[builder]" -v
```

Expected: Both builder tests pass.

### Step 5: Commit

```bash
git add src/index/diskann/impl/pipnn_builder.h \
        src/index/diskann/impl/pipnn_builder.cc \
        tests/ut/test_pipnn_diskann.cc
git commit -m "feat: implement PiPNNBuilder with GEMM leaf building and HashPrune"
```

---

## Task 5: Config + Vamana Graph Serializer

**Files:**
- Create: `src/index/diskann/pipnn_diskann_config.h`
- Create: `src/index/diskann/impl/vamana_serializer.h`

### Step 1: Write serializer test

Add to `tests/ut/test_pipnn_diskann.cc`:

```cpp
#include "index/diskann/impl/vamana_serializer.h"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

TEST_CASE("VamanaSerializer: round-trip read/write", "[pipnn][serializer]") {
    constexpr uint32_t N = 100;

    // Build a simple graph
    std::vector<std::vector<uint32_t>> graph(N);
    std::mt19937 rng(42);
    for (uint32_t i = 0; i < N; i++) {
        uint32_t n_nbrs = 5 + rng() % 5;
        for (uint32_t j = 0; j < n_nbrs; j++) {
            uint32_t nb = rng() % N;
            if (nb != i) graph[i].push_back(nb);
        }
    }

    std::string path = fs::temp_directory_path() / "test_vamana.index";

    knowhere::pipnn::VamanaSerializer::write(graph, /*entry_point=*/42u, path);
    REQUIRE(fs::exists(path));

    // Read header back manually to verify format
    std::ifstream f(path, std::ios::binary);
    uint64_t index_size; f.read((char*)&index_size, 8);
    uint32_t max_deg;    f.read((char*)&max_deg, 4);
    uint32_t ep;         f.read((char*)&ep, 4);
    uint64_t frozen;     f.read((char*)&frozen, 8);

    REQUIRE(ep == 42u);
    REQUIRE(frozen == 0u);
    REQUIRE(max_deg > 0);
    REQUIRE(index_size > 24u);

    fs::remove(path);
}
```

### Step 2: Implement VamanaSerializer

```cpp
// src/index/diskann/impl/vamana_serializer.h
// Writes in-memory graph in DiskANN's save_graph() binary format.
// See: thirdparty/DiskANN/src/index.cpp:389-416
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace knowhere::pipnn {

struct VamanaSerializer {
    // Write graph to file in DiskANN Vamana format.
    // entry_point: index of the medoid/entry node
    static void write(
        const std::vector<std::vector<uint32_t>>& graph,
        uint32_t entry_point,
        const std::string& path) {

        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open: " + path);

        uint32_t n = graph.size();

        // Reserve space for index_size (written again at end)
        uint64_t index_size = 24;  // header size
        out.write((char*)&index_size, sizeof(uint64_t));

        uint32_t max_degree = 0;
        for (auto& nbrs : graph)
            max_degree = std::max(max_degree, (uint32_t)nbrs.size());

        out.write((char*)&max_degree,   sizeof(uint32_t));
        out.write((char*)&entry_point,  sizeof(uint32_t));
        uint64_t num_frozen = 0;
        out.write((char*)&num_frozen,   sizeof(uint64_t));

        // Per-node neighbor lists
        for (uint32_t i = 0; i < n; i++) {
            uint32_t gk = (uint32_t)graph[i].size();
            out.write((char*)&gk, sizeof(uint32_t));
            if (gk > 0)
                out.write((char*)graph[i].data(), gk * sizeof(uint32_t));
            index_size += sizeof(uint32_t) * (gk + 1);
        }

        // Rewrite correct index_size at offset 0
        out.seekp(0);
        out.write((char*)&index_size, sizeof(uint64_t));
        out.close();
    }

    // Find medoid: point nearest to the dataset centroid
    static uint32_t find_medoid(const float* data, uint32_t n, uint32_t d) {
        std::vector<float> centroid(d, 0.0f);
        for (uint32_t i = 0; i < n; i++)
            for (uint32_t j = 0; j < d; j++)
                centroid[j] += data[i*d+j] / (float)n;

        uint32_t best = 0;
        float best_dist = 1e30f;
        for (uint32_t i = 0; i < n; i++) {
            float dist = 0.0f;
            for (uint32_t j = 0; j < d; j++)
                dist += (data[i*d+j] - centroid[j]) * (data[i*d+j] - centroid[j]);
            if (dist < best_dist) { best_dist = dist; best = i; }
        }
        return best;
    }
};

}  // namespace knowhere::pipnn
```

### Step 3: Create PiPNNDiskANNConfig

```cpp
// src/index/diskann/pipnn_diskann_config.h
#pragma once
#include "index/diskann/diskann_config.h"

namespace knowhere {

class PiPNNDiskANNConfig : public DiskANNConfig {
 public:
    // RBC partitioning
    CFG_INT(pipnn_leaf_max_size, "pipnn_leaf_max_size", 1000, 100, 10000, true);
    CFG_INT(pipnn_fanout_l1,     "pipnn_fanout_l1",     10,   1,  50,    true);
    CFG_INT(pipnn_fanout_l2,     "pipnn_fanout_l2",     3,    1,  20,    true);
    // HashPrune
    CFG_INT(pipnn_k_nn,          "pipnn_k_nn",          3,    1,  10,    true);
    CFG_INT(pipnn_hash_bits,     "pipnn_hash_bits",     12,   6,  16,    true);
    // Final RobustPrune
    CFG_BOOL(pipnn_final_prune,  "pipnn_final_prune",   true,         true);

    KNOHWERE_DECLARE_CONFIG(PiPNNDiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_leaf_max_size);
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_fanout_l1);
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_fanout_l2);
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_k_nn);
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_hash_bits);
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_final_prune);
    }
};

}  // namespace knowhere
```

**Note:** Look at `diskann_config.h` for the exact `CFG_INT`, `CFG_BOOL`, and `KNOWHERE_CONFIG_DECLARE_FIELD` macro patterns before writing this — copy the exact macro names and signatures.

### Step 4: Run serializer test

```bash
make -j$(nproc) knowhere_tests && \
./Release/tests/ut/knowhere_tests "[serializer]" -v
```

### Step 5: Commit

```bash
git add src/index/diskann/impl/vamana_serializer.h \
        src/index/diskann/pipnn_diskann_config.h \
        tests/ut/test_pipnn_diskann.cc
git commit -m "feat: add Vamana graph serializer and PiPNNDiskANNConfig"
```

---

## Task 6: PiPNNDiskANNIndexNode (main integration)

**Files:**
- Create: `src/index/diskann/pipnn_diskann.cc`

### Step 1: Write the integration test (the full build+search test)

Add to `tests/ut/test_pipnn_diskann.cc`:

```cpp
#include "filemanager/impl/LocalFileManager.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/version.h"
#include "utils.h"  // GenTestVersionList, etc.

#include <filesystem>
namespace fs = std::filesystem;

namespace {
std::string kPiPNNDir = fs::current_path().string() + "/pipnn_test";
std::string kRawDataPath = kPiPNNDir + "/raw_data";
std::string kIndexPrefix = kPiPNNDir + "/index";
constexpr uint32_t kN = 2000, kDim = 64, kNQ = 20, kK = 10;
constexpr float kMinRecall = 0.7f;
}

TEST_CASE("PiPNN-DiskANN: build and search L2", "[pipnn][diskann][integration]") {
    fs::create_directories(kPiPNNDir);

    // Generate random dataset
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> data(kN * kDim), queries(kNQ * kDim);
    for (auto& x : data)    x = dist(rng);
    for (auto& x : queries) x = dist(rng);

    // Write raw data in DiskANN .fbin format (uint32 n, uint32 d, then float data)
    {
        std::ofstream f(kRawDataPath, std::ios::binary);
        uint32_t n = kN, d = kDim;
        f.write((char*)&n, 4); f.write((char*)&d, 4);
        f.write((char*)data.data(), kN * kDim * sizeof(float));
    }

    auto file_manager = std::make_shared<milvus::LocalFileManager>(kPiPNNDir);
    auto version_list = GenTestVersionList();
    auto index_pack = knowhere::Pack(file_manager);

    auto index = knowhere::IndexFactory::Instance()
        .Create<knowhere::fp32>("PIPNN_DISKANN", version_list[0], index_pack)
        .value();

    // Build
    knowhere::Json build_cfg;
    build_cfg["dim"]                  = kDim;
    build_cfg["metric_type"]          = "L2";
    build_cfg["index_prefix"]         = kIndexPrefix;
    build_cfg["data_path"]            = kRawDataPath;
    build_cfg["max_degree"]           = 32;
    build_cfg["search_list_size"]     = 64;
    build_cfg["pq_code_budget_gb"]    = 0.002f;
    build_cfg["build_dram_budget_gb"] = 0.1f;
    build_cfg["pipnn_leaf_max_size"]  = 200;
    build_cfg["pipnn_fanout_l1"]      = 5;
    build_cfg["pipnn_fanout_l2"]      = 2;
    build_cfg["pipnn_k_nn"]           = 3;
    build_cfg["pipnn_hash_bits"]      = 12;
    build_cfg["pipnn_final_prune"]    = true;

    auto dataset = knowhere::GenDataSet(kN, kDim, data.data());
    auto status = index.Build(dataset, build_cfg);
    REQUIRE(status == knowhere::Status::success);

    // Deserialize (load from disk)
    auto index2 = knowhere::IndexFactory::Instance()
        .Create<knowhere::fp32>("PIPNN_DISKANN", version_list[0], index_pack)
        .value();
    knowhere::Json load_cfg = build_cfg;
    load_cfg["search_cache_budget_gb"] = 0.01f;
    load_cfg["num_nodes_to_cache"] = 0;
    auto deser_status = index2.Deserialize(knowhere::BinarySet{},
                                           std::make_shared<knowhere::Json>(load_cfg));
    REQUIRE(deser_status == knowhere::Status::success);

    // Brute-force ground truth
    auto query_ds = knowhere::GenDataSet(kNQ, kDim, queries.data());
    auto base_ds  = knowhere::GenDataSet(kN, kDim, data.data());
    knowhere::Json bf_cfg;
    bf_cfg["metric_type"] = "L2";
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds,
                                                          bf_cfg, nullptr);

    // ANN search
    knowhere::Json search_cfg = load_cfg;
    search_cfg["k"] = kK;
    search_cfg["beamwidth"] = 4;
    search_cfg["search_list_size"] = 64;
    auto result = index2.Search(query_ds, search_cfg, nullptr);
    REQUIRE(result.has_value());

    // Compute recall
    auto gt_ids     = gt.value()->GetIds();
    auto result_ids = result.value()->GetIds();
    float recall = CalcRecall(gt_ids, result_ids, kNQ, kK);
    INFO("PiPNN-DiskANN L2 recall@" << kK << " = " << recall);
    REQUIRE(recall >= kMinRecall);

    fs::remove_all(kPiPNNDir);
}
```

### Step 2: Implement PiPNNDiskANNIndexNode

Create `src/index/diskann/pipnn_diskann.cc`. Start by reading `src/index/diskann/diskann.cc` in full to understand the complete DiskANNIndexNode pattern, then implement:

```cpp
// src/index/diskann/pipnn_diskann.cc
// Copyright (C) 2019-2023 Zilliz. All rights reserved.
// Apache 2.0 License

#ifdef KNOWHERE_WITH_PIPNN
#ifdef KNOWHERE_WITH_DISKANN

#include "diskann/aux_utils.h"
#include "diskann/utils.h"
#include "index/diskann/diskann_config.h"
#include "index/diskann/pipnn_diskann_config.h"
#include "index/diskann/impl/pipnn_builder.h"
#include "index/diskann/impl/vamana_serializer.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
// ... (same includes as diskann.cc)

namespace knowhere {

template <typename DataType>
class PiPNNDiskANNIndexNode : public IndexNode {
    // Copy all member variables and non-Build methods from DiskANNIndexNode.
    // Only Build() is different.

 public:
    // [copy constructor, Search, Deserialize, etc. from DiskANNIndexNode]

    Status Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                 bool use_knowhere_build_pool) override {
        auto build_conf = static_cast<const PiPNNDiskANNConfig&>(*cfg);

        // === Same validation as DiskANNIndexNode::Build() ===
        // CheckMetric, check index_prefix, check data_path, LoadFile, etc.
        // Copy lines 410-436 from diskann.cc, replacing DiskANNConfig with
        // PiPNNDiskANNConfig.

        // === Stage 1: PiPNN graph construction ===
        // 1a. Read raw vectors from data_path (already written as .fbin)
        size_t n, d;
        diskann::get_bin_metadata(build_conf.data_path.value(), n, d);

        std::vector<DataType> raw_data(n * d);
        diskann::load_bin<DataType>(
            build_conf.data_path.value(), raw_data, n, d);

        // Convert to float32 for PiPNN (Eigen works in float)
        std::vector<float> float_data(n * d);
        for (size_t i = 0; i < n * d; i++)
            float_data[i] = static_cast<float>(raw_data[i]);

        // 1b. Configure and run PiPNN builder
        pipnn::PiPNNBuilder::Config pipnn_cfg;
        pipnn_cfg.k_nn        = build_conf.pipnn_k_nn.value();
        pipnn_cfg.hash_bits   = build_conf.pipnn_hash_bits.value();
        pipnn_cfg.max_degree  = build_conf.max_degree.value();
        pipnn_cfg.final_prune = build_conf.pipnn_final_prune.value();
        pipnn_cfg.num_threads = 0;

        pipnn::RBCPartitioner::Config rbc_cfg;
        rbc_cfg.leaf_max_size = build_conf.pipnn_leaf_max_size.value();
        rbc_cfg.fanout_l1     = build_conf.pipnn_fanout_l1.value();
        rbc_cfg.fanout_l2     = build_conf.pipnn_fanout_l2.value();

        pipnn::PiPNNBuilder builder;
        auto graph = builder.build(float_data.data(), n, d, pipnn_cfg, rbc_cfg);

        // 1c. Find medoid as entry point
        uint32_t entry_point = pipnn::VamanaSerializer::find_medoid(
            float_data.data(), n, d);

        // 1d. Write graph in Vamana format → _mem.index
        std::string mem_index_path = build_conf.index_prefix.value() + "_mem.index";
        pipnn::VamanaSerializer::write(graph, entry_point, mem_index_path);

        // === Stage 2: DiskANN PQ + disk layout ===
        // Construct a BuildConfig that points to our pre-built graph.
        // We use build_disk_index but override so it skips Vamana build.
        // Approach: set accelerate_build=true and provide _mem.index directly
        // by calling the DiskANN pipeline starting AFTER graph build.
        //
        // Practically: call diskann::build_disk_index with the normal config.
        // DiskANN will try to build the graph again — BUT we pre-write _mem.index
        // and DiskANN skips graph build if _mem.index already exists.
        //
        // VERIFY this by checking aux_utils.cpp:1677+ for the skip condition.
        // If DiskANN does NOT skip, call generate_pq_data_from_pivots and
        // create_disk_layout directly:

        auto diskann_metric = /* same metric conversion as diskann.cc:440-448 */;
        diskann::BuildConfig internal_cfg{
            build_conf.data_path.value(),
            build_conf.index_prefix.value(),
            diskann_metric,
            (unsigned)build_conf.max_degree.value(),
            (unsigned)build_conf.search_list_size.value(),
            (double)build_conf.pq_code_budget_gb.value(),
            (double)build_conf.build_dram_budget_gb.value(),
            (uint32_t)build_conf.disk_pq_dims.value(),
            false, false,
            (uint32_t)GetCachedNodeNum(...),
            false};

        // Call build_disk_index — it will skip Vamana if _mem.index exists,
        // then run PQ + create_disk_layout.
        RETURN_IF_ERROR(TryDiskANNCall([&]() {
            int res = diskann::build_disk_index<DataType>(internal_cfg);
            if (res != 0) throw diskann::ANNException("...", -1);
        }));

        // === Stage 3: Register files with FileManager ===
        // Same as diskann.cc:470-482
        // ...

        return Status::success;
    }

    // CreateConfig
    static std::unique_ptr<BaseConfig> StaticCreateConfig() {
        return std::make_unique<PiPNNDiskANNConfig>();
    }
    std::unique_ptr<BaseConfig> CreateConfig() const override {
        return StaticCreateConfig();
    }
};

// Registration
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(
    PIPNN_DISKANN, PiPNNDiskANNIndexNode,
    knowhere::feature::DISKANN)

}  // namespace knowhere
#endif  // KNOWHERE_WITH_DISKANN
#endif  // KNOWHERE_WITH_PIPNN
```

**Critical implementation detail — check if DiskANN skips graph build:**

Read `thirdparty/DiskANN/src/aux_utils.cpp:1751-1800` to check if `build_disk_index` skips Vamana when `_mem.index` exists. If it does NOT, call these functions directly instead:

```cpp
// Alternative Stage 2 if build_disk_index doesn't skip:
diskann::generate_pq_pivots<float>(...);
diskann::generate_quantized_data<float>(...);
diskann::create_disk_layout<DataType>(...);
```

### Step 3: Add PIPNN_DISKANN to index_table.h

Check `include/knowhere/index/index_table.h` — find where `DISKANN` is defined and add `PIPNN_DISKANN` in the same pattern.

### Step 4: Build and run integration test

```bash
cmake -DWITH_UT=ON -DWITH_DISKANN=ON -DWITH_PIPNN=ON \
      -DCMAKE_BUILD_TYPE=Release .. && \
make -j$(nproc) knowhere_tests 2>&1 | tail -30

./Release/tests/ut/knowhere_tests "[pipnn][integration]" -v
```

Expected: Build succeeds, recall >= 0.7.

### Step 5: Commit

```bash
git add src/index/diskann/pipnn_diskann.cc \
        include/knowhere/index/index_table.h \
        tests/ut/test_pipnn_diskann.cc
git commit -m "feat: implement PiPNNDiskANNIndexNode with full build pipeline"
```

---

## Task 7: Recall Comparison vs DiskANN

**Files:**
- Modify: `tests/ut/test_pipnn_diskann.cc`

### Step 1: Add comparison benchmark test

```cpp
TEST_CASE("PiPNN-DiskANN vs DiskANN: recall comparison", "[pipnn][benchmark]") {
    // Build both indexes on same data, compare recall@10
    // PiPNN recall should be within 5% of DiskANN recall
    // (single replica ≈ Vamana single pass per paper)
    constexpr float kRecallTolerance = 0.05f;

    // [Setup same as integration test but build both DISKANN and PIPNN_DISKANN]
    // [Log build times for both]
    // [Assert |pipnn_recall - diskann_recall| <= kRecallTolerance]
}
```

### Step 2: Run and document results

```bash
./Release/tests/ut/knowhere_tests "[benchmark]" -v 2>&1 | tee pipnn_benchmark.txt
```

### Step 3: Commit results

```bash
git add tests/ut/test_pipnn_diskann.cc
git commit -m "test: add PiPNN vs DiskANN recall comparison benchmark"
```

---

## Build Commands Reference

```bash
# Full build with PiPNN (Linux container)
conan install .. --build=missing -o with_ut=True -o with_diskann=True \
  -s compiler.libcxx=libstdc++11 -s build_type=Release
cmake -DWITH_UT=ON -DWITH_DISKANN=ON -DWITH_PIPNN=ON \
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Run all PiPNN tests
./Release/tests/ut/knowhere_tests "[pipnn]" -v

# Run specific test
./Release/tests/ut/knowhere_tests "[hash_prune]" -v
./Release/tests/ut/knowhere_tests "[rbc]" -v
./Release/tests/ut/knowhere_tests "[builder]" -v
./Release/tests/ut/knowhere_tests "[integration]" -v
```

---

## Key Things to Verify Before Each Task

1. **Task 2 (HashPrune):** Hyperplane initialization must use a fixed seed so sketches are reproducible — same `HashPrune(dim, m, max_degree)` instance gives same hashes.

2. **Task 3 (RBC):** The recursive partitioner must terminate — add assertion that `n_leaders < point_ids.size()` to prevent infinite recursion on degenerate cases.

3. **Task 4 (Builder):** The `std::mutex` per point is expensive for large N — consider using `std::atomic` spinlocks or chunked locking if performance becomes a bottleneck.

4. **Task 5 (Serializer):** The index_size field in the Vamana format must be written **twice** (once at start with initial value, once at end with final value). See `index.cpp:411-413`.

5. **Task 6 (IndexNode):** Check `thirdparty/DiskANN/src/aux_utils.cpp` lines 1800-1870 carefully for whether `build_disk_index` has a code path that skips Vamana when `_mem.index` already exists. If not, call PQ/layout functions directly.
