// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "index/diskann/impl/hash_prune.h"
#include "index/diskann/impl/pipnn_diskann_config.h"
#include "index/diskann/impl/rbc_partition.h"

namespace {

using knowhere::pipnn_diskann::HashPrune;
using knowhere::pipnn_diskann::PipnnConfig;
using knowhere::pipnn_diskann::RBCPartitioner;

constexpr uint32_t kDim = 32;
constexpr uint32_t kHashBits = 12;
constexpr uint32_t kMaxDegree = 16;

std::vector<float>
RandVec(uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

float
L2Dist(const float* a, const float* b, uint32_t d) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

}  // namespace

TEST_CASE("PipnnConfig defaults", "[pipnn_diskann]") {
    PipnnConfig config;
    REQUIRE(config.num_partitions == 32);
    REQUIRE(config.overlap_ratio == Catch::Approx(0.5f));
    REQUIRE(config.max_degree == 32);
    REQUIRE(config.hash_bits == 12);
    REQUIRE(config.use_medoid_entry);
}

TEST_CASE("PipnnConfig ToString", "[pipnn_diskann]") {
    PipnnConfig config;
    const std::string s = config.ToString();
    REQUIRE(s.find("partitions=32") != std::string::npos);
    REQUIRE(s.find("overlap=0.500000") != std::string::npos);
    REQUIRE(s.find("max_degree=32") != std::string::npos);
}

TEST_CASE("HashPrune enforces max_degree", "[pipnn][hash_prune]") {
    std::mt19937 rng(42);
    auto p = RandVec(kDim, rng);

    HashPrune prune(kDim, kHashBits, kMaxDegree);
    std::vector<float> sketch_p(kHashBits);
    prune.compute_sketch(p.data(), sketch_p.data());

    for (uint32_t i = 0; i < 128; ++i) {
        auto c = RandVec(kDim, rng);
        std::vector<float> sketch_c(kHashBits);
        prune.compute_sketch(c.data(), sketch_c.data());
        prune.insert(i, sketch_p.data(), sketch_c.data(), L2Dist(p.data(), c.data(), kDim));
    }

    REQUIRE(prune.size() <= kMaxDegree);
    REQUIRE(prune.neighbors().size() == prune.size());
}

TEST_CASE("HashPrune insertion order robustness", "[pipnn][hash_prune]") {
    std::mt19937 rng(123);
    auto p = RandVec(kDim, rng);

    std::vector<std::vector<float>> candidates(80);
    for (auto& c : candidates) {
        c = RandVec(kDim, rng);
    }

    auto run_once = [&](bool reverse) {
        HashPrune prune(kDim, kHashBits, kMaxDegree);
        std::vector<float> sketch_p(kHashBits);
        prune.compute_sketch(p.data(), sketch_p.data());

        if (reverse) {
            for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i) {
                std::vector<float> sketch_c(kHashBits);
                prune.compute_sketch(candidates[i].data(), sketch_c.data());
                prune.insert(static_cast<uint32_t>(i),
                             sketch_p.data(),
                             sketch_c.data(),
                             L2Dist(p.data(), candidates[i].data(), kDim));
            }
        } else {
            for (uint32_t i = 0; i < candidates.size(); ++i) {
                std::vector<float> sketch_c(kHashBits);
                prune.compute_sketch(candidates[i].data(), sketch_c.data());
                prune.insert(i,
                             sketch_p.data(),
                             sketch_c.data(),
                             L2Dist(p.data(), candidates[i].data(), kDim));
            }
        }
        auto ids = prune.neighbors();
        std::sort(ids.begin(), ids.end());
        return ids;
    };

    const auto forward = run_once(false);
    const auto backward = run_once(true);

    REQUIRE(forward.size() <= kMaxDegree);
    REQUIRE(backward.size() <= kMaxDegree);
    // The two orders do not have to be identical, but they should have overlap.
    size_t overlap = 0;
    for (auto id : forward) {
        overlap += std::binary_search(backward.begin(), backward.end(), id) ? 1 : 0;
    }
    REQUIRE(overlap > 0);
}

TEST_CASE("RBC: every point appears in at least one leaf", "[pipnn][rbc]") {
    constexpr uint32_t n = 256;
    constexpr uint32_t dim = 16;

    std::mt19937 rng(2024);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 24;
    config.fanout_l1 = 6;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 17;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> counts(n, 0);
    for (const auto& leaf : leaves) {
        for (auto id : leaf.point_ids) {
            REQUIRE(id < n);
            counts[id] += 1;
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        REQUIRE(counts[i] >= 1);
    }
}

TEST_CASE("RBC: all leaf sizes <= leaf_max_size", "[pipnn][rbc]") {
    constexpr uint32_t n = 640;
    constexpr uint32_t dim = 24;

    std::mt19937 rng(101);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 32;
    config.fanout_l1 = 8;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 29;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());
    for (const auto& leaf : leaves) {
        REQUIRE(leaf.point_ids.size() <= config.leaf_max_size);
    }
}

TEST_CASE("RBC: overlap factor is reasonable", "[pipnn][rbc]") {
    constexpr uint32_t n = 512;
    constexpr uint32_t dim = 20;

    std::mt19937 rng(99);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 40;
    config.fanout_l1 = 8;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 7;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> multiplicity(n, 0);
    for (const auto& leaf : leaves) {
        for (auto id : leaf.point_ids) {
            multiplicity[id] += 1;
        }
    }

    const uint64_t total_assignments = std::accumulate(multiplicity.begin(), multiplicity.end(), uint64_t{0});
    const float overlap_factor = static_cast<float>(total_assignments) / static_cast<float>(n);

    REQUIRE(overlap_factor >= 2.0f);
    REQUIRE(overlap_factor <= 8.0f);
}

TEST_CASE("RBC: handles oversized num_leaders and still terminates", "[pipnn][rbc]") {
    constexpr uint32_t n = 9;
    constexpr uint32_t dim = 8;

    std::mt19937 rng(303);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 1;
    config.fanout_l1 = 4;
    config.fanout_l2 = 3;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.num_leaders = 1024;
    config.base_seed = 17;

    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> counts(n, 0);
    for (const auto& leaf : leaves) {
        REQUIRE(leaf.point_ids.size() <= config.leaf_max_size);
        for (auto id : leaf.point_ids) {
            REQUIRE(id < n);
            counts[id] += 1;
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        REQUIRE(counts[i] >= 1);
    }
}
