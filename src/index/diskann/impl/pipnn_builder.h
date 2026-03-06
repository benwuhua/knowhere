// Copyright (C) 2019-2023 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

#include "index/diskann/impl/hash_prune.h"
#include "index/diskann/impl/rbc_partition.h"

namespace knowhere::pipnn_diskann {

class PiPNNBuilder {
 public:
    struct Config {
        uint32_t k_nn = 64;
        uint32_t hash_bits = 12;
        uint32_t max_degree = 32;
        float alpha = 1.2f;
        bool final_prune = true;
        uint32_t num_threads = 0;
        size_t leaf_max_size = 0;
        uint32_t fanout_l1 = 8;
        uint32_t fanout_l2 = 4;
        uint32_t fanout_rest = 2;
        uint32_t overlap_k = 2;
        uint64_t base_seed = 42;
    };

    PiPNNBuilder();
    explicit PiPNNBuilder(const Config& config);

    std::vector<std::vector<uint32_t>>
    build(const float* data, uint32_t n, uint32_t dim) const;

 private:
    struct BuildContext {
        explicit BuildContext(uint32_t n)
            : adjacency(n),
              node_mutexes(n),
              leaf_total_ns(0),
              gemm_total_ns(0),
              hash_prune_total_ns(0),
              edge_insert_ns(0),
              edge_insert_count(0) {
        }

        std::vector<std::vector<uint32_t>> adjacency;
        std::vector<std::mutex> node_mutexes;
        std::atomic<int64_t> leaf_total_ns;
        std::atomic<int64_t> gemm_total_ns;
        std::atomic<int64_t> hash_prune_total_ns;
        std::atomic<int64_t> edge_insert_ns;
        std::atomic<int64_t> edge_insert_count;
    };

    void
    process_leaf(const float* data, uint32_t dim, const Leaf& leaf, BuildContext& context) const;

    void
    robust_prune_pass(const float* data, uint32_t n, uint32_t dim, std::vector<std::vector<uint32_t>>& adjacency) const;

    static float
    l2sq(const float* a, const float* b, uint32_t dim);

 private:
    Config config_;
};

}  // namespace knowhere::pipnn_diskann
