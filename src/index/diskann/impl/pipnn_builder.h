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

#include <cstdint>
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
    };

    PiPNNBuilder();
    explicit PiPNNBuilder(const Config& config);

    std::vector<std::vector<uint32_t>>
    build(const float* data, uint32_t n, uint32_t dim) const;

 private:
    struct BuildContext;

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
