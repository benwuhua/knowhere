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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace knowhere::pipnn_diskann {

struct Leaf {
    std::vector<uint32_t> point_ids;
};

class RBCPartitioner {
public:
    struct Config {
        size_t leaf_max_size = 1024;
        uint32_t fanout_l1 = 8;
        uint32_t fanout_l2 = 4;
        uint32_t fanout_rest = 2;
        uint32_t overlap_k = 2;
        uint32_t num_leaders = 0;
        uint32_t base_seed = 42;
    };

    RBCPartitioner();
    explicit RBCPartitioner(const Config& config);

    std::vector<Leaf>
    partition(const float* data, uint32_t n, uint32_t dim) const;

private:
    void
    partition_recursive(const float* data, uint32_t dim, const std::vector<uint32_t>& point_ids, size_t depth,
                        std::vector<Leaf>& leaves) const;

    std::vector<uint32_t>
    sample_leaders(const std::vector<uint32_t>& point_ids, uint32_t num_leaders, uint32_t seed) const;

    std::vector<std::vector<uint32_t>>
    assign_to_k_leaders(const float* data, uint32_t dim, const std::vector<uint32_t>& point_ids,
                        const std::vector<uint32_t>& leaders, uint32_t k) const;

    std::vector<std::vector<uint32_t>>
    split_evenly(const std::vector<uint32_t>& point_ids, uint32_t fanout) const;

    uint32_t
    fanout_for_depth(size_t depth) const;

    uint32_t
    effective_num_leaders(size_t subset_size) const;

    static float
    l2sq(const float* a, const float* b, uint32_t dim);

private:
    Config config_;
};

}  // namespace knowhere::pipnn_diskann
