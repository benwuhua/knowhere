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
#include <string>

namespace knowhere::pipnn_diskann {

// PiPNN graph construction parameters
struct PipnnConfig {
    // RBC (Recursive Bisection Clustering) parameters
    uint32_t num_partitions = 32;      // Number of overlapping subsets
    float overlap_ratio = 0.5f;          // Fraction of points shared between adjacent partitions

    // HashPrune parameters
    uint32_t max_degree = 32;              // Max neighbors per node
    uint32_t hash_bits = 12;               // Hash table size (2^12 = 4096)

    // Graph construction
    bool use_medoid_entry = true;          // Use medoid as entry point instead of random
    uint32_t max_build_threads = 0;          // 0 = auto (std::thread::hardware_concurrency)

    std::string
    ToString() const {
        return "PipnnConfig{partitions=" + std::to_string(num_partitions) +
               ", overlap=" + std::to_string(overlap_ratio) +
               ", max_degree=" + std::to_string(max_degree) +
               ", hash_bits=" + std::to_string(hash_bits) + "}";
    }
};

// Default configuration
static constexpr PipnnConfig kDefaultPipnnConfig = {
    .num_partitions = 32,
    .overlap_ratio = 0.5f,
    .max_degree = 32,
    .hash_bits = 12,
    .use_medoid_entry = true,
    .max_build_threads = 0,
};

}  // namespace knowhere::pipnn_diskann
