// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef PAGEANN_CONFIG_H
#define PAGEANN_CONFIG_H

#ifdef KNOWHERE_WITH_PAGEANN

#include "diskann_config.h"

namespace knowhere {

// PageANN Config - aligned with the Paper "Scalable Disk-Based ANN Search with Page-Aligned Graph"
// PageANN is a page-level graph index with page_search, page caching, and hash routing.
// It is NOT a node-level DiskANN optimization; it uses a completely different index format.
class PageANNConfig : public DiskANNConfig {
 public:
    // --- Build parameters (for generate_page_graph) ---

    // Minimum degree per node in the page graph
    CFG_INT min_degree_per_node;

    // Number of PQ chunks for compression (typically 12-32)
    CFG_INT num_pq_chunks;

    // Maximum Vamana graph degree (R)
    CFG_INT max_vamana_degree;

    // Memory budget in GB for page graph generation
    CFG_FLOAT page_build_mem_budget_gb;

    // Enable fully out-of-core page graph construction
    CFG_BOOL full_ooc;

    // --- Search parameters ---

    // Enable hash-based routing for entry point selection (full hash buckets)
    CFG_BOOL use_hash_routing;

    // Enable sampled hash-based routing (lower memory, slightly lower recall)
    CFG_BOOL use_sampled_hash_routing;

    // Hamming radius for hash bucket expansion (0=exact, 1=recommended, 2=wider)
    CFG_INT hash_radius;

    // Number of pages to cache in memory (0 = no caching)
    CFG_INT num_pages_to_cache;

    // PQ path prefix (for using different PQ compression than what was built)
    CFG_STRING pq_path_prefix;

    KNOWHERE_DECLARE_CONFIG(PageANNConfig) {
        // Build parameters
        KNOWHERE_CONFIG_DECLARE_FIELD(min_degree_per_node)
            .description("minimum degree per node in page graph")
            .set_default(23)
            .set_range(1, 256)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(num_pq_chunks)
            .description("number of PQ chunks for compression")
            .set_default(20)
            .set_range(4, 128)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(max_vamana_degree)
            .description("maximum Vamana graph degree (R)")
            .set_default(25)
            .set_range(4, 256)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(page_build_mem_budget_gb)
            .description("memory budget in GB for page graph generation")
            .set_default(4.0f)
            .set_range(0.5f, 256.0f)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(full_ooc)
            .description("enable fully out-of-core page graph construction")
            .set_default(false)
            .for_train();

        // Search parameters
        KNOWHERE_CONFIG_DECLARE_FIELD(use_hash_routing)
            .description("enable hash-based routing for entry point selection")
            .set_default(false)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(use_sampled_hash_routing)
            .description("enable sampled hash-based routing (lower memory)")
            .set_default(false)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(hash_radius)
            .description("Hamming radius for hash bucket expansion (0-2)")
            .set_default(1)
            .set_range(0, 2)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(num_pages_to_cache)
            .description("number of frequently-accessed pages to cache in memory")
            .set_default(0)
            .set_range(0, 10000000)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(pq_path_prefix)
            .description("optional PQ path prefix for different compression level")
            .set_default("")
            .for_deserialize();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        Status parent_status = DiskANNConfig::CheckAndAdjust(param_type, err_msg);
        if (parent_status != Status::success) {
            return parent_status;
        }

        switch (param_type) {
            case PARAM_TYPE::DESERIALIZE: {
                // Cannot enable both hash routing modes simultaneously
                if (use_hash_routing.has_value() && use_hash_routing.value() &&
                    use_sampled_hash_routing.has_value() && use_sampled_hash_routing.value()) {
                    std::string msg =
                        "use_hash_routing and use_sampled_hash_routing cannot both be true";
                    return HandleError(err_msg, msg, Status::invalid_args);
                }
                break;
            }
            default:
                break;
        }

        return Status::success;
    }
};

}  // namespace knowhere

#endif  // KNOWHERE_WITH_PAGEANN

#endif  // PAGEANN_CONFIG_H
