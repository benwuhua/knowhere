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

#include "diskann_config.h"

namespace knowhere {

class PageANNConfig : public DiskANNConfig {
 public:
    // PageANN Optimization Parameters

    // Number of nodes to prefetch per beam iteration
    // Larger values prefetch more aggressively but use more memory
    CFG_INT prefetch_batch_size;

    // Prefetch buffer size in megabytes
    // Larger buffers allow more aggressive prefetching
    CFG_INT prefetch_buffer_mb;

    // Prefetch lookahead as a ratio of beam_width
    // For example, 2.0 means prefetch 2 * beam_width candidates ahead
    CFG_FLOAT prefetch_lookahead_ratio;

    // Enable frequency-aware cache (LFU eviction policy)
    // When enabled, tracks node access frequency and evicts low-frequency nodes
    CFG_BOOL enable_frequency_aware_cache;

    // Memory budget for frequency-aware cache in GB
    // Separate from the base DiskANN node cache
    CFG_FLOAT frequency_cache_budget_gb;

    // Number of queries between cache frequency decay operations
    // Periodic decay prevents frequency overflow and adapts to workload changes
    CFG_INT cache_decay_interval;

    // Cache frequency decay factor
    // For example, 0.99 reduces all frequencies by 1% each decay interval
    CFG_FLOAT cache_decay_factor;

    KNOHWERE_DECLARE_CONFIG(PageANNConfig) {
        // Inherit all DiskANNConfig fields
        DiskANNConfig::static_get();

        // PageANN-specific fields

        KNOWHERE_CONFIG_DECLARE_FIELD(prefetch_batch_size)
            .description("number of nodes to prefetch per beam iteration")
            .set_default(16)
            .set_range(1, 128)
            .for_search();

        KNOWHERE_CONFIG_DECLARE_FIELD(prefetch_buffer_mb)
            .description("prefetch buffer size in megabytes")
            .set_default(256)
            .set_range(64, 2048)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(prefetch_lookahead_ratio)
            .description("prefetch lookahead as ratio of beam_width")
            .set_default(2.0f)
            .set_range(1.0f, 4.0f)
            .for_search();

        KNOWHERE_CONFIG_DECLARE_FIELD(enable_frequency_aware_cache)
            .description("enable frequency-aware cache replacement")
            .set_default(true)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(frequency_cache_budget_gb)
            .description("memory budget for frequency-aware cache in GB")
            .set_default(0.1f)
            .set_range(0.01f, 10.0f)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(cache_decay_interval)
            .description("number of queries between cache frequency decay")
            .set_default(10000)
            .set_range(1000, 100000)
            .for_deserialize();

        KNOWHERE_CONFIG_DECLARE_FIELD(cache_decay_factor)
            .description("decay factor for cache frequency aging")
            .set_default(0.99f)
            .set_range(0.9f, 1.0f)
            .for_deserialize();
    }

    Status
    CheckAndAdjust(PARAM_TYPE param_type, std::string* err_msg) override {
        // First, check parent DiskANN config
        Status parent_status = DiskANNConfig::CheckAndAdjust(param_type, err_msg);
        if (parent_status != Status::success) {
            return parent_status;
        }

        // PageANN-specific validation
        switch (param_type) {
            case PARAM_TYPE::DESERIALIZE: {
                // Validate prefetch_buffer_mb is reasonable
                if (prefetch_buffer_mb.has_value()) {
                    if (prefetch_buffer_mb.value() < 64 || prefetch_buffer_mb.value() > 2048) {
                        std::string msg = "prefetch_buffer_mb must be between 64 and 2048 MB, got: " +
                                       std::to_string(prefetch_buffer_mb.value());
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                }

                // Validate frequency_cache_budget_gb is reasonable
                if (frequency_cache_budget_gb.has_value()) {
                    if (frequency_cache_budget_gb.value() < 0.01f ||
                        frequency_cache_budget_gb.value() > 10.0f) {
                        std::string msg = "frequency_cache_budget_gb must be between 0.01 and 10.0 GB, got: " +
                                       std::to_string(frequency_cache_budget_gb.value());
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                }

                // Validate cache_decay_factor is in valid range
                if (cache_decay_factor.has_value()) {
                    if (cache_decay_factor.value() < 0.9f || cache_decay_factor.value() > 1.0f) {
                        std::string msg = "cache_decay_factor must be between 0.9 and 1.0, got: " +
                                       std::to_string(cache_decay_factor.value());
                        return HandleError(err_msg, msg, Status::out_of_range_in_json);
                    }
                }

                break;
            }
            case PARAM_TYPE::SEARCH: {
                // Validate prefetch_batch_size against beamwidth
                if (prefetch_batch_size.has_value() && beamwidth.has_value()) {
                    // prefetch_batch_size should be reasonable relative to beamwidth
                    // Warn but don't fail if it's too large
                    if (prefetch_batch_size.value() > beamwidth.value() * 4) {
                        LOG_KNOWHERE_WARNING_ << "prefetch_batch_size (" << prefetch_batch_size.value()
                                             << ") is much larger than beamwidth (" << beamwidth.value()
                                             << "), may waste memory";
                    }
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
#endif /* PAGEANN_CONFIG_H */
