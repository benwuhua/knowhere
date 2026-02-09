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

#ifndef PAGEANN_H
#define PAGEANN_H

#ifdef KNOWHERE_WITH_PAGEANN

#include <memory>
#include "knowhere/index/index_node.h"
#include "diskann/prefetch_buffer.h"
#include "diskann/frequency_aware_cache.h"

namespace knowhere {

/**
 * PageANNIndexNode - Enhanced DiskANN with Stage 1 optimizations
 *
 * This index extends DiskANNIndexNode with:
 * - Batch prefetch for reduced I/O latency
 * - Frequency-aware cache (LFU policy)
 * - Enhanced concurrent I/O
 *
 * Design approach:
 * - Extends DiskANNIndexNode without modifying it
 * - Adds optimization layers at the Knowhere level
 * - Uses same disk format as DiskANN for compatibility
 * - Can be toggled via config flags
 */
template <typename DataType>
class PageANNIndexNode : public DiskANNIndexNode<DataType> {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "PageANN only support floating point data type(float32, float16, bfloat16)");

 public:
    using DistType = float;

    /**
     * Constructor
     * @param version Index version
     * @param object Pack containing FileManager
     */
    PageANNIndexNode(const int32_t& version, const Object& object);

    /**
     * Destructor
     */
    ~PageANNIndexNode() override;

    /**
     * Deserialize index from binary set and initialize PageANN optimizations
     *
     * @param binset Binary set containing serialized index
     * @param cfg Configuration (PageANNConfig)
     * @return Status
     *
     * This method:
     * 1. Calls parent DiskANNIndexNode::Deserialize() to load base index
     * 2. Initializes PrefetchBuffer if enable_prefetch is true
     * 3. Initializes FrequencyAwareCache if enable_frequency_aware_cache is true
     */
    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    /**
     * Search with PageANN optimizations
     *
     * @param dataset Query vectors
     * @param cfg Search configuration
     * @param bitset Bitset for filtering
     * @param op_context Operation context
     * @return Result dataset
     *
     * Note: Initial implementation delegates to parent DiskANNIndexNode::Search()
     * Future versions will inject custom prefetch logic here
     */
    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

 private:
    // PageANN optimization components

    // Prefetch buffer for async node data prefetching
    mutable std::unique_ptr<diskann::PrefetchBuffer<DataType>> prefetch_buffer_;

    // Frequency-aware cache with LFU eviction
    mutable std::unique_ptr<diskann::FrequencyAwareCache<DataType>> freq_aware_cache_;

    // Configuration flags (from PageANNConfig)

    // Enable batch prefetch optimization
    bool enable_prefetch_;

    // Enable frequency-aware cache
    bool enable_freq_aware_cache_;

    // Number of nodes to prefetch per beam iteration
    uint32_t prefetch_batch_size_;

    // Prefetch lookahead ratio (multiplier for beam_width)
    float prefetch_lookahead_ratio_;

    // Frequency cache memory budget in GB
    float frequency_cache_budget_gb_;

    // Query count between cache decay operations
    uint32_t cache_decay_interval_;

    // Cache frequency decay factor
    float cache_decay_factor_;

    // Initialization state
    bool optimizations_initialized_;

    /**
     * Initialize PageANN optimizations from config
     *
     * @param cfg PageANNConfig
     * @return Status
     */
    Status initialize_optimizations(std::shared_ptr<Config> cfg);

    /**
     * Cleanup PageANN optimization resources
     */
    void cleanup_optimizations();
};

}  // namespace knowhere

#endif  // KNOWHERE_WITH_PAGEANN

#endif /* PAGEANN_H */
