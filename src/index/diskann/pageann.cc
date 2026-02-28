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

// PageANN Index Node for Knowhere
// Based on the Paper: "Scalable Disk-Based ANN Search with Page-Aligned Graph"
// This wraps the Paper's pageann::PQFlashIndex which implements page-level search.

#ifdef KNOWHERE_WITH_PAGEANN

// PageANN thirdparty (namespace pageann, renamed from diskann to avoid conflicts)
#include "pq_flash_index.h"
#include "linux_aligned_file_reader.h"
#include "disk_utils.h"

// Knowhere headers
#include "index/diskann/pageann_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "knowhere/thread_pool.h"
#include "filemanager/FileManager.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace knowhere {

// =============================================================================
// PageANNIndexNode - wraps Paper PageANN's PQFlashIndex with page_search
//
// This is fundamentally different from DiskANNIndexNode:
// - DiskANN: node-level graph, each disk read fetches one node
// - PageANN: page-level graph, each disk read fetches an entire page of vectors
// - PageANN uses page_search/linux_page_search instead of cached_beam_search
// - PageANN supports hash-based routing and page caching
// =============================================================================
template <typename DataType>
class PageANNIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "PageANN only supports floating point data types (float32, float16, bfloat16)");

 public:
    using DistType = float;

    PageANNIndexNode(const int32_t& version, const Object& object)
        : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>));
        auto pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
        assert(pack != nullptr);
        file_manager_ = pack->GetPack();
    }

    // =========================================================================
    // Build: Vamana build + generate_page_graph (two-stage pipeline)
    // =========================================================================
    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        auto& conf = static_cast<const PageANNConfig&>(*cfg);

        std::string index_prefix = conf.index_prefix.value();
        std::string data_path = conf.data_path.value();
        uint32_t max_degree = static_cast<uint32_t>(conf.max_vamana_degree.value_or(25));
        uint32_t build_list_size = static_cast<uint32_t>(conf.search_list_size.value_or(128));
        float build_dram_budget = conf.build_dram_budget_gb.value_or(4.0f);
        float search_dram_budget = conf.pq_code_budget_gb.value_or(0.1f);
        uint32_t min_degree = static_cast<uint32_t>(conf.min_degree_per_node.value_or(23));
        uint32_t num_pq_chunks = static_cast<uint32_t>(conf.num_pq_chunks.value_or(20));
        float page_build_mem = conf.page_build_mem_budget_gb.value_or(4.0f);
        bool full_ooc = conf.full_ooc.value_or(false);

        std::string metric_str = conf.metric_type.value();
        pageann::Metric metric = pageann::Metric::L2;
        if (metric_str == "IP" || metric_str == "INNER_PRODUCT") {
            metric = pageann::Metric::INNER_PRODUCT;
        } else if (metric_str == "COSINE") {
            metric = pageann::Metric::COSINE;
        }

        // Step 1: Build Vamana disk index (same as DiskANN first step)
        LOG_KNOWHERE_INFO_ << "PageANN: Building Vamana disk index at " << index_prefix;
        try {
            auto ret = pageann::build_disk_index<DataType>(
                data_path.c_str(), index_prefix.c_str(),
                nullptr,  // no labels
                max_degree, build_list_size,
                search_dram_budget, build_dram_budget,
                0,      // num_threads (0 = auto)
                false,  // use_pq_build
                0, false, false, 0, 0, 0, false);

            if (ret != 0) {
                LOG_KNOWHERE_ERROR_ << "PageANN: Vamana build failed with code " << ret;
                return Status::diskann_inner_error;
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "PageANN: Vamana build exception: " << e.what();
            return Status::diskann_inner_error;
        }

        // Step 2: Generate page graph (converts Vamana -> PageANN format)
        LOG_KNOWHERE_INFO_ << "PageANN: Generating page graph (min_degree=" << min_degree
                          << ", R=" << max_degree << ", pq_chunks=" << num_pq_chunks << ")";
        try {
            auto ret = pageann::build_page_graph<DataType>(
                index_prefix, data_path,
                min_degree, max_degree, num_pq_chunks,
                metric, page_build_mem, full_ooc);

            if (ret != 0) {
                LOG_KNOWHERE_ERROR_ << "PageANN: Page graph generation failed with code " << ret;
                return Status::diskann_inner_error;
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "PageANN: Page graph generation exception: " << e.what();
            return Status::diskann_inner_error;
        }

        index_prefix_ = index_prefix;
        LOG_KNOWHERE_INFO_ << "PageANN: Build complete (index prefix: " << index_prefix << ")";
        return Status::success;
    }

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    // =========================================================================
    // Deserialize: load the PageANN page-level index
    // =========================================================================
    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        auto& conf = static_cast<const PageANNConfig&>(*cfg);

        std::string index_prefix = conf.index_prefix.value();
        std::string pq_prefix = conf.pq_path_prefix.value_or("");
        bool use_hash = conf.use_hash_routing.value_or(false);
        bool use_sampled_hash = conf.use_sampled_hash_routing.value_or(false);
        uint32_t radius = static_cast<uint32_t>(conf.hash_radius.value_or(1));
        uint32_t num_pages_cache = static_cast<uint32_t>(conf.num_pages_to_cache.value_or(0));

        std::string metric_str = conf.metric_type.value();
        pageann::Metric metric = pageann::Metric::L2;
        if (metric_str == "IP" || metric_str == "INNER_PRODUCT") {
            metric = pageann::Metric::INNER_PRODUCT;
        } else if (metric_str == "COSINE") {
            metric = pageann::Metric::COSINE;
        }

        // Load index files via file manager (if needed)
        // The Paper's PQFlashIndex::load handles file I/O internally

        // Create aligned file reader for disk I/O
        auto reader = std::make_shared<pageann::LinuxAlignedFileReader>();

        // Create Paper PageANN's PQFlashIndex (page-level search)
        pq_flash_index_ = std::make_unique<pageann::PQFlashIndex<DataType>>(reader, metric);

        uint32_t num_threads = std::thread::hardware_concurrency();

        // Load the page-level index
        LOG_KNOWHERE_INFO_ << "PageANN: Loading index from " << index_prefix
                          << " (hash_routing=" << use_hash
                          << ", sampled_hash=" << use_sampled_hash
                          << ", radius=" << radius << ")";

        int load_result = pq_flash_index_->load(
            num_threads,
            index_prefix.c_str(),
            pq_prefix.empty() ? index_prefix : pq_prefix,
            use_hash,
            use_sampled_hash,
            radius);

        if (load_result != 0) {
            LOG_KNOWHERE_ERROR_ << "PageANN: Failed to load index, error code: " << load_result;
            return Status::diskann_inner_error;
        }

        // Cache frequently-accessed pages (core PageANN optimization)
        if (num_pages_cache > 0) {
            LOG_KNOWHERE_INFO_ << "PageANN: Generating page cache list (" << num_pages_cache << " pages)...";
            std::string warmup_file = index_prefix + "_sample_data.bin";
            std::vector<uint32_t> page_list;

            pq_flash_index_->generate_cache_list_from_sample_queries(
                warmup_file,
                64,               // l_search for sampling
                4,                // beamwidth for sampling
                num_pages_cache,
                num_threads,
                page_list,
                use_hash);

            pq_flash_index_->load_cache_list(page_list);
            LOG_KNOWHERE_INFO_ << "PageANN: Cached " << page_list.size() << " pages in memory";
        }

        dim_ = pq_flash_index_->get_data_dim();
        count_ = pq_flash_index_->get_num_points();
        index_prefix_ = index_prefix;
        use_hash_routing_ = use_hash;
        is_prepared_ = true;

        LOG_KNOWHERE_INFO_ << "PageANN: Index loaded (dim=" << dim_
                          << ", count=" << count_ << ")";
        return Status::success;
    }

    Status
    Serialize(BinarySet& binset) const override {
        // PageANN indexes are file-based; serialization managed by file manager
        return Status::success;
    }

    // =========================================================================
    // Search: uses Paper PageANN's page_search (page-level beam search)
    // =========================================================================
    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!is_prepared_) {
            return expected<DataSetPtr>::Err(Status::empty_index, "PageANN index not loaded");
        }

        auto& conf = static_cast<const PageANNConfig&>(*cfg);

        auto nq = dataset->GetRows();
        auto dim = dataset->GetDim();
        auto xq = dataset->GetTensor();

        auto k = conf.k.value();
        auto l_search = conf.search_list_size.value_or(64);
        auto beamwidth = conf.beamwidth.value_or(4);
        bool use_hash = use_hash_routing_;

        auto p_id = new int64_t[nq * k];
        auto p_dist = new float[nq * k];

        auto query_ptr = static_cast<const DataType*>(xq);

        // Per-query temporary buffer for uint64_t IDs from pageann
        auto tmp_ids = std::make_unique<uint64_t[]>(k);

        // TODO: Use Knowhere thread pool for parallel queries
        for (int64_t i = 0; i < nq; i++) {
            const DataType* query = query_ptr + i * dim;

            // Call Paper PageANN's page_search - the core page-level search
            pq_flash_index_->page_search(
                query,
                static_cast<uint64_t>(k),
                static_cast<uint64_t>(l_search),
                tmp_ids.get(),
                p_dist + i * k,
                static_cast<uint64_t>(beamwidth),
                false,    // use_reorder_data
                nullptr,  // stats
                use_hash);

            // Convert uint64_t -> int64_t
            for (int64_t j = 0; j < k; j++) {
                p_id[i * k + j] = static_cast<int64_t>(tmp_ids[j]);
            }
        }

        auto res = GenResultDataSet(nq, k, p_id, p_dist);
        return res;
    }

    expected<DataSetPtr>
    RangeSearch(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
                milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(
            Status::not_implemented, "PageANN does not support range search");
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(
            Status::not_implemented, "PageANN GetVectorByIds not yet implemented");
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(
            Status::not_implemented, "PageANN GetIndexMeta not yet implemented");
    }

    int64_t
    Dim() const override {
        return dim_.load();
    }

    int64_t
    Count() const override {
        return count_.load();
    }

    int64_t
    Size() const override {
        return 0;  // File-based index, size on disk
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_PAGEANN;
    }

 private:
    std::string index_prefix_;
    std::atomic_bool is_prepared_;
    bool use_hash_routing_ = false;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::unique_ptr<pageann::PQFlashIndex<DataType>> pq_flash_index_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
};

// Explicit template instantiations
// Note: Paper PageANN primarily supports float. fp16/bf16 would need conversion.
template class PageANNIndexNode<knowhere::fp32>;

// Register PAGEANN index type
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(PAGEANN, PageANNIndexNode,
                                                knowhere::feature::DISK)

}  // namespace knowhere

#endif  // KNOWHERE_WITH_PAGEANN
