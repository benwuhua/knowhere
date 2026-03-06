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

#ifdef KNOWHERE_WITH_PIPNN

#include "knowhere/feder/DiskANN.h"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <limits>

#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "fmt/core.h"
#include "index/diskann/impl/pipnn_build_profile.h"
#include "index/diskann/impl/pipnn_builder.h"
#include "index/diskann/impl/vamana_serializer.h"
#include "index/diskann/pipnn_diskann_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/prometheus_client.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"

namespace knowhere {

template <typename DataType>
class PiPNNDiskANNIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "PiPNN-DiskANN only support floating point data type(float32, float16, bfloat16)");

 public:
    using DistType = float;

    PiPNNDiskANNIndexNode(const int32_t& version, const Object& object) : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>));
        auto diskann_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
        assert(diskann_index_pack != nullptr);
        file_manager_ = diskann_index_pack->GetPack();
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override;

    Status
    Train(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    Status
    Add(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        return Status::not_implemented;
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override;

    expected<DataSetPtr>
    CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset, const int64_t* labels, const size_t labels_len,
                  const bool is_cosine, milvus::OpContext* op_context) const override;

    static bool
    StaticHasRawData(const knowhere::BaseConfig& config, const IndexVersion& version) {
        knowhere::MetricType metric_type = config.metric_type.has_value() ? config.metric_type.value() : "";
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return IsMetricType(metric_type, metric::L2) || IsMetricType(metric_type, metric::COSINE);
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override;

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "PiPNN-DiskANN does nothing for serialize";
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override;

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> config) override {
        LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<PiPNNDiskANNConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    Status
    SetFileManager(std::shared_ptr<milvus::FileManager> file_manager) {
        if (file_manager == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Malloc error, file_manager = nullptr.";
            return Status::malloc_error;
        }
        file_manager_ = file_manager;
        return Status::success;
    }

    int64_t
    Dim() const override {
        if (dim_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Dim() function is not supported when index is not ready yet.";
            return 0;
        }
        return dim_.load();
    }

    int64_t
    Size() const override {
        if (!is_prepared_.load() || !pq_flash_index_) {
            LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN not loaded.";
            return 0;
        }
        return pq_flash_index_->cal_size();
    }

    int64_t
    Count() const override {
        if (count_.load() == -1) {
            LOG_KNOWHERE_ERROR_ << "Count() function is not supported when index is not ready yet.";
            return 0;
        }
        return count_.load();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_PIPNN_DISKANN;
    }

    Status
    SetInternalIdToMostExternalIdMap(std::vector<uint32_t>&& map) override {
        internal_id_to_most_external_id_map_ = std::move(map);
        return Status::success;
    }

 private:
    bool
    LoadFile(const std::string& filename) {
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    bool
    AddFile(const std::string& filename) {
        if (!file_manager_->AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file " << filename << ".";
            return false;
        }
        return true;
    }

    uint64_t
    GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim, const uint64_t max_degree);

    std::string index_prefix_;
    mutable std::mutex preparation_lock_;
    std::atomic_bool is_prepared_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    std::unique_ptr<diskann::PQFlashIndex<DataType>> pq_flash_index_;
    std::atomic_int64_t dim_;
    std::atomic_int64_t count_;
    std::shared_ptr<ThreadPool> search_pool_;
    std::vector<uint32_t> internal_id_to_most_external_id_map_;
};

namespace {
static constexpr float kCacheExpansionRate = 1.2;

Status
TryDiskANNCall(std::function<void()>&& diskann_call) {
    try {
        diskann_call();
        return Status::success;
    } catch (const diskann::FileException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN File Exception: " << e.what();
        return Status::disk_file_error;
    } catch (const diskann::ANNException& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
        return Status::diskann_inner_error;
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "DiskANN Other Exception: " << e.what();
        return Status::diskann_inner_error;
    }
}

std::vector<std::string>
GetNecessaryFilenames(const std::string& prefix, const bool need_norm, const bool use_sample_cache,
                      const bool use_sample_warmup) {
    std::vector<std::string> filenames;
    auto pq_pivots_filename = diskann::get_pq_pivots_filename(prefix);
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);

    filenames.push_back(pq_pivots_filename);
    filenames.push_back(diskann::get_pq_rearrangement_perm_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_chunk_offsets_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_centroid_filename(pq_pivots_filename));
    filenames.push_back(diskann::get_pq_compressed_filename(prefix));
    filenames.push_back(disk_index_filename);
    if (need_norm) {
        filenames.push_back(diskann::get_disk_index_max_base_norm_file(disk_index_filename));
    }
    if (use_sample_cache || use_sample_warmup) {
        filenames.push_back(diskann::get_sample_data_filename(prefix));
    }
    return filenames;
}

std::vector<std::string>
GetOptionalFilenames(const std::string& prefix) {
    std::vector<std::string> filenames;
    auto disk_index_filename = diskann::get_disk_index_filename(prefix);
    filenames.push_back(diskann::get_disk_index_centroids_filename(disk_index_filename));
    filenames.push_back(diskann::get_disk_index_medoids_filename(disk_index_filename));
    filenames.push_back(diskann::get_cached_nodes_file(prefix));
    filenames.push_back(diskann::get_emb_list_offset_file(prefix));
    return filenames;
}

inline bool
AnyIndexFileExist(const std::string& index_prefix) {
    auto file_exist = [](std::vector<std::string> filenames) -> bool {
        for (auto& filename : filenames) {
            if (file_exists(filename)) {
                return true;
            }
        }
        return false;
    };
    return file_exist(GetNecessaryFilenames(index_prefix, diskann::INNER_PRODUCT, true, true)) ||
           file_exist(GetOptionalFilenames(index_prefix));
}

inline bool
CheckMetric(const std::string& diskann_metric) {
    if (diskann_metric != knowhere::metric::L2 && diskann_metric != knowhere::metric::IP &&
        diskann_metric != knowhere::metric::COSINE) {
        LOG_KNOWHERE_ERROR_ << "DiskANN currently only supports floating point "
                               "data for Minimum Euclidean "
                               "distance(L2), Max Inner Product Search(IP) "
                               "and Minimum Cosine Search(COSINE)."
                            << std::endl;
        return false;
    } else {
        return true;
    }
}
}  // namespace

template <typename DataType>
Status
PiPNNDiskANNIndexNode<DataType>::Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg,
                                       bool use_knowhere_build_pool) {
    (void)dataset;
    (void)use_knowhere_build_pool;
    using clock = std::chrono::steady_clock;

    assert(file_manager_ != nullptr);
    auto build_conf = static_cast<const PiPNNDiskANNConfig&>(*cfg);
    if (!CheckMetric(build_conf.metric_type.value())) {
        LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << build_conf.metric_type.value();
        return Status::invalid_metric_type;
    }
    if (!(build_conf.index_prefix.has_value() && build_conf.data_path.has_value())) {
        LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN file path for build is empty.";
        return Status::invalid_param_in_json;
    }
    if (AnyIndexFileExist(build_conf.index_prefix.value())) {
        LOG_KNOWHERE_ERROR_ << "This index prefix already has index files.";
        return Status::disk_file_error;
    }
    if (!LoadFile(build_conf.data_path.value())) {
        LOG_KNOWHERE_ERROR_ << "Failed load the raw data before building.";
        return Status::disk_file_error;
    }

    index_prefix_ = build_conf.index_prefix.value();

    size_t count = 0;
    size_t dim = 0;
    diskann::get_bin_metadata(build_conf.data_path.value(), count, dim);
    if (count == 0 || dim == 0 || count > std::numeric_limits<uint32_t>::max() ||
        dim > std::numeric_limits<uint32_t>::max()) {
        LOG_KNOWHERE_ERROR_ << "Invalid fbin metadata count=" << count << " dim=" << dim;
        return Status::invalid_args;
    }

    count_.store(static_cast<int64_t>(count));
    dim_.store(static_cast<int64_t>(dim));

    std::unique_ptr<float[]> float_data;
    size_t loaded_count = 0;
    size_t loaded_dim = 0;
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        diskann::load_bin<float>(build_conf.data_path.value(), float_data, loaded_count, loaded_dim);
    }));
    if (loaded_count != count || loaded_dim != dim) {
        LOG_KNOWHERE_ERROR_ << "Loaded data shape mismatch, expected (" << count << ", " << dim << ") got ("
                            << loaded_count << ", " << loaded_dim << ")";
        return Status::disk_file_error;
    }

    pipnn_diskann::PiPNNBuilder::Config pipnn_cfg;
    pipnn_cfg.k_nn = static_cast<uint32_t>(build_conf.pipnn_k_nn.value());
    pipnn_cfg.hash_bits = static_cast<uint32_t>(build_conf.pipnn_hash_bits.value());
    pipnn_cfg.max_degree = static_cast<uint32_t>(build_conf.max_degree.value());
    pipnn_cfg.final_prune = build_conf.pipnn_final_prune.value();
    pipnn_cfg.leaf_max_size = static_cast<size_t>(build_conf.pipnn_leaf_max_size.value());
    pipnn_cfg.fanout_l1 = static_cast<uint32_t>(build_conf.pipnn_fanout_l1.value());
    pipnn_cfg.fanout_l2 = static_cast<uint32_t>(build_conf.pipnn_fanout_l2.value());

    pipnn_diskann::BuildProfile build_profile;
    const auto graph_stage_start = clock::now();
    pipnn_diskann::PiPNNBuilder builder(pipnn_cfg);
    auto graph = builder.build(float_data.get(), static_cast<uint32_t>(count), static_cast<uint32_t>(dim));
    if (graph.size() != count) {
        LOG_KNOWHERE_ERROR_ << "PiPNN graph size mismatch, expected " << count << " got " << graph.size();
        return Status::diskann_inner_error;
    }

    const uint32_t entry_point =
        pipnn_diskann::VamanaSerializer::find_medoid(float_data.get(), static_cast<uint32_t>(count),
                                                     static_cast<uint32_t>(dim));
    const std::string mem_index_path = index_prefix_ + "_mem.index";
    try {
        pipnn_diskann::VamanaSerializer::write(graph, entry_point, mem_index_path);
    } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "Failed to write PiPNN mem index: " << e.what();
        return Status::disk_file_error;
    }
    build_profile.graph_construction_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - graph_stage_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Build stage: "
                       << build_profile.stage_ms(pipnn_diskann::BuildStage::kGraphConstruction) << " ms ("
                       << pipnn_diskann::BuildStageName(pipnn_diskann::BuildStage::kGraphConstruction)
                       << ", num_points=" << count << ", dim=" << dim << ")";

    const bool need_norm = IsMetricType(build_conf.metric_type.value(), knowhere::metric::IP) ||
                           IsMetricType(build_conf.metric_type.value(), knowhere::metric::COSINE);
    const auto diskann_metric = [m = build_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();
    auto num_nodes_to_cache =
        GetCachedNodeNum(build_conf.search_cache_budget_gb.value(), dim, build_conf.max_degree.value());
    diskann::BuildConfig diskann_internal_build_config{build_conf.data_path.value(),
                                                       index_prefix_,
                                                       diskann_metric,
                                                       static_cast<unsigned>(build_conf.max_degree.value()),
                                                       static_cast<unsigned>(build_conf.search_list_size.value()),
                                                       static_cast<double>(build_conf.pq_code_budget_gb.value()),
                                                       static_cast<double>(build_conf.build_dram_budget_gb.value()),
                                                       static_cast<uint32_t>(build_conf.disk_pq_dims.value()),
                                                       false,
                                                       build_conf.accelerate_build.value(),
                                                       static_cast<uint32_t>(num_nodes_to_cache),
                                                       build_conf.shuffle_build.value()};
    const auto pq_stage_start = clock::now();
    RETURN_IF_ERROR(TryDiskANNCall([&]() {
        int res = diskann::build_disk_index<DataType>(diskann_internal_build_config);
        if (res != 0) {
            throw diskann::ANNException("diskann::build_disk_index returned non-zero value: " + std::to_string(res),
                                        -1);
        }
    }));
    build_profile.pq_and_disk_layout_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - pq_stage_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Build stage: "
                       << build_profile.stage_ms(pipnn_diskann::BuildStage::kPQAndDiskLayout) << " ms ("
                       << pipnn_diskann::BuildStageName(pipnn_diskann::BuildStage::kPQAndDiskLayout)
                       << ", num_points=" << count << ", dim=" << dim << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Build stage: "
                       << build_profile.stage_ms(pipnn_diskann::BuildStage::kTotal) << " ms ("
                       << pipnn_diskann::BuildStageName(pipnn_diskann::BuildStage::kTotal)
                       << ", graph_pct=" << (build_profile.total_ns() == 0
                                                 ? 0.0
                                                 : 100.0 * static_cast<double>(build_profile.graph_construction_ns) /
                                                       static_cast<double>(build_profile.total_ns()))
                       << ", pq_disk_pct=" << (build_profile.total_ns() == 0
                                                   ? 0.0
                                                   : 100.0 * static_cast<double>(build_profile.pq_and_disk_layout_ns) /
                                                         static_cast<double>(build_profile.total_ns()))
                       << ")";

    for (auto& filename : GetNecessaryFilenames(index_prefix_, need_norm, true, true)) {
        if (!AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        if (file_exists(filename) && !AddFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to add file " << filename << ".";
            return Status::disk_file_error;
        }
    }

    is_prepared_.store(false);
    return Status::success;
}

template <typename DataType>
Status
PiPNNDiskANNIndexNode<DataType>::Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) {
    (void)binset;

    auto prep_conf = static_cast<const PiPNNDiskANNConfig&>(*cfg);
    if (!CheckMetric(prep_conf.metric_type.value())) {
        return Status::invalid_metric_type;
    }

    std::lock_guard<std::mutex> lock(preparation_lock_);
    if (is_prepared_.load()) {
        return Status::success;
    }
    if (!(prep_conf.index_prefix.has_value())) {
        LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN file path for deserialize is empty.";
        return Status::invalid_param_in_json;
    }
    index_prefix_ = prep_conf.index_prefix.value();

    bool is_ip = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP);
    bool need_norm = IsMetricType(prep_conf.metric_type.value(), knowhere::metric::IP) ||
                     IsMetricType(prep_conf.metric_type.value(), knowhere::metric::COSINE);
    auto diskann_metric = [m = prep_conf.metric_type.value()] {
        if (IsMetricType(m, knowhere::metric::L2)) {
            return diskann::Metric::L2;
        } else if (IsMetricType(m, knowhere::metric::COSINE)) {
            return diskann::Metric::COSINE;
        } else {
            return diskann::Metric::INNER_PRODUCT;
        }
    }();

    for (auto& filename : GetNecessaryFilenames(
             index_prefix_, need_norm, prep_conf.search_cache_budget_gb.value() > 0 && !prep_conf.use_bfs_cache.value(),
             prep_conf.warm_up.value())) {
        if (!LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }
    for (auto& filename : GetOptionalFilenames(index_prefix_)) {
        auto is_exist_op = file_manager_->IsExisted(filename);
        if (!is_exist_op.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Failed to check existence of file " << filename << ".";
            return Status::disk_file_error;
        }
        if (is_exist_op.value() && !LoadFile(filename)) {
            return Status::disk_file_error;
        }
    }

    search_pool_ = ThreadPool::GetGlobalSearchThreadPool();

    std::shared_ptr<AlignedFileReader> reader = nullptr;
    reader.reset(new LinuxAlignedFileReader());

    pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<DataType>>(reader, diskann_metric);
    auto disk_ann_call = [&]() {
        int res = pq_flash_index_->load(search_pool_->size(), index_prefix_.c_str());
        if (res != 0) {
            throw diskann::ANNException("pq_flash_index_->load returned non-zero value: " + std::to_string(res), -1);
        }
    };
    if (TryDiskANNCall(disk_ann_call) != Status::success) {
        LOG_KNOWHERE_ERROR_ << "Failed to load PiPNN-DiskANN.";
        return Status::diskann_inner_error;
    }

    count_.store(pq_flash_index_->get_num_points());
    if (is_ip) {
        dim_.store(pq_flash_index_->get_data_dim() - 1);
    } else {
        dim_.store(pq_flash_index_->get_data_dim());
    }

    std::string warmup_query_file = diskann::get_sample_data_filename(index_prefix_);
    auto cached_nodes_file = diskann::get_cached_nodes_file(index_prefix_);
    std::vector<uint32_t> node_list;
    if (file_exists(cached_nodes_file)) {
        LOG_KNOWHERE_INFO_ << "Reading cached nodes from file.";
        size_t num_nodes, nodes_id_dim;
        std::unique_ptr<uint32_t[]> cached_nodes_ids = nullptr;
        diskann::load_bin<uint32_t>(cached_nodes_file, cached_nodes_ids, num_nodes, nodes_id_dim);
        node_list.assign(cached_nodes_ids.get(), cached_nodes_ids.get() + num_nodes);
    } else {
        auto num_nodes_to_cache = GetCachedNodeNum(prep_conf.search_cache_budget_gb.value(), pq_flash_index_->get_data_dim(),
                                                   pq_flash_index_->get_max_degree());
        if (num_nodes_to_cache > pq_flash_index_->get_num_points() / 3) {
            LOG_KNOWHERE_ERROR_ << "Failed to generate cache, num_nodes_to_cache(" << num_nodes_to_cache
                                << ") is larger than 1/3 of the total data number.";
            return Status::invalid_args;
        }
        if (num_nodes_to_cache > 0) {
            LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " sample nodes around medoid(s).";
            if (prep_conf.use_bfs_cache.value()) {
                LOG_KNOWHERE_INFO_ << "Use bfs to generate cache list";
                if (TryDiskANNCall([&]() { pq_flash_index_->cache_bfs_levels(num_nodes_to_cache, node_list); }) !=
                    Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate bfs cache for PiPNN-DiskANN.";
                    return Status::diskann_inner_error;
                }
            } else {
                LOG_KNOWHERE_INFO_ << "Use sample_queries to generate cache list";
                if (TryDiskANNCall([&]() {
                        pq_flash_index_->async_generate_cache_list_from_sample_queries(warmup_query_file, 15, 6,
                                                                                       num_nodes_to_cache);
                    }) != Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate cache from sample queries for PiPNN-DiskANN.";
                    return Status::diskann_inner_error;
                }
            }
        }
        LOG_KNOWHERE_INFO_ << "End of preparing PiPNN-DiskANN index.";
    }

    if (node_list.size() > 0) {
        if (TryDiskANNCall([&]() { pq_flash_index_->load_cache_list(node_list); }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load cache for PiPNN-DiskANN.";
            return Status::diskann_inner_error;
        }
    }

    if (prep_conf.warm_up.value()) {
        LOG_KNOWHERE_INFO_ << "Warming up.";
        uint64_t warmup_L = 20;
        uint64_t warmup_num = 0;
        uint64_t warmup_dim = 0;
        uint64_t warmup_aligned_dim = 0;
        DataType* warmup = nullptr;
        if (TryDiskANNCall([&]() {
                diskann::load_aligned_bin<DataType>(warmup_query_file, warmup, warmup_num, warmup_dim,
                                                    warmup_aligned_dim);
            }) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load warmup file for PiPNN-DiskANN.";
            return Status::disk_file_error;
        }
        std::vector<int64_t> warmup_result_ids_64(warmup_num, 0);
        std::vector<DistType> warmup_result_dists(warmup_num, 0);

        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(warmup_num);
        for (_s64 i = 0; i < (int64_t)warmup_num; ++i) {
            futures.emplace_back(search_pool_->push([&, index = i]() {
                pq_flash_index_->cached_beam_search(warmup + (index * warmup_aligned_dim), 1, warmup_L,
                                                    warmup_result_ids_64.data() + (index * 1),
                                                    warmup_result_dists.data() + (index * 1), 4);
            }));
        }

        bool failed = TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success;

        if (warmup != nullptr) {
            diskann::aligned_free(warmup);
        }

        if (failed) {
            LOG_KNOWHERE_ERROR_ << "Failed to do search on warmup file for PiPNN-DiskANN.";
            return Status::diskann_inner_error;
        }
    }

    is_prepared_.store(true);
    LOG_KNOWHERE_INFO_ << "End of PiPNN-DiskANN loading.";
    return Status::success;
}

template <typename DataType>
expected<DataSetPtr>
PiPNNDiskANNIndexNode<DataType>::Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg,
                                        const BitsetView& bitset_, milvus::OpContext* op_context) const {
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load PiPNN-DiskANN.";
        return expected<DataSetPtr>::Err(Status::empty_index, "PiPNNDiskANN not loaded");
    }

    auto search_conf = static_cast<const PiPNNDiskANNConfig&>(*cfg);
    if (!CheckMetric(search_conf.metric_type.value())) {
        return expected<DataSetPtr>::Err(Status::invalid_metric_type, "unsupported metric type");
    }
    auto k = static_cast<uint64_t>(search_conf.k.value());
    auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value());
    auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value());
    auto filter_ratio = static_cast<float>(search_conf.filter_threshold.value());
    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = static_cast<const DataType*>(dataset->GetTensor());

    feder::diskann::FederResultUniq feder_result;
    if (search_conf.trace_visit.value()) {
        if (nq != 1) {
            return expected<DataSetPtr>::Err(Status::invalid_args, "nq must be 1");
        }
        feder_result = std::make_unique<feder::diskann::FederResult>();
        feder_result->visit_info_.SetQueryConfig(search_conf.k.value(), search_conf.beamwidth.value(),
                                                 search_conf.search_list_size.value(), search_conf.beamwidth.value());
    }

    BitsetView bitset(bitset_);
    if (!internal_id_to_most_external_id_map_.empty()) {
        bitset.set_out_ids(internal_id_to_most_external_id_map_.data(), internal_id_to_most_external_id_map_.size());
    }

    auto p_id = std::make_unique<int64_t[]>(k * nq);
    auto p_dist = std::make_unique<DistType[]>(k * nq);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            diskann::QueryStats stats;
            pq_flash_index_->cached_beam_search(xq + (index * dim), k, lsearch, p_id_ptr + (index * k),
                                                p_dist_ptr + (index * k), beamwidth, false, &stats, feder_result,
                                                bitset, filter_ratio);
#ifdef NOT_COMPILE_FOR_SWIG
            knowhere_diskann_search_hops.Observe(stats.n_hops);
#endif
        }));
    }

    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some search failed");
    }

    auto res = GenResultDataSet(nq, k, std::move(p_id), std::move(p_dist));

    if (feder_result != nullptr) {
        Json json_visit_info, json_id_set;
        nlohmann::to_json(json_visit_info, feder_result->visit_info_);
        nlohmann::to_json(json_id_set, feder_result->id_set_);
        res->SetJsonInfo(json_visit_info.dump());
        res->SetJsonIdSet(json_id_set.dump());
    }
    return res;
}

template <typename DataType>
expected<DataSetPtr>
PiPNNDiskANNIndexNode<DataType>::CalcDistByIDs(const DataSetPtr dataset, const BitsetView& bitset,
                                                const int64_t* labels, const size_t labels_len, const bool is_cosine,
                                                milvus::OpContext* op_context) const {
    (void)bitset;
    (void)is_cosine;
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load PiPNN-DiskANN.";
        return expected<DataSetPtr>::Err(Status::empty_index, "PiPNNDiskANN not loaded");
    }
    if (!search_pool_) {
        LOG_KNOWHERE_ERROR_ << "Search thread pool is not initialized.";
        return expected<DataSetPtr>::Err(Status::internal_error, "search pool not initialized");
    }
    if (dataset == nullptr || dataset->GetTensor() == nullptr) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "empty query dataset");
    }
    if (labels == nullptr && labels_len != 0) {
        return expected<DataSetPtr>::Err(Status::invalid_args, "labels is nullptr");
    }

    auto nq = dataset->GetRows();
    auto dim = dataset->GetDim();
    auto xq = static_cast<const DataType*>(dataset->GetTensor());
    auto p_dist = std::make_unique<DistType[]>(nq * labels_len);

    std::vector<folly::Future<folly::Unit>> futures;
    futures.reserve(nq);
    for (int64_t row = 0; row < nq; ++row) {
        futures.emplace_back(search_pool_->push([&, index = row, p_dist_ptr = p_dist.get()]() {
            knowhere::checkCancellation(op_context);
            pq_flash_index_->calc_dist_by_ids(xq + (index * dim), labels, static_cast<int64_t>(labels_len),
                                              p_dist_ptr + index * labels_len);
        }));
    }
    if (TryDiskANNCall([&]() { WaitAllSuccess(futures); }) != Status::success) {
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some calc dist by ids failed");
    }

    std::unique_ptr<int64_t[]> ids = nullptr;
    return GenResultDataSet(nq, labels_len, std::move(ids), std::move(p_dist));
}

template <typename DataType>
expected<DataSetPtr>
PiPNNDiskANNIndexNode<DataType>::GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const {
    (void)op_context;
    if (!is_prepared_.load() || !pq_flash_index_) {
        LOG_KNOWHERE_ERROR_ << "Failed to load PiPNN-DiskANN.";
        return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
    }
    auto dim = Dim();
    auto rows = dataset->GetRows();
    auto ids = dataset->GetIds();
    auto* data = new DataType[dim * rows];
    if (data == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Failed to allocate memory for data.";
        return expected<DataSetPtr>::Err(Status::malloc_error, "failed to allocate memory for data");
    }

    if (TryDiskANNCall([&]() { pq_flash_index_->get_vector_by_ids(ids, rows, data); }) != Status::success) {
        delete[] data;
        return expected<DataSetPtr>::Err(Status::diskann_inner_error, "failed to get vector");
    }

    return GenResultDataSet(rows, dim, data);
}

template <typename DataType>
expected<DataSetPtr>
PiPNNDiskANNIndexNode<DataType>::GetIndexMeta(std::unique_ptr<Config> cfg) const {
    std::vector<int64_t> entry_points;
    for (size_t i = 0; i < pq_flash_index_->get_num_medoids(); i++) {
        entry_points.push_back(pq_flash_index_->get_medoids()[i]);
    }
    auto diskann_conf = static_cast<const PiPNNDiskANNConfig&>(*cfg);
    feder::diskann::DiskANNMeta meta(diskann_conf.data_path.value(), diskann_conf.max_degree.value(),
                                     diskann_conf.search_list_size.value(), diskann_conf.pq_code_budget_gb.value(),
                                     diskann_conf.build_dram_budget_gb.value(), diskann_conf.disk_pq_dims.value(),
                                     diskann_conf.accelerate_build.value(), Count(), entry_points);
    std::unordered_set<int64_t> id_set(entry_points.begin(), entry_points.end());

    Json json_meta, json_id_set;
    nlohmann::to_json(json_meta, meta);
    nlohmann::to_json(json_id_set, id_set);
    return GenResultDataSet(json_meta.dump(), json_id_set.dump());
}

template <typename DataType>
uint64_t
PiPNNDiskANNIndexNode<DataType>::GetCachedNodeNum(const float cache_dram_budget, const uint64_t data_dim,
                                                  const uint64_t max_degree) {
    uint32_t one_cached_node_budget = (max_degree + 1) * sizeof(unsigned) + sizeof(DataType) * data_dim;
    auto num_nodes_to_cache =
        static_cast<uint64_t>(1024 * 1024 * 1024 * cache_dram_budget) / (one_cached_node_budget * kCacheExpansionRate);
    return num_nodes_to_cache;
}

KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(PIPNN_DISKANN, PiPNNDiskANNIndexNode, knowhere::feature::DISK)

}  // namespace knowhere

#endif
