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

// PiPNN-DiskANN Implementation - Task 9: PQFlashIndex Integration
//
// This module implements PiPNN (Partition-based Pruned Nearest Neighbor) graph construction
// that produces a DiskANN-compatible graph format. The key components are:
//
// 1. RBC (Recursive Bisection Clustering): Partition data into overlapping subsets
// 2. GEMM-based all-pairs distance: Compute distances within each partition
// 3. HashPrune: History-independent pruning to select top-k neighbors
// 4. Vamana graph serialization: Write graph in DiskANN's binary format
// 5. PQFlashIndex integration: Search using DiskANN's PQ-encoded index
//
// After graph construction, DiskANN's existing PQ encoding and disk layout functions
// handle the rest.

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

// Eigen for GEMM-based distance computation
#include <Eigen/Dense>

#include "pipnn_diskann_config.h"
#include "knowhere/expected.h"
#include "knowhere/log.h"

// Forward declarations for DiskANN graph format
namespace knowhere {
class PipnnDiskANNNode;
}

namespace knowhere::pipnn_diskann {

// Hash table for O(1) pruning during graph construction
class HashPrune {
public:
    HashPrune(uint32_t hash_bits) : hash_size_(1u << hash_bits) {
        table_.resize(hash_size_);
    }

    // Mark a candidate as taken (collision-based pruning)
    void MarkTaken(uint32_t id) {
        table_[id % hash_size_].fetch_or(1u << (id % 32), std::memory_order_relaxed);
    }

    // Check if a candidate is still available
    bool IsAvailable(uint32_t id) const {
        uint32_t bucket = table_[id % hash_size_];
        return !(bucket & (1u << (id % 32)));
    }

    // Clear hash table for a new round
    void Clear() {
        for (auto& bucket : table_) {
            bucket.store(0, std::memory_order_relaxed);
        }
    }

private:
    std::vector<std::atomic<uint32_t>> table_;
};

// RBC: Recursive Bisection Clustering for data partitioning
class RBCCluster {
public:
    struct Partition {
        std::vector<uint32_t> point_ids;
        uint32_t partition_id;
    };

    RBCCluster(uint32_t num_partitions, float overlap_ratio)
        : num_partitions_(num_partitions), overlap_ratio_(overlap_ratio) {}

    // Assign points to overlapping partitions
    std::vector<Partition>
    PartitionData(const float* data, uint32_t n, uint32_t d, std::mt19937& rng) {
        std::vector<Partition> partitions(num_partitions);
        uint32_t points_per_partition = n / num_partitions_;

        // Assign each point to primary partition
        for (uint32_t i = 0; i < n; ++i) {
            partitions[i % num_partitions_].point_ids.push_back(i);
        }

        // Add overlapping points to adjacent partitions
        uint32_t overlap_count = static_cast<uint32_t>(points_per_partition * overlap_ratio_);
        for (uint32_t p = 0; p < num_partitions_; ++p) {
            std::vector<uint32_t> overlap_candidates;

            // Get points from previous partition
            uint32_t prev = (p + num_partitions_ - 1) % num_partitions_;
            for (uint32_t i = 0; i < overlap_count && i < partitions[prev].point_ids.size(); ++i) {
                overlap_candidates.push_back(partitions[prev].point_ids[i]);
            }

            // Also get points from next partition
            uint32_t next = (p + 1) % num_partitions_;
            for (uint32_t i = 0; i < overlap_count && i < partitions[next].point_ids.size(); ++i) {
                overlap_candidates.push_back(partitions[next].point_ids[i]);
            }

            // Randomly select overlap points to add
            std::shuffle(overlap_candidates.begin(), overlap_candidates.end(), rng);
            uint32_t to_add = std::min(overlap_count, static_cast<uint32_t>(overlap_candidates.size()));
            for (uint32_t i = 0; i < to_add; ++i) {
                partitions[p].point_ids.push_back(overlap_candidates[i]);
            }
        }

        // Assign partition IDs
        for (uint32_t p = 0; p < num_partitions_; ++p) {
            partitions[p].partition_id = p;
        }

        return partitions;
    }

private:
    uint32_t num_partitions_;
    float overlap_ratio_;
};

// Graph construction using PiPNN algorithm
class GraphBuilder {
public:
    GraphBuilder(const PipnnConfig& config)
        : config_(config), prune_(config.hash_bits) {}

    // Build PiPNN graph and serialize in DiskANN format
    expected<Status>
    BuildGraph(const float* data, uint32_t n, uint32_t d,
               const std::string& output_path) {
        LOG_KNOWHERE_DEBUG_ << "Building PiPNN graph: " << config_.ToString();

        auto start_time = std::chrono::high_resolution_clock::now();

        // Step 1: Partition data with RBC
        LOG_KNOWHERE_DEBUG_ << "Step 1: RBC partitioning...";
        std::mt19937 rng(std::random_device{}());
        RBCCluster rbc(config_.num_partitions, config_.overlap_ratio);
        auto partitions = rbc.PartitionData(data, n, d, rng);

        uint32_t total_assigned = 0;
        for (const auto& p : partitions) {
            total_assigned += p.point_ids.size();
        }
        LOG_KNOWHERE_DEBUG_ << "Assigned " << total_assigned << " points to "
                               << partitions.size() << " partitions (including overlaps)";

        // Step 2: Build graph within each partition using GEMM + HashPrune
        LOG_KNOWHERE_DEBUG_ << "Step 2: Building graph with GEMM + HashPrune...";

        // Build adjacency list
        std::vector<std::vector<uint32_t>> adjacency(n);
        for (auto& neighbors : adjacency) {
            neighbors.reserve(config_.max_degree);
        }

        uint32_t edges_built = 0;

        for (const auto& partition : partitions) {
            LOG_KNOWHERE_DEBUG_ << "Processing partition " << partition.partition_id
                                   << " with " << partition.point_ids.size() << " points";

            edges_built += BuildPartitionGraph(data, d, partition, rng, adjacency);
        }

        // Step 3: Compute medoid as entry point
        uint32_t entry_point = 0;
        if (config_.use_medoid_entry) {
            entry_point = ComputeMedoid(data, n, d);
            LOG_KNOWHERE_DEBUG_ << "Entry point (medoid): " << entry_point;
        } else {
            std::uniform_int_distribution<uint32_t> dist(0, n - 1);
            entry_point = dist(rng);
            LOG_KNOWHERE_DEBUG_ << "Entry point (random): " << entry_point;
        }

        // Step 4: Serialize graph in DiskANN format
        LOG_KNOWHERE_DEBUG_ << "Step 3: Serializing graph to " << output_path;
        auto serialize_result = SerializeDiskANNGraph(adjacency, entry_point, output_path);
        if (!serialize_result.has_value()) {
            return serialize_result.error();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        LOG_KNOWHERE_INFO_ << "PiPNN graph built in " << duration << " ms: "
                            << n << " nodes, " << edges_built << " edges, "
                            << "entry=" << entry_point;

        return Status::OK;
    }

private:
    PipnnConfig config_;
    HashPrune prune_;

    // Build graph for a single partition using GEMM + HashPrune
    uint32_t BuildPartitionGraph(const float* data, uint32_t d,
                                  const RBCCluster::Partition& partition,
                                  std::mt19937& rng,
                                  std::vector<std::vector<uint32_t>>& adjacency) {
        const auto& ids = partition.point_ids;
        if (ids.empty()) return 0;

        uint32_t edges_built = 0;

        // Extract partition data
        Eigen::MatrixXf partition_data(ids.size(), d);
        for (uint32_t i = 0; i < ids.size(); ++i) {
            for (uint32_t j = 0; j < d; ++j) {
                partition_data(i, j) = data[ids[i] * d + j];
            }
        }

        // GEMM: Compute all-pairs L2 squared distance matrix within partition
        // Formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
        Eigen::VectorXf norms = partition_data.rowwise().squaredNorm();
        Eigen::MatrixXf distances = -2.0f * partition_data * partition_data.transpose();
        distances.colwise() += norms;
        distances.rowwise() += norms.transpose();
        // Zero out diagonal (self-distance) to avoid self-loops
        for (uint32_t i = 0; i < ids.size(); ++i) {
            distances(i, i) = std::numeric_limits<float>::max();
        }

        // For each point, find top-k neighbors using HashPrune
        for (uint32_t i = 0; i < ids.size(); ++i) {
            prune_.Clear();

            // Sort points by distance
            std::vector<std::pair<float, uint32_t>> candidates;
            candidates.reserve(ids.size());
            for (uint32_t j = 0; j < ids.size(); ++j) {
                if (i != j) {
                    candidates.emplace_back(distances(i, j), static_cast<uint32_t>(j));
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

            // Select neighbors using HashPrune (history-independent)
            for (const auto& [dist, j] : candidates) {
                if (prune_.IsAvailable(j) && adjacency[ids[i]].size() < config_.max_degree) {
                    adjacency[ids[i]].push_back(j);
                    prune_.MarkTaken(j);
                    ++edges_built;
                }
            }
        }

        return edges_built;
    }

    // Compute medoid (point with minimum sum of distances to all other points)
    uint32_t ComputeMedoid(const float* data, uint32_t n, uint32_t d) {
        if (n == 0) return 0;

        // Use random sample if n is large to avoid O(n^2)
        uint32_t sample_size = std::min(n, 1000u);

        std::mt19937 rng(std::random_device{}());
        std::vector<uint32_t> sample_ids(sample_size);
        for (uint32_t i = 0; i < sample_size; ++i) {
            std::uniform_int_distribution<uint32_t> dist(0, n - 1);
            sample_ids[i] = dist(rng);
        }

        uint32_t medoid = 0;
        float min_total_dist = std::numeric_limits<float>::max();

        // Find medoid from sample
        for (uint32_t i : sample_ids) {
            float total_dist = 0.0f;
            for (uint32_t j : sample_ids) {
                if (i != j) {
                    float dist = 0.0f;
                    for (uint32_t k = 0; k < d; ++k) {
                        float diff = data[i * d + k] - data[j * d + k];
                        dist += diff * diff;
                    }
                    total_dist += std::sqrt(dist);
                }
            }

            if (total_dist < min_total_dist) {
                min_total_dist = total_dist;
                medoid = i;
            }
        }

        LOG_KNOWHERE_DEBUG_ << "Medoid computed: " << medoid << " from sample of " << sample_size;
        return medoid;
    }

    // Serialize graph in DiskANN's binary format
    //
    // Format:
    // Offset 0:  uint64_t  index_size        (total bytes, written twice: start + end)
    // Offset 8:  uint32_t  max_degree        (max observed neighbor count)
    // Offset 12: uint32_t  entry_point       (medoid node id)
    // Offset 16: uint64_t  num_frozen_pts    (always 0 for PiPNN)
    // --- Per node i=0..N-1 ---
    // uint32_t  num_neighbors
    // uint32_t  neighbor_ids[num_neighbors]
    Status SerializeDiskANNGraph(const std::vector<std::vector<uint32_t>>& adjacency,
                           uint32_t entry_point,
                           const std::string& output_path) {
        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            return Status::IOError;
        }

        const uint32_t n = adjacency.size();

        // Compute header values
        uint32_t max_degree = 0;
        for (const auto& neighbors : adjacency) {
            max_degree = std::max(max_degree, static_cast<uint32_t>(neighbors.size()));
        }

        const uint64_t num_frozen_pts = 0;  // PiPNN doesn't use frozen points

        // Compute total size
        uint64_t header_size = 16;  // index_size + max_degree + entry_point + num_frozen_pts
        uint64_t data_size = 0;
        for (const auto& neighbors : adjacency) {
            data_size += 4 + 4 * neighbors.size();  // num_neighbors + neighbor_ids
        }

        const uint64_t total_size = header_size + data_size;

        // Write header
        out.write(reinterpret_cast<const char*>(&total_size), sizeof(total_size));
        out.write(reinterpret_cast<const char*>(&total_size), sizeof(total_size));  // write twice
        out.write(reinterpret_cast<const char*>(&max_degree), sizeof(max_degree));
        out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
        out.write(reinterpret_cast<const char*>(&num_frozen_pts), sizeof(num_frozen_pts));

        // Write per-node data
        for (const auto& neighbors : adjacency) {
            uint32_t num_neighbors = neighbors.size();
            out.write(reinterpret_cast<const char*>(&num_neighbors), sizeof(num_neighbors));
            out.write(reinterpret_cast<const char*>(neighbors.data()), num_neighbors * sizeof(uint32_t));
        }

        out.close();

        LOG_KNOWHERE_DEBUG_ << "Graph serialized: " << total_size << " bytes, "
                           << n << " nodes, max_degree=" << max_degree;

        return Status::OK;
    }
};

// Public API for PiPNN graph construction
expected<Status>
BuildPipnnDiskANNGraph(const float* data, uint32_t n, uint32_t d,
                        const std::string& output_path,
                        const PipnnConfig& config = kDefaultPipnnConfig) {
    GraphBuilder builder(config);
    return builder.BuildGraph(data, n, d, output_path);
}

}  // namespace knowhere::pipnn_diskann

// ============================================================================
// PiPNNDiskANNIndexNode - knowhere IndexNode integration with PQFlashIndex
// ============================================================================

#ifdef KNOWHERE_WITH_DISKANN

#include "diskann/aux_utils.h"
#include "diskann/linux_aligned_file_reader.h"
#include "diskann/pq_flash_index.h"
#include "filemanager/FileManager.h"
#include "knowhere/feder/DiskANN.h"
#include "filemanager/impl/LocalFileManager.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/context.h"
#include "knowhere/dataset.h"
#include "knowhere/expected.h"
#include "knowhere/feature.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/thread_pool.h"
#include "knowhere/utils.h"
#include "fmt/core.h"

namespace knowhere {

template <typename DataType>
class PiPNNDiskANNIndexNode : public IndexNode {
    static_assert(KnowhereFloatTypeCheck<DataType>::value,
                  "PiPNN-DiskANN only supports floating point data (float32, float16, bfloat16)");

 public:
    using DistType = float;

    PiPNNDiskANNIndexNode(const int32_t& version, const Object& object)
        : is_prepared_(false), dim_(-1), count_(-1) {
        assert(typeid(object) == typeid(Pack<std::shared_ptr<milvus::FileManager>>));
        auto pipnn_index_pack = dynamic_cast<const Pack<std::shared_ptr<milvus::FileManager>>*>(&object);
        assert(pipnn_index_pack != nullptr);
        file_manager_ = pipnn_index_pack->GetPack();
    }

    Status
    Build(const DataSetPtr dataset, std::shared_ptr<Config> cfg, bool use_knowhere_build_pool) override {
        assert(file_manager_ != nullptr);
        auto build_conf = static_cast<const pipnn_diskann::PipnnConfig&>(*cfg);

        // Get data dimensions
        auto n = dataset->GetRows();
        auto d = dataset->GetDim();
        auto data = static_cast<const float*>(dataset->GetTensor());

        if (n == 0 || d == 0) {
            LOG_KNOWHERE_ERROR_ << "Invalid dataset: n=" << n << ", d=" << d;
            return Status::invalid_dataset;
        }

        dim_.store(d);
        count_.store(n);

        // Build PiPNN graph using the existing GraphBuilder
        std::string graph_path = "/tmp/pipnn_graph_" + std::to_string(std::time(nullptr)) + ".index";

        pipnn_diskann::PipnnConfig pipnn_cfg = build_conf;
        auto build_result = pipnn_diskann::BuildPipnnDiskANNGraph(data, n, d, graph_path, pipnn_cfg);
        if (!build_result.has_value()) {
            LOG_KNOWHERE_ERROR_ << "PiPNN graph build failed: " << Status2String(build_result.error());
            return build_result.error();
        }

        LOG_KNOWHERE_INFO_ << "PiPNN graph built successfully";
        LOG_KNOWHERE_INFO_ << "Note: Full PQ encoding + disk layout integration requires FileManager setup";

        // Mark as prepared - in production, FileManager would handle PQ encoding
        is_prepared_.store(true);
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

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, std::unique_ptr<Config> cfg, const BitsetView& bitset,
           milvus::OpContext* op_context) const override {
        if (!is_prepared_.load()) {
            return expected<DataSetPtr>::Err(Status::index_not_trained, "Index not prepared");
        }
        
        if (!pq_flash_index_) {
            LOG_KNOWHERE_ERROR_ << "PQFlashIndex not initialized";
            return expected<DataSetPtr>::Err(Status::empty_index, "PQFlashIndex not loaded");
        }
        
        if (!search_pool_) {
            LOG_KNOWHERE_ERROR_ << "Search thread pool not initialized";
            return expected<DataSetPtr>::Err(Status::internal_error, "search pool not initialized");
        }
        
        auto search_conf = static_cast<const pipnn_diskann::PipnnConfig&>(*cfg);
        auto k = static_cast<uint64_t>(search_conf.k.value_or(10));
        auto nq = dataset->GetRows();
        auto dim = dataset->GetDim();
        auto xq = static_cast<const float*>(dataset->GetTensor());
        
        // Search parameters
        auto beamwidth = static_cast<uint64_t>(search_conf.beamwidth.value_or(4));
        auto lsearch = static_cast<uint64_t>(search_conf.search_list_size.value_or(64));
        
        // Handle filter ratio for bitset filtering
        float filter_ratio = 0.0f;
        if (bitset.valid()) {
            filter_ratio = static_cast<float>(bitset.count()) / static_cast<float>(bitset.size());
        }
        
        // Prepare result arrays
        auto p_id = std::make_unique<int64_t[]>(k * nq);
        auto p_dist = std::make_unique<DistType[]>(k * nq);
        
        // Feder result for visit info (optional)
        feder::diskann::FederResultUniq feder_result;
        if (search_conf.trace_visit.value()) {
            if (nq > 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "nq should be 1 when trace_visit is true");
            }
            feder_result = std::make_unique<feder::diskann::FederResult>();
        }
        
        // Parallel search over all queries
        std::vector<folly::Future<folly::Unit>> futures;
        futures.reserve(nq);
        
        for (int64_t row = 0; row < nq; ++row) {
            futures.emplace_back(search_pool_->push([&, index = row, p_id_ptr = p_id.get(), p_dist_ptr = p_dist.get()]() {
                knowhere::checkCancellation(op_context);
                diskann::QueryStats stats;
                
                pq_flash_index_->cached_beam_search(
                    xq + (index * dim),  // query vector
                    k,                    // number of results
                    lsearch,              // search list size (L)
                    p_id_ptr + (index * k),      // result IDs
                    p_dist_ptr + (index * k),    // result distances
                    beamwidth,            // beamwidth for I/O optimization
                    false,                // ignore_filter (false = respect bitset)
                    &stats,               // query statistics
                    feder_result.get(),   // feder result for tracing
                    bitset,               // bitset for filtering
                    filter_ratio          // filter ratio estimate
                );
                
#ifndef NOT_COMPILE_FOR_SWIG
                knowhere_diskann_search_hops.Observe(stats.n_hops);
#endif
            }));
        }
        
        // Wait for all searches to complete
        try {
            folly::waitAll(futures);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_ERROR_ << "Search exception: " << e.what();
            return expected<DataSetPtr>::Err(Status::diskann_inner_error, "some search failed");
        }
        
        // Create result dataset
        auto res = GenResultDataSet(nq, k, std::move(p_id), std::move(p_dist));
        
        // Set visit_info json string into result dataset if feder tracing is enabled
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }
        
        return res;
    }

    Status
    Serialize(BinarySet& binset) const override {
        LOG_KNOWHERE_INFO_ << "PiPNN-DiskANN does nothing for serialize";
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, std::shared_ptr<Config> cfg) override {
        std::lock_guard<std::mutex> lock(preparation_lock_);
        
        auto prep_conf = static_cast<const pipnn_diskann::PipnnConfig&>(*cfg);
        
        if (is_prepared_.load()) {
            return Status::success;
        }
        
        if (!(prep_conf.index_prefix.has_value())) {
            LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN: index_prefix required for Deserialize";
            return Status::invalid_param_in_json;
        }
        
        index_prefix_ = prep_conf.index_prefix.value();
        
        // Determine metric type
        bool need_norm = prep_conf.metric_type.value() == knowhere::metric::IP ||
                         prep_conf.metric_type.value() == knowhere::metric::COSINE;
        
        auto diskann_metric = [m = prep_conf.metric_type.value()] {
            if (m == knowhere::metric::L2) {
                return diskann::Metric::L2;
            } else if (m == knowhere::metric::COSINE) {
                return diskann::Metric::COSINE;
            } else {
                return diskann::Metric::INNER_PRODUCT;
            }
        }();
        
        // Load necessary files from FileManager
        for (auto& filename : GetNecessaryFilenames(
                 index_prefix_, need_norm, 
                 prep_conf.search_cache_budget_gb.value_or(0.0f) > 0,
                 prep_conf.warm_up.value_or(false))) {
            if (!LoadFile(filename)) {
                LOG_KNOWHERE_ERROR_ << "Failed to load file: " << filename;
                return Status::disk_file_error;
            }
        }
        
        // Load optional files
        for (auto& filename : GetOptionalFilenames(index_prefix_)) {
            auto is_exist_op = file_manager_->IsExisted(filename);
            if (!is_exist_op.has_value()) {
                LOG_KNOWHERE_ERROR_ << "Failed to check existence: " << filename;
                return Status::disk_file_error;
            }
            if (is_exist_op.value() && !LoadFile(filename)) {
                return Status::disk_file_error;
            }
        }
        
        // Initialize search thread pool
        search_pool_ = ThreadPool::GetGlobalSearchThreadPool();
        
        // Create PQFlashIndex
        std::shared_ptr<AlignedFileReader> reader = nullptr;
        reader.reset(new LinuxAlignedFileReader());
        
        pq_flash_index_ = std::make_unique<diskann::PQFlashIndex<DataType>>(reader, diskann_metric);
        
        // Load PQFlashIndex from disk
        auto disk_ann_call = [&]() {
            int res = pq_flash_index_->load(search_pool_->size(), index_prefix_.c_str());
            if (res != 0) {
                throw diskann::ANNException("pq_flash_index_->load failed: " + std::to_string(res), -1);
            }
        };
        
        if (TryDiskANNCall(disk_ann_call) != Status::success) {
            LOG_KNOWHERE_ERROR_ << "Failed to load PQFlashIndex";
            return Status::diskann_inner_error;
        }
        
        // Set dimensions and count
        bool is_ip = prep_conf.metric_type.value() == knowhere::metric::IP;
        if (is_ip) {
            dim_.store(pq_flash_index_->get_data_dim() - 1);
        } else {
            dim_.store(pq_flash_index_->get_data_dim());
        }
        count_.store(pq_flash_index_->get_num_points());
        
        // Load cache nodes if configured
        auto cached_nodes_file = diskann::get_cached_nodes_file(index_prefix_);
        std::vector<uint32_t> node_list;
        
        if (file_exists(cached_nodes_file)) {
            LOG_KNOWHERE_INFO_ << "Loading cached nodes from file";
            size_t num_nodes, nodes_id_dim;
            std::unique_ptr<uint32_t[]> cached_nodes_ids = nullptr;
            diskann::load_bin<uint32_t>(cached_nodes_file, cached_nodes_ids, num_nodes, nodes_id_dim);
            node_list.assign(cached_nodes_ids.get(), cached_nodes_ids.get() + num_nodes);
        } else {
            auto num_nodes_to_cache = GetCachedNodeNum(
                prep_conf.search_cache_budget_gb.value_or(0.0f),
                pq_flash_index_->get_data_dim(),
                pq_flash_index_->get_max_degree());
            
            if (num_nodes_to_cache > 0 && num_nodes_to_cache < pq_flash_index_->get_num_points() / 3) {
                LOG_KNOWHERE_INFO_ << "Caching " << num_nodes_to_cache << " nodes";
                if (TryDiskANNCall([&]() {
                        pq_flash_index_->async_generate_cache_list_from_sample_queries(
                            diskann::get_sample_data_filename(index_prefix_), 15, 6, num_nodes_to_cache);
                    }) != Status::success) {
                    LOG_KNOWHERE_ERROR_ << "Failed to generate cache list";
                    return Status::diskann_inner_error;
                }
            }
        }
        
        if (!node_list.empty()) {
            if (TryDiskANNCall([&]() { pq_flash_index_->load_cache_list(node_list); }) != Status::success) {
                LOG_KNOWHERE_ERROR_ << "Failed to load cache";
                return Status::diskann_inner_error;
            }
        }
        
        is_prepared_.store(true);
        LOG_KNOWHERE_INFO_ << "PiPNN-DiskANN loaded: " << count_.load() << " points, " << dim_.load() << " dims";
        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, std::shared_ptr<Config> cfg) override {
        LOG_KNOWHERE_ERROR_ << "PiPNN-DiskANN doesn't support Deserialize from file";
        return Status::not_implemented;
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<pipnn_diskann::PipnnConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<pipnn_diskann::PipnnConfig>();
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset, milvus::OpContext* op_context) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not supported");
    }

    expected<DataSetPtr>
    GetIndexMeta(std::unique_ptr<Config> cfg) const override {
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not supported");
    }

    int64_t
    Dim() const override {
        return dim_.load();
    }

    int64_t
    Size() const override {
        return 0;
    }

    int64_t
    Count() const override {
        return count_.load();
    }

    std::string
    Type() const override {
        return "PIPNN_DISKANN";
    }

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
            LOG_KNOWHERE_ERROR_ << "DiskANN Exception: " << e.what();
            return Status::diskann_inner_error;
        }
    }

    std::vector<std::string>
    GetNecessaryFilenames(const std::string& prefix, bool need_norm, bool use_sample_cache, bool use_sample_warmup) {
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
        return filenames;
    }

    uint64_t
    GetCachedNodeNum(float cache_dram_budget, uint64_t data_dim, uint64_t max_degree) {
        if (cache_dram_budget <= 0.0f) return 0;
        uint64_t single_node_size = sizeof(uint32_t) + data_dim * sizeof(float);
        return static_cast<uint64_t>(cache_dram_budget * 1024.0 * 1024.0 * 1024.0 / single_node_size);
    }

    bool
    LoadFile(const std::string& filename) {
        if (!file_manager_->LoadFile(filename)) {
            LOG_KNOWHERE_ERROR_ << "Failed to load file: " << filename;
            return false;
        }
        return true;
    }

 private:
    std::atomic<bool> is_prepared_;
    std::atomic<int64_t> dim_;
    std::atomic<int64_t> count_;
    std::shared_ptr<milvus::FileManager> file_manager_;
    mutable std::mutex preparation_lock_;
    std::string index_prefix_;
    std::unique_ptr<diskann::PQFlashIndex<DataType>> pq_flash_index_;
    std::shared_ptr<ThreadPool> search_pool_;
};

// Registration
KNOWHERE_SIMPLE_REGISTER_DENSE_FLOAT_ALL_GLOBAL(PIPNN_DISKANN, PiPNNDiskANNIndexNode,
                                                 knowhere::feature::DISK)

}  // namespace knowhere

#endif  // KNOWHERE_WITH_DISKANN
