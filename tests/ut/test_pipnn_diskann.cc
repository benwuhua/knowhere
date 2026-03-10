// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/impl/pipnn_build_profile.h"
#include "index/diskann/impl/hash_prune.h"
#include "index/diskann/impl/pipnn_builder.h"
#include "index/diskann/impl/pipnn_diskann_config.h"
#include "index/diskann/impl/rbc_partition.h"
#include "index/diskann/impl/vamana_serializer.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "knowhere/version.h"
#include "utils.h"

namespace {

using knowhere::pipnn_diskann::HashPrune;
using knowhere::pipnn_diskann::BuildProfile;
using knowhere::pipnn_diskann::BuildStage;
using knowhere::pipnn_diskann::BuildStageName;
using knowhere::pipnn_diskann::NsToMs;
using knowhere::pipnn_diskann::PiPNNBuilder;
using knowhere::pipnn_diskann::PipnnConfig;
using knowhere::pipnn_diskann::RBCPartitioner;
namespace fs = std::filesystem;

constexpr uint32_t kDim = 32;
constexpr uint32_t kHashBits = 12;
constexpr uint32_t kMaxDegree = 16;
constexpr uint32_t kE2ENumRows = 1000000;
constexpr uint32_t kE2EDim = 128;
constexpr uint32_t kE2ENumQueries = 100;
constexpr uint32_t kE2EK = 10;

struct PiPNNRecallTuningConfig {
    std::string label = "baseline";
    uint32_t leaf_max_size = 1000;
    uint32_t fanout_l1 = 10;
    uint32_t fanout_l2 = 3;
    uint32_t overlap_k = 2;
    uint32_t k_nn = 3;
    uint32_t hash_bits = 12;
    bool final_prune = true;
    bool enable_cross_leaf_union = false;
    bool enable_retained_degree_limit = true;
    bool enable_boundary_leader_bridge = false;
};

std::vector<float>
RandVec(uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(rng);
    }
    return v;
}

float
L2Dist(const float* a, const float* b, uint32_t d) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < d; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

struct ScopedTempDir {
    explicit ScopedTempDir(const std::string& prefix) {
        const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        dir = fs::temp_directory_path() / (prefix + "_" + std::to_string(now));
        std::error_code ec;
        fs::remove_all(dir, ec);
        REQUIRE(fs::create_directories(dir));
    }

    ~ScopedTempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    fs::path dir;
};

struct ScopedEnvVar {
    ScopedEnvVar(const char* key, const char* value) : key_(key) {
        const char* current = std::getenv(key_);
        if (current != nullptr) {
            had_original_ = true;
            original_value_ = current;
        }

        if (value != nullptr) {
            REQUIRE(setenv(key_, value, 1) == 0);
        } else {
            REQUIRE(unsetenv(key_) == 0);
        }
    }

    ~ScopedEnvVar() {
        if (had_original_) {
            setenv(key_, original_value_.c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }

    const char* key_;
    bool had_original_ = false;
    std::string original_value_;
};

knowhere::Json
MakeBaseJson(const uint32_t dim, const uint32_t k) {
    knowhere::Json json;
    json["dim"] = dim;
    json["metric_type"] = knowhere::metric::L2;
    json["k"] = k;
    return json;
}

knowhere::Json
MakeBuildJson(const std::string& index_prefix, const std::string& data_path, const uint32_t dim, const uint32_t rows) {
    auto json = MakeBaseJson(dim, kE2EK);
    json["index_prefix"] = index_prefix;
    json["data_path"] = data_path;
    json["max_degree"] = 32;
    json["search_list_size"] = 100;
    json["pq_code_budget_gb"] = sizeof(float) * dim * rows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    json["search_cache_budget_gb"] = sizeof(float) * dim * rows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    json["build_dram_budget_gb"] = 8.0;
    return json;
}

uint32_t
GetEnvUintOrDefault(const char* key, uint32_t default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    char* end = nullptr;
    const auto parsed = std::strtoul(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        INFO("Ignoring invalid env override for " << key << ": " << raw);
        return default_value;
    }
    return static_cast<uint32_t>(parsed);
}

bool
GetEnvBoolOrDefault(const char* key, bool default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return default_value;
    }

    if (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 || std::strcmp(raw, "TRUE") == 0) {
        return true;
    }
    if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 || std::strcmp(raw, "FALSE") == 0) {
        return false;
    }

    INFO("Ignoring invalid env override for " << key << ": " << raw);
    return default_value;
}

std::optional<std::string>
GetEnvString(const char* key) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return std::nullopt;
    }
    return std::string(raw);
}

struct ExternalRecallDatasetPaths {
    std::string base_fbin;
    std::string query_fbin;
    std::string gt_ibin;
    std::string label;
};

std::optional<ExternalRecallDatasetPaths>
LoadExternalRecallDatasetPathsFromEnv() {
    const auto base_fbin = GetEnvString("KNOWHERE_PIPNN_BASE_FBIN");
    const auto query_fbin = GetEnvString("KNOWHERE_PIPNN_QUERY_FBIN");
    const auto gt_ibin = GetEnvString("KNOWHERE_PIPNN_GT_IBIN");
    const auto dataset_label = GetEnvString("KNOWHERE_PIPNN_DATASET_LABEL");
    if (!base_fbin.has_value() && !query_fbin.has_value() && !gt_ibin.has_value()) {
        return std::nullopt;
    }

    REQUIRE(base_fbin.has_value());
    REQUIRE(query_fbin.has_value());
    REQUIRE(gt_ibin.has_value());
    return ExternalRecallDatasetPaths{*base_fbin, *query_fbin, *gt_ibin,
                                      dataset_label.value_or("external_public_dataset")};
}

template <typename T>
std::vector<T>
LoadTypedBinWithHeader(const std::string& path, uint32_t& rows, uint32_t& dim) {
    std::ifstream input(path, std::ios::binary);
    REQUIRE(input.is_open());
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    REQUIRE(input.good());

    std::vector<T> values(static_cast<size_t>(rows) * dim);
    if (!values.empty()) {
        input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
        REQUIRE(input.good());
    }
    return values;
}

std::vector<int64_t>
LoadGroundTruthIbin(const std::string& path, uint32_t& rows, uint32_t& topk) {
    uint32_t raw_rows = 0;
    uint32_t raw_topk = 0;
    const auto raw_ids = LoadTypedBinWithHeader<uint32_t>(path, raw_rows, raw_topk);
    rows = raw_rows;
    topk = raw_topk;
    std::vector<int64_t> ids(raw_ids.begin(), raw_ids.end());
    return ids;
}

PiPNNRecallTuningConfig
LoadPiPNNRecallTuningFromEnv() {
    PiPNNRecallTuningConfig config;
    config.leaf_max_size = GetEnvUintOrDefault("KNOWHERE_PIPNN_LEAF_MAX_SIZE", config.leaf_max_size);
    config.fanout_l1 = GetEnvUintOrDefault("KNOWHERE_PIPNN_FANOUT_L1", config.fanout_l1);
    config.fanout_l2 = GetEnvUintOrDefault("KNOWHERE_PIPNN_FANOUT_L2", config.fanout_l2);
    config.overlap_k = GetEnvUintOrDefault("KNOWHERE_PIPNN_OVERLAP_K", config.overlap_k);
    config.k_nn = GetEnvUintOrDefault("KNOWHERE_PIPNN_K_NN", config.k_nn);
    config.hash_bits = GetEnvUintOrDefault("KNOWHERE_PIPNN_HASH_BITS", config.hash_bits);
    config.final_prune = GetEnvBoolOrDefault("KNOWHERE_PIPNN_FINAL_PRUNE", config.final_prune);
    config.enable_cross_leaf_union =
        GetEnvBoolOrDefault("KNOWHERE_PIPNN_ENABLE_CROSS_LEAF_UNION", config.enable_cross_leaf_union);
    config.enable_retained_degree_limit =
        GetEnvBoolOrDefault("KNOWHERE_PIPNN_ENABLE_RETAINED_DEGREE_LIMIT", config.enable_retained_degree_limit);
    config.enable_boundary_leader_bridge =
        GetEnvBoolOrDefault("KNOWHERE_PIPNN_ENABLE_BOUNDARY_LEADER_BRIDGE", config.enable_boundary_leader_bridge);
    return config;
}

void
ApplyPiPNNRecallTuning(knowhere::Json& build_json, const PiPNNRecallTuningConfig& config) {
    build_json["pipnn_leaf_max_size"] = config.leaf_max_size;
    build_json["pipnn_fanout_l1"] = config.fanout_l1;
    build_json["pipnn_fanout_l2"] = config.fanout_l2;
    build_json["pipnn_overlap_k"] = config.overlap_k;
    build_json["pipnn_k_nn"] = config.k_nn;
    build_json["pipnn_hash_bits"] = config.hash_bits;
    build_json["pipnn_final_prune"] = config.final_prune;
    build_json["pipnn_enable_cross_leaf_union"] = config.enable_cross_leaf_union;
}

std::string
DescribePiPNNRecallTuning(const PiPNNRecallTuningConfig& config) {
    return "label=" + config.label + ", leaf_max_size=" + std::to_string(config.leaf_max_size) + ", fanout_l1=" +
           std::to_string(config.fanout_l1) + ", fanout_l2=" + std::to_string(config.fanout_l2) +
           ", overlap_k=" + std::to_string(config.overlap_k) + ", k_nn=" + std::to_string(config.k_nn) +
           ", hash_bits=" + std::to_string(config.hash_bits) +
           ", final_prune=" + (config.final_prune ? "true" : "false") +
           ", cross_leaf_union=" + (config.enable_cross_leaf_union ? "true" : "false") +
           ", retained_degree_limit=" + (config.enable_retained_degree_limit ? "true" : "false") +
           ", boundary_leader_bridge=" + (config.enable_boundary_leader_bridge ? "true" : "false");
}

struct SerializedVamanaGraph {
    uint64_t index_size = 0;
    uint32_t max_degree = 0;
    uint32_t entry_point = 0;
    uint64_t num_frozen_pts = 0;
    std::vector<std::vector<uint32_t>> adjacency;
};

struct GraphSearchStats {
    double avg_degree = 0.0;
    uint32_t min_degree = 0;
    uint32_t max_degree = 0;
    size_t zero_degree_nodes = 0;
    size_t oversized_nodes = 0;
    size_t invalid_edges = 0;
    size_t asym_edges = 0;
    uint32_t entry_point_degree = 0;
    size_t degree_le_8 = 0;
    size_t degree_9_16 = 0;
    size_t degree_17_24 = 0;
    size_t degree_25_31 = 0;
    size_t degree_eq_max = 0;
};

struct LocalGraphRecallStats {
    float avg_recall = 0.0f;
    float min_recall = 0.0f;
    float max_recall = 0.0f;
    size_t zero_hit_queries = 0;
};

struct GraphNeighborhoodQualityStats {
    double avg_neighbor_recall = 0.0;
    double min_neighbor_recall = 0.0;
    double max_neighbor_recall = 0.0;
    double avg_neighbor_precision = 0.0;
    double min_neighbor_precision = 0.0;
    double max_neighbor_precision = 0.0;
    double avg_entry_neighbor_recall = 0.0;
    double avg_entry_neighbor_precision = 0.0;
    size_t zero_neighbor_recall_nodes = 0;
    size_t evaluated_nodes = 0;
    size_t entry_overlap_nodes = 0;
    size_t candidate_pool_size = 0;
};

struct GraphPairOverlapStats {
    double avg_neighbor_overlap = 0.0;
    double min_neighbor_overlap = 0.0;
    double max_neighbor_overlap = 0.0;
    double entry_neighbor_overlap = 0.0;
    size_t entry_overlap_nodes = 0;
    size_t zero_neighbor_overlap_nodes = 0;
    size_t evaluated_nodes = 0;
    bool entry_point_matches = false;
};

SerializedVamanaGraph
LoadSerializedVamanaGraph(const std::string& path, const uint32_t expected_nodes) {
    fs::path resolved_path(path);
    if (!fs::exists(resolved_path)) {
        const auto parent = resolved_path.parent_path();
        if (fs::exists(parent)) {
            for (const auto& entry : fs::directory_iterator(parent)) {
                if (entry.is_regular_file() && entry.path().extension() == ".index" &&
                    entry.path().filename().string().find("_mem.index") != std::string::npos) {
                    resolved_path = entry.path();
                    break;
                }
            }
        }
    }
    INFO("LoadSerializedVamanaGraph path=" << path << ", resolved_path=" << resolved_path.string());
    std::ifstream input(resolved_path, std::ios::binary);
    REQUIRE(input.is_open());

    SerializedVamanaGraph graph;
    input.read(reinterpret_cast<char*>(&graph.index_size), sizeof(graph.index_size));
    input.read(reinterpret_cast<char*>(&graph.max_degree), sizeof(graph.max_degree));
    input.read(reinterpret_cast<char*>(&graph.entry_point), sizeof(graph.entry_point));
    input.read(reinterpret_cast<char*>(&graph.num_frozen_pts), sizeof(graph.num_frozen_pts));
    REQUIRE(input.good());

    graph.adjacency.resize(expected_nodes);
    for (uint32_t node = 0; node < expected_nodes; ++node) {
        uint32_t degree = 0;
        input.read(reinterpret_cast<char*>(&degree), sizeof(degree));
        REQUIRE(input.good());
        graph.adjacency[node].resize(degree);
        if (degree > 0) {
            input.read(reinterpret_cast<char*>(graph.adjacency[node].data()),
                       static_cast<std::streamsize>(degree * sizeof(uint32_t)));
            REQUIRE(input.good());
        }
    }

    return graph;
}

GraphSearchStats
CollectGraphSearchStats(const SerializedVamanaGraph& graph) {
    GraphSearchStats stats;
    stats.min_degree = graph.adjacency.empty() ? 0 : std::numeric_limits<uint32_t>::max();

    uint64_t total_degree = 0;
    for (uint32_t node = 0; node < graph.adjacency.size(); ++node) {
        const auto& neighbors = graph.adjacency[node];
        const auto degree = static_cast<uint32_t>(neighbors.size());
        total_degree += degree;
        stats.min_degree = std::min(stats.min_degree, degree);
        stats.max_degree = std::max(stats.max_degree, degree);
        if (degree == 0) {
            ++stats.zero_degree_nodes;
        }
        if (degree <= 8) {
            ++stats.degree_le_8;
        } else if (degree <= 16) {
            ++stats.degree_9_16;
        } else if (degree <= 24) {
            ++stats.degree_17_24;
        } else if (degree < graph.max_degree) {
            ++stats.degree_25_31;
        } else if (degree == graph.max_degree) {
            ++stats.degree_eq_max;
        }
        if (degree > graph.max_degree) {
            ++stats.oversized_nodes;
        }
        for (const auto neighbor : neighbors) {
            if (neighbor >= graph.adjacency.size()) {
                ++stats.invalid_edges;
                continue;
            }
            const auto& reverse = graph.adjacency[neighbor];
            if (std::find(reverse.begin(), reverse.end(), node) == reverse.end()) {
                ++stats.asym_edges;
            }
        }
    }

    stats.avg_degree = graph.adjacency.empty() ? 0.0 : static_cast<double>(total_degree) / graph.adjacency.size();
    if (graph.entry_point < graph.adjacency.size()) {
        stats.entry_point_degree = static_cast<uint32_t>(graph.adjacency[graph.entry_point].size());
    }
    return stats;
}

std::vector<int64_t>
SearchSerializedGraphTopK(const float* base, const uint32_t rows, const uint32_t dim, const float* query,
                          const SerializedVamanaGraph& graph, const uint32_t topk, const uint32_t search_list_size) {
    using Candidate = std::pair<float, uint32_t>;
    struct CandidateGreater {
        bool
        operator()(const Candidate& lhs, const Candidate& rhs) const {
            return lhs.first > rhs.first;
        }
    };
    struct ResultWorse {
        bool
        operator()(const Candidate& lhs, const Candidate& rhs) const {
            return lhs.first < rhs.first;
        }
    };

    REQUIRE(graph.entry_point < rows);
    std::priority_queue<Candidate, std::vector<Candidate>, CandidateGreater> candidates;
    std::priority_queue<Candidate, std::vector<Candidate>, ResultWorse> best;
    std::vector<uint8_t> visited(rows, 0);

    const auto push_node = [&](const uint32_t node_id) {
        const float dist = L2Dist(base + static_cast<size_t>(node_id) * dim, query, dim);
        candidates.emplace(dist, node_id);
        best.emplace(dist, node_id);
        visited[node_id] = 1;
    };

    push_node(graph.entry_point);
    while (!candidates.empty()) {
        const auto [cur_dist, cur] = candidates.top();
        candidates.pop();

        if (best.size() >= search_list_size && cur_dist > best.top().first) {
            break;
        }

        for (const auto neighbor : graph.adjacency[cur]) {
            if (neighbor >= rows || visited[neighbor]) {
                continue;
            }
            visited[neighbor] = 1;
            const float dist = L2Dist(base + static_cast<size_t>(neighbor) * dim, query, dim);
            if (best.size() < search_list_size || dist < best.top().first) {
                candidates.emplace(dist, neighbor);
                best.emplace(dist, neighbor);
                if (best.size() > search_list_size) {
                    best.pop();
                }
            }
        }
    }

    std::vector<Candidate> ordered;
    ordered.reserve(best.size());
    while (!best.empty()) {
        ordered.push_back(best.top());
        best.pop();
    }
    std::sort(ordered.begin(), ordered.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    std::vector<int64_t> topk_ids;
    for (size_t i = 0; i < ordered.size() && i < topk; ++i) {
        topk_ids.push_back(static_cast<int64_t>(ordered[i].second));
    }
    while (topk_ids.size() < topk) {
        topk_ids.push_back(-1);
    }
    return topk_ids;
}

std::vector<std::vector<int64_t>>
SearchSerializedGraphBatch(const knowhere::DataSet& base, const knowhere::DataSet& queries, const uint32_t topk,
                           const SerializedVamanaGraph& graph, const uint32_t search_list_size) {
    auto* base_tensor = static_cast<const float*>(base.GetTensor());
    auto* query_tensor = static_cast<const float*>(queries.GetTensor());
    const auto rows = static_cast<uint32_t>(base.GetRows());
    const auto nq = static_cast<uint32_t>(queries.GetRows());
    const auto dim = static_cast<uint32_t>(base.GetDim());

    std::vector<std::vector<int64_t>> results;
    results.reserve(nq);
    for (uint32_t i = 0; i < nq; ++i) {
        results.push_back(SearchSerializedGraphTopK(base_tensor,
                                                    rows,
                                                    dim,
                                                    query_tensor + static_cast<size_t>(i) * dim,
                                                    graph,
                                                    topk,
                                                    search_list_size));
    }
    return results;
}

LocalGraphRecallStats
CollectLocalGraphRecallStats(const std::vector<std::vector<int64_t>>& gt,
                             const std::vector<std::vector<int64_t>>& results,
                             const size_t topk) {
    REQUIRE(gt.size() == results.size());
    LocalGraphRecallStats stats;
    stats.min_recall = gt.empty() ? 0.0f : 1.0f;

    for (size_t qi = 0; qi < gt.size(); ++qi) {
        size_t hits = 0;
        for (size_t ri = 0; ri < topk && ri < results[qi].size(); ++ri) {
            const auto id = results[qi][ri];
            if (id == -1) {
                continue;
            }
            hits += std::find(gt[qi].begin(), gt[qi].begin() + std::min(topk, gt[qi].size()), id) !=
                            gt[qi].begin() + std::min(topk, gt[qi].size())
                        ? 1
                        : 0;
        }
        const float recall = topk == 0 ? 0.0f : static_cast<float>(hits) / static_cast<float>(topk);
        stats.avg_recall += recall;
        stats.min_recall = std::min(stats.min_recall, recall);
        stats.max_recall = std::max(stats.max_recall, recall);
        if (hits == 0) {
            ++stats.zero_hit_queries;
        }
    }

    if (!gt.empty()) {
        stats.avg_recall /= static_cast<float>(gt.size());
    }
    return stats;
}

GraphNeighborhoodQualityStats
CollectGraphNeighborhoodQualityStats(const knowhere::DataSet& base,
                                     const SerializedVamanaGraph& graph,
                                     const uint32_t topk) {
    constexpr uint32_t kProbeNodeLimit = 16;
    constexpr uint32_t kCandidatePoolLimit = 20000;

    GraphNeighborhoodQualityStats stats;
    if (topk == 0 || graph.adjacency.empty()) {
        return stats;
    }

    const auto rows = static_cast<uint32_t>(base.GetRows());
    const auto dim = static_cast<uint32_t>(base.GetDim());
    const auto* base_tensor = static_cast<const float*>(base.GetTensor());
    REQUIRE(rows == graph.adjacency.size());

    std::vector<uint32_t> candidate_pool;
    candidate_pool.reserve(std::min<uint32_t>(rows, kCandidatePoolLimit));
    const uint32_t pool_stride = std::max<uint32_t>(1, rows / std::max<uint32_t>(1, kCandidatePoolLimit));
    for (uint32_t node = 0; node < rows && candidate_pool.size() < kCandidatePoolLimit; node += pool_stride) {
        candidate_pool.push_back(node);
    }
    if (candidate_pool.empty()) {
        candidate_pool.push_back(0);
    }
    stats.candidate_pool_size = candidate_pool.size();

    std::vector<uint32_t> probe_nodes;
    probe_nodes.reserve(kProbeNodeLimit + 1);
    if (graph.entry_point < rows) {
        probe_nodes.push_back(graph.entry_point);
    }
    const uint32_t probe_stride = std::max<uint32_t>(1, rows / std::max<uint32_t>(1, kProbeNodeLimit));
    for (uint32_t node = 0; node < rows && probe_nodes.size() < kProbeNodeLimit; node += probe_stride) {
        if (std::find(probe_nodes.begin(), probe_nodes.end(), node) == probe_nodes.end()) {
            probe_nodes.push_back(node);
        }
    }

    auto collect_truth = [&](const uint32_t node_id) {
        std::vector<std::pair<float, uint32_t>> dists;
        dists.reserve(candidate_pool.size());
        const auto* query = base_tensor + static_cast<size_t>(node_id) * dim;
        for (const auto other : candidate_pool) {
            if (other == node_id) {
                continue;
            }
            dists.emplace_back(L2Dist(query, base_tensor + static_cast<size_t>(other) * dim, dim), other);
        }
        const auto effective_topk = std::min<size_t>(topk, dists.size());
        if (effective_topk == 0) {
            return std::vector<uint32_t>{};
        }
        if (dists.size() > effective_topk) {
            std::nth_element(dists.begin(), dists.begin() + effective_topk, dists.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            dists.resize(effective_topk);
        }
        std::sort(dists.begin(), dists.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        std::vector<uint32_t> truth;
        truth.reserve(effective_topk);
        for (const auto& [_, id] : dists) {
            truth.push_back(id);
        }
        return truth;
    };

    stats.min_neighbor_recall = 1.0;
    stats.min_neighbor_precision = 1.0;
    for (const auto node : probe_nodes) {
        const auto truth = collect_truth(node);
        const auto truth_k = truth.size();
        if (truth_k == 0) {
            continue;
        }

        size_t hits = 0;
        const auto& neighbors = graph.adjacency[node];
        const auto eval_neighbors = std::min<size_t>(neighbors.size(), topk);
        for (size_t i = 0; i < eval_neighbors; ++i) {
            hits += std::find(truth.begin(), truth.end(), neighbors[i]) != truth.end() ? 1 : 0;
        }

        const double recall = static_cast<double>(hits) / static_cast<double>(truth_k);
        const double precision = eval_neighbors == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(eval_neighbors);
        stats.avg_neighbor_recall += recall;
        stats.avg_neighbor_precision += precision;
        stats.min_neighbor_recall = std::min(stats.min_neighbor_recall, recall);
        stats.max_neighbor_recall = std::max(stats.max_neighbor_recall, recall);
        stats.min_neighbor_precision = std::min(stats.min_neighbor_precision, precision);
        stats.max_neighbor_precision = std::max(stats.max_neighbor_precision, precision);
        stats.zero_neighbor_recall_nodes += hits == 0 ? 1 : 0;
        ++stats.evaluated_nodes;

        if (node == graph.entry_point) {
            stats.avg_entry_neighbor_recall = recall;
            stats.avg_entry_neighbor_precision = precision;
            stats.entry_overlap_nodes = hits;
        }
    }

    if (stats.evaluated_nodes == 0) {
        stats.min_neighbor_recall = 0.0;
        stats.min_neighbor_precision = 0.0;
        return stats;
    }

    stats.avg_neighbor_recall /= static_cast<double>(stats.evaluated_nodes);
    stats.avg_neighbor_precision /= static_cast<double>(stats.evaluated_nodes);
    return stats;
}

GraphPairOverlapStats
CollectGraphPairOverlapStats(const SerializedVamanaGraph& lhs, const SerializedVamanaGraph& rhs, const uint32_t topk) {
    constexpr uint32_t kProbeNodeLimit = 16;

    GraphPairOverlapStats stats;
    if (topk == 0 || lhs.adjacency.empty() || rhs.adjacency.empty()) {
        return stats;
    }

    REQUIRE(lhs.adjacency.size() == rhs.adjacency.size());
    const uint32_t rows = static_cast<uint32_t>(lhs.adjacency.size());
    stats.entry_point_matches = lhs.entry_point == rhs.entry_point;

    std::vector<uint32_t> probe_nodes;
    probe_nodes.reserve(kProbeNodeLimit + 2);
    if (lhs.entry_point < rows) {
        probe_nodes.push_back(lhs.entry_point);
    }
    if (rhs.entry_point < rows && std::find(probe_nodes.begin(), probe_nodes.end(), rhs.entry_point) == probe_nodes.end()) {
        probe_nodes.push_back(rhs.entry_point);
    }
    const uint32_t probe_stride = std::max<uint32_t>(1, rows / std::max<uint32_t>(1, kProbeNodeLimit));
    for (uint32_t node = 0; node < rows && probe_nodes.size() < kProbeNodeLimit; node += probe_stride) {
        if (std::find(probe_nodes.begin(), probe_nodes.end(), node) == probe_nodes.end()) {
            probe_nodes.push_back(node);
        }
    }

    stats.min_neighbor_overlap = 1.0;
    for (const auto node : probe_nodes) {
        const auto& lhs_neighbors = lhs.adjacency[node];
        const auto& rhs_neighbors = rhs.adjacency[node];
        const auto eval_lhs = std::min<size_t>(lhs_neighbors.size(), topk);
        const auto eval_rhs = std::min<size_t>(rhs_neighbors.size(), topk);
        const auto denom = std::max<size_t>(1, std::min(eval_lhs, eval_rhs));

        size_t hits = 0;
        for (size_t i = 0; i < eval_lhs; ++i) {
            hits += std::find(rhs_neighbors.begin(), rhs_neighbors.begin() + eval_rhs, lhs_neighbors[i]) !=
                            rhs_neighbors.begin() + eval_rhs
                        ? 1
                        : 0;
        }

        const double overlap = static_cast<double>(hits) / static_cast<double>(denom);
        stats.avg_neighbor_overlap += overlap;
        stats.min_neighbor_overlap = std::min(stats.min_neighbor_overlap, overlap);
        stats.max_neighbor_overlap = std::max(stats.max_neighbor_overlap, overlap);
        stats.zero_neighbor_overlap_nodes += hits == 0 ? 1 : 0;
        ++stats.evaluated_nodes;

        if (node == lhs.entry_point || node == rhs.entry_point) {
            stats.entry_neighbor_overlap = std::max(stats.entry_neighbor_overlap, overlap);
            stats.entry_overlap_nodes = std::max(stats.entry_overlap_nodes, hits);
        }
    }

    if (stats.evaluated_nodes == 0) {
        stats.min_neighbor_overlap = 0.0;
        return stats;
    }

    stats.avg_neighbor_overlap /= static_cast<double>(stats.evaluated_nodes);
    return stats;
}

knowhere::Json
MakeDeserializeJson(const std::string& index_prefix, const uint32_t dim, const uint32_t rows) {
    auto json = MakeBaseJson(dim, kE2EK);
    json["index_prefix"] = index_prefix;
    json["search_cache_budget_gb"] = sizeof(float) * dim * rows * 0.125 / (1024.0 * 1024.0 * 1024.0);
    return json;
}

knowhere::Json
MakeSearchJson(const uint32_t dim, const uint32_t k) {
    auto json = MakeBaseJson(dim, k);
    json["search_list_size"] = 64;
    json["beamwidth"] = 8;
    return json;
}

}  // namespace

TEST_CASE("PipnnConfig defaults", "[pipnn_diskann]") {
    PipnnConfig config;
    REQUIRE(config.num_partitions == 32);
    REQUIRE(config.overlap_ratio == Catch::Approx(0.5f));
    REQUIRE(config.max_degree == 32);
    REQUIRE(config.hash_bits == 12);
    REQUIRE(config.use_medoid_entry);
}

TEST_CASE("PipnnConfig ToString", "[pipnn_diskann]") {
    PipnnConfig config;
    const std::string s = config.ToString();
    REQUIRE(s.find("partitions=32") != std::string::npos);
    REQUIRE(s.find("overlap=0.500000") != std::string::npos);
    REQUIRE(s.find("max_degree=32") != std::string::npos);
}

TEST_CASE("PiPNN build profile separates graph and PQ stages", "[pipnn][profile]") {
    BuildProfile profile;
    profile.graph_construction_ns = 125000000;
    profile.pq_and_disk_layout_ns = 875000000;

    REQUIRE(BuildStageName(BuildStage::kGraphConstruction) == "PiPNN graph construction");
    REQUIRE(BuildStageName(BuildStage::kPQAndDiskLayout) == "DiskANN PQ/disk layout");
    REQUIRE(BuildStageName(BuildStage::kTotal) == "PiPNN-DiskANN total build");
    REQUIRE(profile.total_ns() == 1000000000);
    REQUIRE(profile.stage_ms(BuildStage::kGraphConstruction) == Catch::Approx(125.0));
    REQUIRE(profile.stage_ms(BuildStage::kPQAndDiskLayout) == Catch::Approx(875.0));
    REQUIRE(profile.stage_ms(BuildStage::kTotal) == Catch::Approx(1000.0));
    REQUIRE(NsToMs(500000) == Catch::Approx(0.5));
}

TEST_CASE("PiPNN recall tuning populates build json", "[pipnn][perf]") {
    PiPNNRecallTuningConfig config;
    config.leaf_max_size = 768;
    config.fanout_l1 = 6;
    config.fanout_l2 = 2;
    config.overlap_k = 1;
    config.k_nn = 5;
    config.hash_bits = 10;
    config.final_prune = false;
    config.enable_cross_leaf_union = false;

    auto build_json = MakeBuildJson("/tmp/pipnn_perf", "/tmp/raw.bin", kE2EDim, 1000);
    ApplyPiPNNRecallTuning(build_json, config);

    REQUIRE(build_json["pipnn_leaf_max_size"] == config.leaf_max_size);
    REQUIRE(build_json["pipnn_fanout_l1"] == config.fanout_l1);
    REQUIRE(build_json["pipnn_fanout_l2"] == config.fanout_l2);
    REQUIRE(build_json["pipnn_overlap_k"] == config.overlap_k);
    REQUIRE(build_json["pipnn_k_nn"] == config.k_nn);
    REQUIRE(build_json["pipnn_hash_bits"] == config.hash_bits);
    REQUIRE(build_json["pipnn_final_prune"] == config.final_prune);
    REQUIRE(build_json["pipnn_enable_cross_leaf_union"] == config.enable_cross_leaf_union);
}

TEST_CASE("HashPrune enforces max_degree", "[pipnn][hash_prune]") {
    std::mt19937 rng(42);
    auto p = RandVec(kDim, rng);

    HashPrune prune(kDim, kHashBits, kMaxDegree);
    std::vector<float> sketch_p(kHashBits);
    prune.compute_sketch(p.data(), sketch_p.data());

    for (uint32_t i = 0; i < 128; ++i) {
        auto c = RandVec(kDim, rng);
        std::vector<float> sketch_c(kHashBits);
        prune.compute_sketch(c.data(), sketch_c.data());
        prune.insert(i, sketch_p.data(), sketch_c.data(), L2Dist(p.data(), c.data(), kDim));
    }

    REQUIRE(prune.size() <= kMaxDegree);
    REQUIRE(prune.neighbors().size() == prune.size());
}

TEST_CASE("HashPrune insertion order robustness", "[pipnn][hash_prune]") {
    std::mt19937 rng(123);
    auto p = RandVec(kDim, rng);

    std::vector<std::vector<float>> candidates(80);
    for (auto& c : candidates) {
        c = RandVec(kDim, rng);
    }

    auto run_once = [&](bool reverse) {
        HashPrune prune(kDim, kHashBits, kMaxDegree);
        std::vector<float> sketch_p(kHashBits);
        prune.compute_sketch(p.data(), sketch_p.data());

        if (reverse) {
            for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i) {
                std::vector<float> sketch_c(kHashBits);
                prune.compute_sketch(candidates[i].data(), sketch_c.data());
                prune.insert(static_cast<uint32_t>(i),
                             sketch_p.data(),
                             sketch_c.data(),
                             L2Dist(p.data(), candidates[i].data(), kDim));
            }
        } else {
            for (uint32_t i = 0; i < candidates.size(); ++i) {
                std::vector<float> sketch_c(kHashBits);
                prune.compute_sketch(candidates[i].data(), sketch_c.data());
                prune.insert(i,
                             sketch_p.data(),
                             sketch_c.data(),
                             L2Dist(p.data(), candidates[i].data(), kDim));
            }
        }
        auto ids = prune.neighbors();
        std::sort(ids.begin(), ids.end());
        return ids;
    };

    const auto forward = run_once(false);
    const auto backward = run_once(true);

    REQUIRE(forward.size() <= kMaxDegree);
    REQUIRE(backward.size() <= kMaxDegree);
    // The two orders do not have to be identical, but they should have overlap.
    size_t overlap = 0;
    for (auto id : forward) {
        overlap += std::binary_search(backward.begin(), backward.end(), id) ? 1 : 0;
    }
    REQUIRE(overlap > 0);
}

TEST_CASE("RBC: every point appears in at least one leaf", "[pipnn][rbc]") {
    constexpr uint32_t n = 256;
    constexpr uint32_t dim = 16;

    std::mt19937 rng(2024);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 24;
    config.fanout_l1 = 6;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 17;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> counts(n, 0);
    for (const auto& leaf : leaves) {
        for (auto id : leaf.point_ids) {
            REQUIRE(id < n);
            counts[id] += 1;
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        REQUIRE(counts[i] >= 1);
    }
}

TEST_CASE("RBC: all leaf sizes <= leaf_max_size", "[pipnn][rbc]") {
    constexpr uint32_t n = 640;
    constexpr uint32_t dim = 24;

    std::mt19937 rng(101);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 32;
    config.fanout_l1 = 8;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 29;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());
    for (const auto& leaf : leaves) {
        REQUIRE(leaf.point_ids.size() <= config.leaf_max_size);
    }
}

TEST_CASE("RBC: overlap factor is reasonable", "[pipnn][rbc]") {
    constexpr uint32_t n = 512;
    constexpr uint32_t dim = 20;

    std::mt19937 rng(99);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 40;
    config.fanout_l1 = 8;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.base_seed = 7;
    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> multiplicity(n, 0);
    for (const auto& leaf : leaves) {
        for (auto id : leaf.point_ids) {
            multiplicity[id] += 1;
        }
    }

    const uint64_t total_assignments = std::accumulate(multiplicity.begin(), multiplicity.end(), uint64_t{0});
    const float overlap_factor = static_cast<float>(total_assignments) / static_cast<float>(n);

    REQUIRE(overlap_factor >= 2.0f);
    REQUIRE(overlap_factor <= 8.0f);
}

TEST_CASE("RBC: handles oversized num_leaders and still terminates", "[pipnn][rbc]") {
    constexpr uint32_t n = 9;
    constexpr uint32_t dim = 8;

    std::mt19937 rng(303);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    RBCPartitioner::Config config;
    config.leaf_max_size = 1;
    config.fanout_l1 = 4;
    config.fanout_l2 = 3;
    config.fanout_rest = 2;
    config.overlap_k = 2;
    config.num_leaders = 1024;
    config.base_seed = 17;

    RBCPartitioner partitioner(config);
    const auto leaves = partitioner.partition(data.data(), n, dim);

    REQUIRE_FALSE(leaves.empty());

    std::vector<uint32_t> counts(n, 0);
    for (const auto& leaf : leaves) {
        REQUIRE(leaf.point_ids.size() <= config.leaf_max_size);
        for (auto id : leaf.point_ids) {
            REQUIRE(id < n);
            counts[id] += 1;
        }
    }

    for (uint32_t i = 0; i < n; ++i) {
        REQUIRE(counts[i] >= 1);
    }
}

TEST_CASE("PiPNNBuilder: basic graph construction", "[pipnn][builder]") {
    constexpr uint32_t n = 320;
    constexpr uint32_t dim = 16;

    std::mt19937 rng(2025);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    PiPNNBuilder::Config config;
    config.k_nn = 48;
    config.hash_bits = 12;
    config.max_degree = 24;
    config.alpha = 1.2f;
    config.final_prune = true;
    config.num_threads = 2;

    PiPNNBuilder builder(config);
    const auto adjacency = builder.build(data.data(), n, dim);

    REQUIRE(adjacency.size() == n);
    for (uint32_t i = 0; i < n; ++i) {
        REQUIRE(adjacency[i].size() >= 1);
        REQUIRE(adjacency[i].size() <= config.max_degree);
        for (auto nbr : adjacency[i]) {
            REQUIRE(nbr < n);
            REQUIRE(nbr != i);
        }
    }
}

TEST_CASE("PiPNNBuilder: graph connectivity", "[pipnn][builder]") {
    constexpr uint32_t n = 400;
    constexpr uint32_t dim = 24;

    std::mt19937 rng(77);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    PiPNNBuilder::Config config;
    config.k_nn = 56;
    config.hash_bits = 12;
    config.max_degree = 28;
    config.alpha = 1.2f;
    config.final_prune = true;
    config.num_threads = 2;

    PiPNNBuilder builder(config);
    const auto adjacency = builder.build(data.data(), n, dim);

    REQUIRE(adjacency.size() == n);

    std::vector<uint8_t> visited(n, 0);
    std::vector<uint32_t> q;
    q.reserve(n);
    q.push_back(0);
    visited[0] = 1;

    for (size_t head = 0; head < q.size(); ++head) {
        const auto u = q[head];
        for (auto v : adjacency[u]) {
            if (!visited[v]) {
                visited[v] = 1;
                q.push_back(v);
            }
        }
    }

    const size_t reached = std::accumulate(visited.begin(), visited.end(), size_t{0});
    REQUIRE(reached >= static_cast<size_t>(n * 8 / 10));
}

TEST_CASE("PiPNNBuilder: concurrent edge insertion preserves symmetric unique adjacency", "[pipnn][builder]") {
    constexpr uint32_t n = 256;
    constexpr uint32_t dim = 20;

    std::mt19937 rng(1234);
    std::vector<float> data(n * dim);
    for (auto& v : data) {
        v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    PiPNNBuilder::Config config;
    config.k_nn = 64;
    config.hash_bits = 12;
    config.max_degree = 32;
    config.final_prune = false;
    config.num_threads = 4;
    config.leaf_max_size = 24;
    config.fanout_l1 = 8;
    config.fanout_l2 = 4;
    config.fanout_rest = 2;
    config.overlap_k = 3;

    PiPNNBuilder builder(config);
    const auto adjacency = builder.build(data.data(), n, dim);

    REQUIRE(adjacency.size() == n);
    for (uint32_t u = 0; u < n; ++u) {
        const auto& neighbors = adjacency[u];
        REQUIRE(neighbors.size() >= 1);
        REQUIRE(neighbors.size() <= config.max_degree);

        std::vector<uint32_t> sorted_neighbors = neighbors;
        std::sort(sorted_neighbors.begin(), sorted_neighbors.end());
        REQUIRE(std::adjacent_find(sorted_neighbors.begin(), sorted_neighbors.end()) == sorted_neighbors.end());

        for (uint32_t v : neighbors) {
            REQUIRE(v < n);
            REQUIRE(v != u);
        }
    }
}

TEST_CASE("VamanaSerializer: round-trip read/write", "[pipnn][serializer]") {
    constexpr uint32_t n = 100;

    std::vector<std::vector<uint32_t>> graph(n);
    std::mt19937 rng(42);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t n_nbrs = 5 + rng() % 5;
        for (uint32_t j = 0; j < n_nbrs; ++j) {
            const uint32_t nb = rng() % n;
            if (nb != i) {
                graph[i].push_back(nb);
            }
        }
    }

    const std::string path = (fs::temp_directory_path() / "test_vamana.index").string();

    knowhere::pipnn_diskann::VamanaSerializer::write(graph, 42u, path);
    REQUIRE(fs::exists(path));

    std::ifstream f(path, std::ios::binary);
    REQUIRE(f.is_open());

    uint64_t index_size = 0;
    uint32_t max_deg = 0;
    uint32_t ep = 0;
    uint64_t frozen = 0;

    f.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
    f.read(reinterpret_cast<char*>(&max_deg), sizeof(max_deg));
    f.read(reinterpret_cast<char*>(&ep), sizeof(ep));
    f.read(reinterpret_cast<char*>(&frozen), sizeof(frozen));

    REQUIRE(ep == 42u);
    REQUIRE(frozen == 0u);
    REQUIRE(max_deg > 0);
    REQUIRE(index_size > 24u);

    fs::remove(path);
}

TEST_CASE("VamanaSerializer: find_medoid", "[pipnn][serializer]") {
    constexpr uint32_t n = 50;
    constexpr uint32_t dim = 8;

    std::mt19937 rng(99);
    std::vector<float> data(n * dim);
    for (auto& x : data) {
        x = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    }

    const uint32_t medoid = knowhere::pipnn_diskann::VamanaSerializer::find_medoid(data.data(), n, dim);
    REQUIRE(medoid < n);

    std::vector<float> centroid(dim, 0.0f);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            centroid[j] += data[i * dim + j] / static_cast<float>(n);
        }
    }

    float medoid_dist = 0.0f;
    float best_dist = 1e30f;
    for (uint32_t i = 0; i < n; ++i) {
        float dist = 0.0f;
        for (uint32_t j = 0; j < dim; ++j) {
            const float diff = data[i * dim + j] - centroid[j];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
        }
        if (i == medoid) {
            medoid_dist = dist;
        }
    }

    REQUIRE(medoid_dist == Catch::Approx(best_dist).epsilon(0.01));
}

#ifdef KNOWHERE_WITH_PIPNN
TEST_CASE("PiPNNDiskANNIndexNode build+search full pipeline", "[pipnn_diskann][e2e]") {
    ScopedTempDir temp_dir("knowhere_pipnn_e2e");
    const auto raw_path = (temp_dir.dir / "raw_data.bin").string();
    const auto pipnn_index_prefix = (temp_dir.dir / "pipnn_index" / "pipnn").string();
    REQUIRE(fs::create_directories(fs::path(pipnn_index_prefix).parent_path()));

    auto base_ds = GenDataSet(kE2ENumRows, kE2EDim, 30);
    auto query_ds = GenDataSet(kE2ENumQueries, kE2EDim, 42);
    WriteRawDataToDisk<float>(raw_path, static_cast<const float*>(base_ds->GetTensor()), kE2ENumRows, kE2EDim);

    auto version = GenTestVersionList();
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    auto create_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_PIPNN_DISKANN, version, diskann_index_pack);
    REQUIRE(create_res.has_value());
    auto pipnn_index = create_res.value();

    auto build_json = MakeBuildJson(pipnn_index_prefix, raw_path, kE2EDim, kE2ENumRows);
    auto build_status = pipnn_index.Build(nullptr, build_json);
    REQUIRE(build_status == knowhere::Status::success);

    knowhere::BinarySet binset;
    REQUIRE(pipnn_index.Serialize(binset) == knowhere::Status::success);

    auto search_create_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_PIPNN_DISKANN, version, diskann_index_pack);
    REQUIRE(search_create_res.has_value());
    auto pipnn_search_index = search_create_res.value();
    auto deserialize_json = MakeDeserializeJson(pipnn_index_prefix, kE2EDim, kE2ENumRows);
    REQUIRE(pipnn_search_index.Deserialize(binset, deserialize_json) == knowhere::Status::success);

    auto search_json = MakeSearchJson(kE2EDim, kE2EK);
    auto search_res = pipnn_search_index.Search(query_ds, search_json, nullptr);
    REQUIRE(search_res.has_value());

    auto result = search_res.value();
    REQUIRE(result->GetRows() == kE2ENumQueries);
    REQUIRE(result->GetDim() == kE2EK);

    const auto* distances = result->GetDistance();
    const auto* ids = result->GetIds();
    for (int64_t i = 0; i < result->GetRows(); ++i) {
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            const auto idx = i * result->GetDim() + j;
            if (ids[idx] != -1) {
                REQUIRE(distances[idx] >= 0.0f);
            }
        }
    }
}

TEST_CASE("PiPNN postprocess-only path supports build/search/deserialize regression", "[pipnn_diskann][e2e][postprocess]") {
    constexpr uint32_t kRows = 20000;
    constexpr uint32_t kDimLocal = 64;
    constexpr uint32_t kQueries = 20;
    constexpr uint32_t kTopK = 10;

    ScopedTempDir temp_dir("knowhere_pipnn_postprocess");
    const auto raw_path = (temp_dir.dir / "raw_data.bin").string();
    const auto pipnn_index_prefix = (temp_dir.dir / "pipnn_index" / "pipnn").string();
    REQUIRE(fs::create_directories(fs::path(pipnn_index_prefix).parent_path()));

    auto base_ds = GenDataSet(kRows, kDimLocal, 30);
    auto query_ds = GenDataSet(kQueries, kDimLocal, 42);
    WriteRawDataToDisk<float>(raw_path, static_cast<const float*>(base_ds->GetTensor()), kRows, kDimLocal);

    auto version = GenTestVersionList();
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);

    auto create_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_PIPNN_DISKANN, version, diskann_index_pack);
    REQUIRE(create_res.has_value());
    auto build_index = create_res.value();

    auto build_json = MakeBuildJson(pipnn_index_prefix, raw_path, kDimLocal, kRows);
    REQUIRE(build_index.Build(nullptr, build_json) == knowhere::Status::success);

    const auto mem_index_path = fs::path(pipnn_index_prefix + "_mem.index");
    REQUIRE(fs::exists(mem_index_path));
    REQUIRE(fs::file_size(mem_index_path) > 0);

    auto search_json = MakeSearchJson(kDimLocal, kTopK);
    // Keep Search self-prepare path deterministic even when runtime state falls
    // back to lazy loading: explicitly pass index_prefix in e2e postprocess test.
    search_json["index_prefix"] = pipnn_index_prefix;
    auto build_deserialize_json = MakeDeserializeJson(pipnn_index_prefix, kDimLocal, kRows);
    REQUIRE(build_index.Deserialize(knowhere::BinarySet{}, build_deserialize_json) == knowhere::Status::success);
    auto build_search_res = build_index.Search(query_ds, search_json, nullptr);
    REQUIRE(build_search_res.has_value());

    knowhere::BinarySet binset;
    REQUIRE(build_index.Serialize(binset) == knowhere::Status::success);

    auto load_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_PIPNN_DISKANN, version, diskann_index_pack);
    REQUIRE(load_res.has_value());
    auto load_index = load_res.value();

    auto deserialize_json = MakeDeserializeJson(pipnn_index_prefix, kDimLocal, kRows);
    REQUIRE(load_index.Deserialize(binset, deserialize_json) == knowhere::Status::success);

    auto deserialize_search_res = load_index.Search(query_ds, search_json, nullptr);
    REQUIRE(deserialize_search_res.has_value());

    auto* build_result = build_search_res.value().get();
    auto* deserialize_result = deserialize_search_res.value().get();
    REQUIRE(build_result->GetRows() == deserialize_result->GetRows());
    REQUIRE(build_result->GetDim() == deserialize_result->GetDim());

    const auto total = build_result->GetRows() * build_result->GetDim();
    const auto* ids_before = build_result->GetIds();
    const auto* ids_after = deserialize_result->GetIds();
    for (int64_t i = 0; i < total; ++i) {
        REQUIRE(ids_before[i] == ids_after[i]);
    }
}

TEST_CASE("PiPNN vs DiskANN recall@10 comparison", "[pipnn_diskann][e2e][recall]") {
    struct BuildAndSearchResult {
        knowhere::DataSetPtr result;
        int64_t build_ms;
        std::string mem_index_path;
        std::optional<SerializedVamanaGraph> serialized_graph;
    };
    struct RecallProbeResult {
        PiPNNRecallTuningConfig tuning;
        float graph_direct_recall = 0.0f;
        float postprocess_recall = 0.0f;
        float forced_graph_direct_recall = 0.0f;
        int64_t postprocess_build_ms = 0;
        GraphSearchStats pipnn_graph_stats;
        GraphSearchStats forced_graph_stats;
        GraphNeighborhoodQualityStats pipnn_neighborhood_quality;
        GraphNeighborhoodQualityStats forced_neighborhood_quality;
    };

    ScopedTempDir temp_dir("knowhere_pipnn_recall");
    const auto raw_path = (temp_dir.dir / "raw_data.bin").string();
    const auto pipnn_forced_index_prefix = (temp_dir.dir / "pipnn_forced_index" / "pipnn").string();
    const auto diskann_index_prefix = (temp_dir.dir / "diskann_index" / "diskann").string();
    REQUIRE(fs::create_directories(fs::path(pipnn_forced_index_prefix).parent_path()));
    REQUIRE(fs::create_directories(fs::path(diskann_index_prefix).parent_path()));

    auto external_dataset = LoadExternalRecallDatasetPathsFromEnv();
    uint32_t dataset_rows = kE2ENumRows;
    uint32_t dataset_dim = kE2EDim;
    uint32_t dataset_queries = kE2ENumQueries;
    uint32_t dataset_topk = kE2EK;
    knowhere::DataSetPtr base_ds;
    knowhere::DataSetPtr query_ds;
    knowhere::DataSetPtr gt_dataset;

    if (external_dataset.has_value()) {
        uint32_t base_rows = 0;
        uint32_t base_dim = 0;
        auto base_data = LoadTypedBinWithHeader<float>(external_dataset->base_fbin, base_rows, base_dim);
        uint32_t query_rows = 0;
        uint32_t query_dim = 0;
        auto query_data = LoadTypedBinWithHeader<float>(external_dataset->query_fbin, query_rows, query_dim);
        uint32_t gt_rows = 0;
        uint32_t gt_topk = 0;
        auto gt_ids = LoadGroundTruthIbin(external_dataset->gt_ibin, gt_rows, gt_topk);

        REQUIRE(base_dim == query_dim);
        REQUIRE(query_rows == gt_rows);
        REQUIRE(gt_topk >= kE2EK);

        dataset_rows = base_rows;
        dataset_dim = base_dim;
        dataset_queries = query_rows;
        dataset_topk = kE2EK;

        auto base_tensor = std::make_unique<float[]>(base_data.size());
        std::copy(base_data.begin(), base_data.end(), base_tensor.get());
        base_ds = knowhere::GenDataSet(base_rows, base_dim, base_tensor.release());
        base_ds->SetIsOwner(true);

        auto query_tensor = std::make_unique<float[]>(query_data.size());
        std::copy(query_data.begin(), query_data.end(), query_tensor.get());
        query_ds = knowhere::GenDataSet(query_rows, query_dim, query_tensor.release());
        query_ds->SetIsOwner(true);

        auto gt_ids_owner = std::make_unique<int64_t[]>(static_cast<size_t>(query_rows) * kE2EK);
        std::copy(gt_ids.begin(), gt_ids.begin() + static_cast<ptrdiff_t>(query_rows) * kE2EK, gt_ids_owner.get());
        auto gt_dist_owner = std::make_unique<float[]>(static_cast<size_t>(query_rows) * kE2EK);
        std::fill_n(gt_dist_owner.get(), static_cast<size_t>(query_rows) * kE2EK, 0.0f);
        gt_dataset = knowhere::GenResultDataSet(query_rows, kE2EK, std::move(gt_ids_owner), std::move(gt_dist_owner));
        WriteRawDataToDisk<float>(raw_path, base_data.data(), base_rows, base_dim);
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Dataset] label=" << external_dataset->label << ", rows=" << base_rows
                           << ", dim=" << base_dim << ", queries=" << query_rows << ", gt_topk=" << gt_topk
                           << ", mode=external_public_dataset";
    } else {
        base_ds = GenDataSet(kE2ENumRows, kE2EDim, 30);
        query_ds = GenDataSet(kE2ENumQueries, kE2EDim, 42);
        WriteRawDataToDisk<float>(raw_path, static_cast<const float*>(base_ds->GetTensor()), kE2ENumRows, kE2EDim);

        auto gt_json = MakeBaseJson(kE2EDim, kE2EK);
        auto gt_res = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, gt_json, nullptr);
        REQUIRE(gt_res.has_value());
        gt_dataset = gt_res.value();
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Dataset] label=synthetic_diag, rows=" << kE2ENumRows
                           << ", dim=" << kE2EDim << ", queries=" << kE2ENumQueries
                           << ", mode=synthetic";
    }

    auto version = GenTestVersionList();
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    const auto baseline_tuning = LoadPiPNNRecallTuningFromEnv();
    std::vector<PiPNNRecallTuningConfig> tuning_matrix;

    auto repair_off = baseline_tuning;
    repair_off.label = "retained_degree_limit_off";
    repair_off.enable_cross_leaf_union = false;
    repair_off.enable_retained_degree_limit = false;
    tuning_matrix.push_back(repair_off);

    auto repair_on = baseline_tuning;
    repair_on.label = "retained_degree_limit_on";
    repair_on.enable_cross_leaf_union = false;
    repair_on.enable_retained_degree_limit = true;
    repair_on.enable_boundary_leader_bridge = false;
    tuning_matrix.push_back(repair_on);

    auto boundary_bridge = baseline_tuning;
    boundary_bridge.label = "boundary_leader_bridge_on";
    boundary_bridge.enable_cross_leaf_union = false;
    boundary_bridge.enable_retained_degree_limit = true;
    boundary_bridge.enable_boundary_leader_bridge = true;
    tuning_matrix.push_back(boundary_bridge);

    for (const auto& tuning : tuning_matrix) {
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Matrix] tuning=" << DescribePiPNNRecallTuning(tuning);
    }

    auto build_and_search = [&](const std::string& index_type, const std::string& index_prefix,
                                const char* scenario_label, const char* force_diskann_build_index,
                                const PiPNNRecallTuningConfig* pipnn_tuning) {
        std::unique_ptr<ScopedEnvVar> force_build_guard;
        std::unique_ptr<ScopedEnvVar> retained_degree_limit_guard;
        std::unique_ptr<ScopedEnvVar> boundary_leader_bridge_guard;
        if (index_type == knowhere::IndexEnum::INDEX_PIPNN_DISKANN) {
            force_build_guard = std::make_unique<ScopedEnvVar>("KNOWHERE_PIPNN_FORCE_DISKANN_BUILD_INDEX",
                                                               force_diskann_build_index);
            retained_degree_limit_guard = std::make_unique<ScopedEnvVar>(
                "KNOWHERE_PIPNN_ENABLE_RETAINED_DEGREE_LIMIT",
                pipnn_tuning != nullptr
                    ? (pipnn_tuning->enable_retained_degree_limit ? "1" : "0")
                    : nullptr);
            boundary_leader_bridge_guard = std::make_unique<ScopedEnvVar>(
                "KNOWHERE_PIPNN_ENABLE_BOUNDARY_LEADER_BRIDGE",
                pipnn_tuning != nullptr
                    ? (pipnn_tuning->enable_boundary_leader_bridge ? "1" : "0")
                    : nullptr);
        }

        auto create_res =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack);
        REQUIRE(create_res.has_value());
        auto index = create_res.value();

        auto build_json = MakeBuildJson(index_prefix, raw_path, dataset_dim, dataset_rows);
        if (index_type == knowhere::IndexEnum::INDEX_PIPNN_DISKANN && pipnn_tuning != nullptr) {
            ApplyPiPNNRecallTuning(build_json, *pipnn_tuning);
        }
        const auto start = std::chrono::steady_clock::now();
        REQUIRE(index.Build(nullptr, build_json) == knowhere::Status::success);
        const auto end = std::chrono::steady_clock::now();
        const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        const auto mem_index_path = index_prefix + "_mem.index";
        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        auto search_create_res =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack);
        REQUIRE(search_create_res.has_value());
        auto search_index = search_create_res.value();
        auto deserialize_json = MakeDeserializeJson(index_prefix, dataset_dim, dataset_rows);
        REQUIRE(search_index.Deserialize(binset, deserialize_json) == knowhere::Status::success);

        fs::path resolved_mem_index_path(mem_index_path);
        if (!fs::exists(resolved_mem_index_path)) {
            for (const auto& entry : fs::recursive_directory_iterator(temp_dir.dir)) {
                if (entry.is_regular_file() && entry.path().extension() == ".index" &&
                    entry.path().filename().string().find("_mem.index") != std::string::npos) {
                    resolved_mem_index_path = entry.path();
                    break;
                }
            }
        }

        std::optional<SerializedVamanaGraph> serialized_graph;
        if (fs::exists(resolved_mem_index_path)) {
            serialized_graph =
                LoadSerializedVamanaGraph(resolved_mem_index_path.string(), static_cast<uint32_t>(base_ds->GetRows()));
        }

        auto search_json = MakeSearchJson(dataset_dim, kE2EK);
        auto search_res = search_index.Search(query_ds, search_json, nullptr);
        REQUIRE(search_res.has_value());
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall A/B] scenario=" << scenario_label << ", index_type=" << index_type
                           << ", build_ms=" << build_ms;
        return BuildAndSearchResult{search_res.value(),
                                    build_ms,
                                    resolved_mem_index_path.string(),
                                    std::move(serialized_graph)};
    };

    const auto pipnn_forced = build_and_search(knowhere::IndexEnum::INDEX_PIPNN_DISKANN,
                                               pipnn_forced_index_prefix,
                                               "pipnn_force_diskann_build_index",
                                               "1",
                                               &baseline_tuning);
    const auto diskann_baseline =
        build_and_search(knowhere::IndexEnum::INDEX_DISKANN, diskann_index_prefix, "diskann_baseline", nullptr, nullptr);

    std::optional<GraphSearchStats> forced_graph_stats;
    std::optional<GraphNeighborhoodQualityStats> forced_neighborhood_quality;
    std::optional<std::vector<std::vector<int64_t>>> forced_graph_direct_results;

    if (pipnn_forced.serialized_graph.has_value()) {
        forced_graph_stats = CollectGraphSearchStats(*pipnn_forced.serialized_graph);
        forced_neighborhood_quality = CollectGraphNeighborhoodQualityStats(*base_ds, *pipnn_forced.serialized_graph, kE2EK);
        forced_graph_direct_results =
            SearchSerializedGraphBatch(*base_ds, *query_ds, kE2EK, *pipnn_forced.serialized_graph, 100);
    }

    std::vector<std::vector<int64_t>> gt_ids;
    gt_ids.reserve(dataset_queries);
    const auto* gt_id_ptr = gt_dataset->GetIds();
    for (uint32_t qi = 0; qi < dataset_queries; ++qi) {
        gt_ids.emplace_back(gt_id_ptr + static_cast<size_t>(qi) * kE2EK,
                            gt_id_ptr + static_cast<size_t>(qi + 1) * kE2EK);
    }

    const float pipnn_forced_recall = GetKNNRecall(*gt_dataset, *pipnn_forced.result);
    const float forced_graph_direct_recall = forced_graph_direct_results.has_value()
                                                 ? GetKNNRecall(*gt_dataset, *forced_graph_direct_results)
                                                 : -1.0f;
    const float diskann_recall = GetKNNRecall(*gt_dataset, *diskann_baseline.result);
    const auto forced_local_recall = forced_graph_direct_results.has_value()
                                         ? std::optional<LocalGraphRecallStats>(
                                               CollectLocalGraphRecallStats(gt_ids, *forced_graph_direct_results, kE2EK))
                                         : std::nullopt;

    LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=force_diskann_build_index"
                       << ", graph_direct_recall@10=" << forced_graph_direct_recall
                       << ", force_diskann_build_index=" << pipnn_forced_recall
                       << ", diskann_baseline=" << diskann_recall
                       << ", serialized_graph_available=" << pipnn_forced.serialized_graph.has_value();
    if (forced_graph_stats.has_value()) {
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=force_diskann_build_index"
                           << ", graph_stats avg_degree=" << forced_graph_stats->avg_degree
                           << ", min_degree=" << forced_graph_stats->min_degree
                           << ", max_degree=" << forced_graph_stats->max_degree
                           << ", zero_degree_nodes=" << forced_graph_stats->zero_degree_nodes
                           << ", oversized_nodes=" << forced_graph_stats->oversized_nodes
                           << ", invalid_edges=" << forced_graph_stats->invalid_edges
                           << ", asym_edges=" << forced_graph_stats->asym_edges
                           << ", entry_point_degree=" << forced_graph_stats->entry_point_degree
                           << ", degree_le_8=" << forced_graph_stats->degree_le_8
                           << ", degree_9_16=" << forced_graph_stats->degree_9_16
                           << ", degree_17_24=" << forced_graph_stats->degree_17_24
                           << ", degree_25_31=" << forced_graph_stats->degree_25_31
                           << ", degree_eq_max=" << forced_graph_stats->degree_eq_max;
    }
    if (forced_local_recall.has_value()) {
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=force_diskann_build_index"
                           << ", local_graph_recall avg=" << forced_local_recall->avg_recall
                           << ", min=" << forced_local_recall->min_recall
                           << ", max=" << forced_local_recall->max_recall
                           << ", zero_hit_queries=" << forced_local_recall->zero_hit_queries;
    }
    if (forced_neighborhood_quality.has_value()) {
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=force_diskann_build_index"
                           << ", neighborhood_quality avg_recall=" << forced_neighborhood_quality->avg_neighbor_recall
                           << ", min_recall=" << forced_neighborhood_quality->min_neighbor_recall
                           << ", max_recall=" << forced_neighborhood_quality->max_neighbor_recall
                           << ", avg_precision=" << forced_neighborhood_quality->avg_neighbor_precision
                           << ", entry_recall=" << forced_neighborhood_quality->avg_entry_neighbor_recall
                           << ", entry_precision=" << forced_neighborhood_quality->avg_entry_neighbor_precision
                           << ", zero_recall_nodes=" << forced_neighborhood_quality->zero_neighbor_recall_nodes
                           << ", evaluated_nodes=" << forced_neighborhood_quality->evaluated_nodes
                           << ", candidate_pool_size=" << forced_neighborhood_quality->candidate_pool_size;
    }

    std::vector<RecallProbeResult> probe_results;
    probe_results.reserve(tuning_matrix.size());
    for (const auto& tuning : tuning_matrix) {
        const auto tuning_dir = temp_dir.dir / ("pipnn_" + tuning.label + "_index");
        const auto tuning_prefix = (tuning_dir / "pipnn").string();
        REQUIRE(fs::create_directories(tuning_dir));

        const auto pipnn_postprocess = build_and_search(knowhere::IndexEnum::INDEX_PIPNN_DISKANN,
                                                        tuning_prefix,
                                                        tuning.label.c_str(),
                                                        nullptr,
                                                        &tuning);
        REQUIRE(pipnn_postprocess.serialized_graph.has_value());
        const auto& serialized_graph = *pipnn_postprocess.serialized_graph;
        const auto pipnn_graph_stats = CollectGraphSearchStats(serialized_graph);
        const auto pipnn_neighborhood_quality = CollectGraphNeighborhoodQualityStats(*base_ds, serialized_graph, kE2EK);
        const auto forced_graph_overlap =
            forced_graph_stats.has_value() ? std::optional<GraphPairOverlapStats>(CollectGraphPairOverlapStats(
                                          serialized_graph,
                                          *pipnn_forced.serialized_graph,
                                          kE2EK))
                                          : std::nullopt;
        const auto graph_direct_results =
            SearchSerializedGraphBatch(*base_ds, *query_ds, kE2EK, serialized_graph, 100);

        const float graph_direct_recall = GetKNNRecall(*gt_dataset, graph_direct_results);
        const float pipnn_postprocess_recall = GetKNNRecall(*gt_dataset, *pipnn_postprocess.result);
        const auto local_recall = CollectLocalGraphRecallStats(gt_ids, graph_direct_results, kE2EK);

        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                           << ", graph_direct_recall@10=" << graph_direct_recall
                           << ", postprocess_only=" << pipnn_postprocess_recall
                           << ", force_diskann_build_index_graph_direct=" << forced_graph_direct_recall
                           << ", force_diskann_build_index=" << pipnn_forced_recall
                           << ", diskann_baseline=" << diskann_recall;
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                           << ", mem_index header index_size=" << serialized_graph.index_size
                           << ", max_degree=" << serialized_graph.max_degree
                           << ", entry_point=" << serialized_graph.entry_point
                           << ", frozen_pts=" << serialized_graph.num_frozen_pts;
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                           << ", pipnn_graph_stats avg_degree=" << pipnn_graph_stats.avg_degree
                           << ", min_degree=" << pipnn_graph_stats.min_degree
                           << ", max_degree=" << pipnn_graph_stats.max_degree
                           << ", zero_degree_nodes=" << pipnn_graph_stats.zero_degree_nodes
                           << ", oversized_nodes=" << pipnn_graph_stats.oversized_nodes
                           << ", invalid_edges=" << pipnn_graph_stats.invalid_edges
                           << ", asym_edges=" << pipnn_graph_stats.asym_edges
                           << ", entry_point_degree=" << pipnn_graph_stats.entry_point_degree
                           << ", degree_le_8=" << pipnn_graph_stats.degree_le_8
                           << ", degree_9_16=" << pipnn_graph_stats.degree_9_16
                           << ", degree_17_24=" << pipnn_graph_stats.degree_17_24
                           << ", degree_25_31=" << pipnn_graph_stats.degree_25_31
                           << ", degree_eq_max=" << pipnn_graph_stats.degree_eq_max;
        if (forced_graph_stats.has_value()) {
            LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                               << ", forced_graph_stats avg_degree=" << forced_graph_stats->avg_degree
                               << ", min_degree=" << forced_graph_stats->min_degree
                               << ", max_degree=" << forced_graph_stats->max_degree
                               << ", zero_degree_nodes=" << forced_graph_stats->zero_degree_nodes
                               << ", oversized_nodes=" << forced_graph_stats->oversized_nodes
                               << ", invalid_edges=" << forced_graph_stats->invalid_edges
                               << ", asym_edges=" << forced_graph_stats->asym_edges
                               << ", entry_point_degree=" << forced_graph_stats->entry_point_degree
                               << ", degree_le_8=" << forced_graph_stats->degree_le_8
                               << ", degree_9_16=" << forced_graph_stats->degree_9_16
                               << ", degree_17_24=" << forced_graph_stats->degree_17_24
                               << ", degree_25_31=" << forced_graph_stats->degree_25_31
                               << ", degree_eq_max=" << forced_graph_stats->degree_eq_max;
        }
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                           << ", local_graph_recall avg=" << local_recall.avg_recall
                           << ", min=" << local_recall.min_recall
                           << ", max=" << local_recall.max_recall
                           << ", zero_hit_queries=" << local_recall.zero_hit_queries
                           << ", forced_local_graph_recall_avg="
                           << (forced_local_recall.has_value() ? forced_local_recall->avg_recall : -1.0f);
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                           << ", neighborhood_quality avg_recall=" << pipnn_neighborhood_quality.avg_neighbor_recall
                           << ", min_recall=" << pipnn_neighborhood_quality.min_neighbor_recall
                           << ", max_recall=" << pipnn_neighborhood_quality.max_neighbor_recall
                           << ", avg_precision=" << pipnn_neighborhood_quality.avg_neighbor_precision
                           << ", entry_recall=" << pipnn_neighborhood_quality.avg_entry_neighbor_recall
                           << ", entry_precision=" << pipnn_neighborhood_quality.avg_entry_neighbor_precision
                           << ", zero_recall_nodes=" << pipnn_neighborhood_quality.zero_neighbor_recall_nodes
                           << ", evaluated_nodes=" << pipnn_neighborhood_quality.evaluated_nodes
                           << ", candidate_pool_size=" << pipnn_neighborhood_quality.candidate_pool_size
                           << ", forced_avg_recall="
                           << (forced_neighborhood_quality.has_value() ? forced_neighborhood_quality->avg_neighbor_recall : -1.0)
                           << ", forced_avg_precision="
                           << (forced_neighborhood_quality.has_value() ? forced_neighborhood_quality->avg_neighbor_precision : -1.0);
        if (forced_graph_overlap.has_value()) {
            LOG_KNOWHERE_INFO_ << "[PiPNN Candidate Control] scenario=" << tuning.label
                               << ", sampled_neighborhood_overlap=" << forced_graph_overlap->avg_neighbor_overlap
                               << ", min_overlap=" << forced_graph_overlap->min_neighbor_overlap
                               << ", max_overlap=" << forced_graph_overlap->max_neighbor_overlap
                               << ", zero_overlap_nodes=" << forced_graph_overlap->zero_neighbor_overlap_nodes
                               << ", evaluated_nodes=" << forced_graph_overlap->evaluated_nodes
                               << ", entry_neighbor_overlap=" << forced_graph_overlap->entry_neighbor_overlap
                               << ", entry_overlap_nodes=" << forced_graph_overlap->entry_overlap_nodes
                               << ", entry_point_matches="
                               << (forced_graph_overlap->entry_point_matches ? "true" : "false");
        }
        if (forced_graph_stats.has_value()) {
            LOG_KNOWHERE_INFO_ << "[PiPNN Recall Probe] scenario=" << tuning.label
                               << ", graph_delta avg_degree="
                               << (pipnn_graph_stats.avg_degree - forced_graph_stats->avg_degree)
                               << ", asym_edges=" << static_cast<int64_t>(pipnn_graph_stats.asym_edges) -
                                                         static_cast<int64_t>(forced_graph_stats->asym_edges)
                               << ", entry_point_degree="
                               << static_cast<int64_t>(pipnn_graph_stats.entry_point_degree) -
                                      static_cast<int64_t>(forced_graph_stats->entry_point_degree)
                               << ", avg_neighbor_recall="
                               << (pipnn_neighborhood_quality.avg_neighbor_recall -
                                   (forced_neighborhood_quality.has_value() ? forced_neighborhood_quality->avg_neighbor_recall : 0.0))
                               << ", avg_neighbor_precision="
                               << (pipnn_neighborhood_quality.avg_neighbor_precision -
                                   (forced_neighborhood_quality.has_value() ? forced_neighborhood_quality->avg_neighbor_precision : 0.0))
                               << ", graph_direct_recall@10=" << graph_direct_recall
                               << ", forced_graph_direct_recall@10=" << forced_graph_direct_recall;
        }
        LOG_KNOWHERE_INFO_ << "[PiPNN Recall A/B] scenario=" << tuning.label
                           << ", build_ms postprocess_only=" << pipnn_postprocess.build_ms
                           << ", force_diskann_build_index=" << pipnn_forced.build_ms
                           << ", diskann_baseline=" << diskann_baseline.build_ms;

        REQUIRE(serialized_graph.entry_point < base_ds->GetRows());
        REQUIRE(pipnn_graph_stats.invalid_edges == 0);
        REQUIRE(pipnn_graph_stats.oversized_nodes == 0);
        if (forced_graph_stats.has_value()) {
            REQUIRE(forced_graph_stats->invalid_edges == 0);
            REQUIRE(forced_graph_stats->oversized_nodes == 0);
        }

        probe_results.push_back(RecallProbeResult{tuning,
                                                  graph_direct_recall,
                                                  pipnn_postprocess_recall,
                                                  forced_graph_direct_recall,
                                                  pipnn_postprocess.build_ms,
                                                  pipnn_graph_stats,
                                                  forced_graph_stats.value_or(GraphSearchStats{}),
                                                  pipnn_neighborhood_quality,
                                                  forced_neighborhood_quality.value_or(GraphNeighborhoodQualityStats{})});
    }

    REQUIRE(probe_results.size() == tuning_matrix.size());
    const auto best_postprocess =
        std::max_element(probe_results.begin(), probe_results.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.postprocess_recall < rhs.postprocess_recall;
        });
    REQUIRE(best_postprocess != probe_results.end());
    LOG_KNOWHERE_INFO_ << "[PiPNN Recall Matrix] best_postprocess scenario=" << best_postprocess->tuning.label
                       << ", postprocess_recall@10=" << best_postprocess->postprocess_recall
                       << ", graph_direct_recall@10=" << best_postprocess->graph_direct_recall
                       << ", forced_graph_direct_recall@10=" << best_postprocess->forced_graph_direct_recall
                       << ", postprocess_build_ms=" << best_postprocess->postprocess_build_ms
                       << ", forced_build_ms=" << pipnn_forced.build_ms
                       << ", diskann_build_ms=" << diskann_baseline.build_ms;
    REQUIRE(pipnn_forced_recall + 0.05f >= diskann_recall);
}
#endif
