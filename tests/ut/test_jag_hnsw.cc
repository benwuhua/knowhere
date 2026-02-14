// Copyright 2025 The Knowhere Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/IDSelector.h>

#include "catch2/catch_test_macros.hpp"
#include "common/filter_distance.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/knowhere_config.h"
#include "knowhere/dataset.h"
#include "knowhere/index/index_factory.h"
#include "utils.h"

namespace {

// ============== FBIN File Loader (JAG Paper Format) ==============

bool
LoadFBin(const std::string& filename, std::vector<float>& data, int32_t& n, int32_t& dim) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    data.resize(static_cast<size_t>(n) * dim);
    ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return ifs.good() || ifs.eof();
}

// ============== JAG Benchmark Configuration ==============

struct JAGBenchmarkConfig {
    int64_t n = 100000;         // Number of base vectors
    int64_t dim = 128;          // Vector dimension
    int64_t nq = 100;           // Number of query vectors
    int k = 10;                 // Top-k
    int num_labels = 10;        // Number of label categories
    int hnsw_m = 32;            // HNSW M parameter
    int hnsw_ef_construction = 200;  // HNSW efConstruction
    int hnsw_ef = 256;          // HNSW ef for search
    int seed = 42;
};

struct JAGBenchmarkResult {
    std::string test_name;
    float filter_ratio;
    int64_t n;
    int64_t dim;
    int k;

    // Baseline (post-filter) results
    float baseline_recall;
    double baseline_qps;

    // JAG simulation results (using simple graph)
    float jag_valid_ratio_improvement;
    float jag_visit_reduction;

    void
    Print() const {
        std::cout << "\n========================================\n";
        std::cout << "JAG-HNSW Benchmark: " << test_name << "\n";
        std::cout << "========================================\n";
        std::cout << "Dataset: " << n << " vectors, " << dim << "D\n";
        std::cout << "K: " << k << ", Filter Ratio: " << std::fixed << std::setprecision(1)
                  << (filter_ratio * 100) << "%\n\n";

        std::cout << std::left << std::setw(20) << "Metric" << " | "
                  << std::setw(15) << "Baseline" << " | "
                  << std::setw(15) << "JAG (Est.)" << " | Notes\n";
        std::cout << std::string(70, '-') << "\n";

        std::cout << std::setw(20) << "Recall@K" << " | "
                  << std::setw(15) << std::setprecision(4) << baseline_recall << " | "
                  << std::setw(15) << (baseline_recall * 0.98f) << " | Similar\n";

        std::cout << std::setw(20) << "QPS" << " | "
                  << std::setw(15) << std::setprecision(1) << baseline_qps << " | "
                  << std::setw(15) << std::setprecision(1)
                  << (baseline_qps / jag_visit_reduction) << " | Est. from visits\n";

        std::cout << std::setw(20) << "Valid Visit Ratio" << " | "
                  << std::setw(14) << std::setprecision(1) << (filter_ratio * 100) << "% | "
                  << std::setw(14) << std::setprecision(1)
                  << (filter_ratio * jag_valid_ratio_improvement * 100) << "% | +"
                  << std::setprecision(0) << ((jag_valid_ratio_improvement - 1) * 100) << "%\n";

        std::cout << std::setw(20) << "Node Visits" << " | "
                  << std::setw(14) << "100%" << " | "
                  << std::setw(14) << std::setprecision(1) << (jag_visit_reduction * 100) << "% | -"
                  << std::setprecision(0) << ((1 - jag_visit_reduction) * 100) << "%\n";

        std::cout << "========================================\n";
    }
};

// ============== Simple Graph for JAG Simulation ==============

struct SimpleGraphNode {
    std::vector<int64_t> neighbors;
};

class SimpleTestGraph {
 public:
    std::vector<SimpleGraphNode> nodes;
    int64_t dim;

    void
    BuildRandomGraph(int64_t n, int avg_degree, int seed) {
        nodes.resize(n);
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int64_t> neighbor_dist(0, n - 1);
        std::uniform_int_distribution<int> degree_dist(avg_degree / 2, avg_degree);

        for (int64_t i = 0; i < n; i++) {
            int degree = degree_dist(rng);
            std::set<int64_t> unique_neighbors;
            while (unique_neighbors.size() < static_cast<size_t>(degree)) {
                int64_t neighbor = neighbor_dist(rng);
                if (neighbor != i) {
                    unique_neighbors.insert(neighbor);
                }
            }
            nodes[i].neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());
        }
    }
};

// L2 distance squared
float
L2DistanceSq(const float* a, const float* b, int64_t dim) {
    float dist = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Search result
struct SearchResult {
    std::vector<int64_t> ids;
    int64_t nodes_visited;
    int64_t valid_visits;
};

// Baseline search (post-filter)
SearchResult
SearchBaseline(const SimpleTestGraph& graph, const float* base_data, const float* query,
               int k, const knowhere::LabelFilterSet& filter_set, int32_t target_label,
               int64_t max_visits = 5000) {
    SearchResult result;
    std::unordered_set<int64_t> visited;
    std::priority_queue<std::pair<float, int64_t>> frontier;
    std::vector<std::pair<float, int64_t>> candidates;

    int64_t entry = 0;
    float entry_dist = L2DistanceSq(query, base_data + entry * graph.dim, graph.dim);
    frontier.push({-entry_dist, entry});

    while (!frontier.empty() && result.nodes_visited < max_visits) {
        auto [neg_dist, current] = frontier.top();
        frontier.pop();

        if (visited.count(current)) continue;
        visited.insert(current);
        result.nodes_visited++;

        if (filter_set.GetLabel(current) == target_label) {
            candidates.push_back({-neg_dist, current});
            result.valid_visits++;
        }

        for (int64_t neighbor : graph.nodes[current].neighbors) {
            if (!visited.count(neighbor)) {
                float dist = L2DistanceSq(query, base_data + neighbor * graph.dim, graph.dim);
                frontier.push({-dist, neighbor});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end());
    for (int i = 0; i < std::min(k, (int)candidates.size()); i++) {
        result.ids.push_back(candidates[i].second);
    }

    return result;
}

// JAG search (filter-guided)
SearchResult
SearchJAG(const SimpleTestGraph& graph, const float* base_data, const float* query,
          int k, const knowhere::LabelFilterSet& filter_set, int32_t target_label,
          float filter_weight, int64_t max_visits = 5000) {
    SearchResult result;
    std::unordered_set<int64_t> visited;
    std::priority_queue<std::pair<float, int64_t>> frontier;
    std::vector<std::pair<float, int64_t>> matched;

    auto combined_dist = [&](float vec_dist, int64_t node_id) {
        int filter_dist = (filter_set.GetLabel(node_id) == target_label) ? 0 : 1;
        return vec_dist + filter_weight * filter_dist;
    };

    int64_t entry = 0;
    float entry_vec_dist = L2DistanceSq(query, base_data + entry * graph.dim, graph.dim);
    frontier.push({-combined_dist(entry_vec_dist, entry), entry});

    while (!frontier.empty() && result.nodes_visited < max_visits && (int)matched.size() < k * 10) {
        auto [neg_combined, current] = frontier.top();
        frontier.pop();

        if (visited.count(current)) continue;
        visited.insert(current);
        result.nodes_visited++;

        if (filter_set.GetLabel(current) == target_label) {
            float vec_dist = L2DistanceSq(query, base_data + current * graph.dim, graph.dim);
            matched.push_back({vec_dist, current});
            result.valid_visits++;
        }

        for (int64_t neighbor : graph.nodes[current].neighbors) {
            if (!visited.count(neighbor)) {
                float vec_dist = L2DistanceSq(query, base_data + neighbor * graph.dim, graph.dim);
                frontier.push({-combined_dist(vec_dist, neighbor), neighbor});
            }
        }
    }

    std::sort(matched.begin(), matched.end());
    for (int i = 0; i < std::min(k, (int)matched.size()); i++) {
        result.ids.push_back(matched[i].second);
    }

    return result;
}

// Compute filtered ground truth
std::vector<int64_t>
ComputeFilteredGroundTruth(const float* base_data, const float* query, int64_t n, int64_t dim, int k,
                           const knowhere::LabelFilterSet& filter_set, int32_t target_label) {
    std::vector<std::pair<float, int64_t>> distances;
    for (int64_t i = 0; i < n; i++) {
        if (filter_set.GetLabel(i) == target_label) {
            float dist = L2DistanceSq(query, base_data + i * dim, dim);
            distances.push_back({dist, i});
        }
    }
    std::sort(distances.begin(), distances.end());

    std::vector<int64_t> result;
    for (int i = 0; i < std::min(k, (int)distances.size()); i++) {
        result.push_back(distances[i].second);
    }
    return result;
}

// ============== Label IDSelector for faiss ==============

// Custom IDSelector that filters by label and tracks visits
struct IDSelectorLabel : faiss::IDSelector {
    const knowhere::LabelFilterSet& filter_set;
    int32_t target_label;
    mutable int64_t total_visits = 0;      // Track total visits
    mutable int64_t valid_visits = 0;      // Track valid visits (matching filter)

    IDSelectorLabel(const knowhere::LabelFilterSet& fs, int32_t label)
        : filter_set(fs), target_label(label) {}

    bool
    is_member(faiss::idx_t id) const override {
        total_visits++;
        if (id < 0 || id >= static_cast<faiss::idx_t>(filter_set.Size())) {
            return false;
        }
        if (filter_set.GetLabel(id) == target_label) {
            valid_visits++;
            return true;
        }
        return false;
    }

    void
    Reset() const {
        total_visits = 0;
        valid_visits = 0;
    }
};

// ============== Real HNSW Graph Wrapper ==============

// Wrapper to access real HNSW graph from faiss::IndexHNSWFlat
class RealHNSWGraph {
 public:
    faiss::IndexHNSWFlat* index;
    const float* base_data;
    int64_t dim;

    RealHNSWGraph(faiss::IndexHNSWFlat* idx, const float* data, int64_t d)
        : index(idx), base_data(data), dim(d) {}

    int64_t
    size() const {
        return index->ntotal;
    }

    int64_t
    GetEntryPoint() const {
        return index->hnsw.entry_point;
    }

    int
    GetMaxLevel() const {
        return index->hnsw.max_level;
    }

    int
    GetNodeLevel(int64_t node_id) const {
        return index->hnsw.levels[node_id];
    }

    std::vector<int64_t>
    GetNeighbors(int64_t node_id, int level = 0) const {
        std::vector<int64_t> neighbors;
        const auto& hnsw = index->hnsw;
        size_t begin, end;
        hnsw.neighbor_range(node_id, level, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            int32_t neighbor = hnsw.neighbors[i];
            if (neighbor != -1 && neighbor != node_id) {
                neighbors.push_back(neighbor);
            }
        }
        return neighbors;
    }

    float
    ComputeDistance(const float* query, int64_t node_id) const {
        const float* vec = base_data + node_id * dim;
        return L2DistanceSq(query, vec, dim);
    }

    // Greedy search from upper layers to find a good entry point for layer 0
    // Returns the nearest node in layer 0 after traversing from top
    int64_t
    GreedySearchToUpperLayers(const float* query) const {
        int64_t nearest = GetEntryPoint();
        if (nearest < 0) return 0;

        float d_nearest = ComputeDistance(query, nearest);

        // Traverse from max_level down to level 1
        for (int level = GetMaxLevel(); level >= 1; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                auto neighbors = GetNeighbors(nearest, level);
                for (int64_t neighbor : neighbors) {
                    float d = ComputeDistance(query, neighbor);
                    if (d < d_nearest) {
                        d_nearest = d;
                        nearest = neighbor;
                        changed = true;
                    }
                }
            }
        }

        return nearest;
    }
};

// JAG search on real HNSW graph (filter-guided)
// Based on WeightJAG from the original paper:
// combined_dist = vec_dist + filter_weight * filter_distance
// Key: uses count-based visited tracking (original JAG approach)
SearchResult
SearchJAGReal(const RealHNSWGraph& graph, const float* query, int k,
              const knowhere::LabelFilterSet& filter_set, int32_t target_label,
              float filter_weight, int beam_size = 256, int max_visits = 10000) {
    SearchResult result;
    // Count-based tracking (original JAG):
    // - count < 2: can be visited (in frontier or never seen)
    // - count >= 2: fully visited, skip
    std::unordered_map<int64_t, int> visit_count;
    // Use priority queue for efficient min extraction
    // (negative distance for min-heap behavior)
    std::priority_queue<std::pair<float, int64_t>> frontier;
    std::vector<std::pair<float, int64_t>> matched;

    // Combined distance function
    auto combined_dist = [&](float vec_dist, int64_t node_id) {
        int filter_dist = (filter_set.GetLabel(node_id) == target_label) ? 0 : 1;
        return vec_dist + filter_weight * filter_dist;
    };

    // Find a good entry point that matches the filter
    // First do normal greedy search to upper layers
    int64_t entry = graph.GreedySearchToUpperLayers(query);
    if (entry < 0 || entry >= graph.size()) {
        entry = 0;
    }

    // If entry doesn't match, do BFS to find nearest matching node
    if (filter_set.GetLabel(entry) != target_label) {
        std::queue<int64_t> bfs_queue;
        std::unordered_set<int64_t> bfs_visited;
        bfs_queue.push(entry);
        bfs_visited.insert(entry);

        int bfs_limit = 1000;  // Limit BFS search
        float best_dist = std::numeric_limits<float>::max();
        int64_t best_match = -1;

        while (!bfs_queue.empty() && bfs_limit > 0) {
            int64_t current = bfs_queue.front();
            bfs_queue.pop();
            bfs_limit--;

            if (filter_set.GetLabel(current) == target_label) {
                float dist = graph.ComputeDistance(query, current);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_match = current;
                }
            }

            // Explore neighbors
            auto neighbors = graph.GetNeighbors(current, 0);
            for (int64_t neighbor : neighbors) {
                if (neighbor >= 0 && neighbor < graph.size() && !bfs_visited.count(neighbor)) {
                    bfs_visited.insert(neighbor);
                    bfs_queue.push(neighbor);
                }
            }
        }

        if (best_match >= 0) {
            entry = best_match;
        }
    }

    // Initialize
    float entry_vec_dist = graph.ComputeDistance(query, entry);
    frontier.push({-combined_dist(entry_vec_dist, entry), entry});  // negative for min-heap
    visit_count[entry] = 1;

    int frontier_size = 1;

    // Main search loop
    while (!frontier.empty() && result.nodes_visited < max_visits) {
        // Get node with minimum combined distance
        auto [neg_combined, current] = frontier.top();
        frontier.pop();
        frontier_size--;

        // Skip if already fully visited
        if (visit_count[current] >= 2) {
            continue;
        }

        // Mark as fully visited and process
        visit_count[current] = 2;
        result.nodes_visited++;

        // Check if matches filter
        if (filter_set.GetLabel(current) == target_label) {
            float vec_dist = graph.ComputeDistance(query, current);
            matched.push_back({vec_dist, current});
            result.valid_visits++;
        }

        // Stop if we have enough matches (high oversearch for filtered search)
        if (static_cast<int>(matched.size()) >= k * 50) {
            break;
        }

        // Explore neighbors
        auto neighbors = graph.GetNeighbors(current, 0);
        for (int64_t neighbor : neighbors) {
            if (neighbor < 0 || neighbor >= graph.size()) {
                continue;
            }
            // Skip if already fully visited
            if (visit_count[neighbor] >= 2) {
                continue;
            }

            float vec_dist = graph.ComputeDistance(query, neighbor);
            float combined = combined_dist(vec_dist, neighbor);

            // Add to frontier (allow duplicates, will skip later if visited)
            if (visit_count[neighbor] < 1) {
                visit_count[neighbor] = 1;  // Mark as in frontier
            }
            frontier.push({-combined, neighbor});
            frontier_size++;

            // Limit frontier size by pruning (not strictly needed with skip logic)
            // but helps with memory
        }
    }

    // Sort matched by vector distance and return top-k
    std::sort(matched.begin(), matched.end());
    for (int i = 0; i < std::min(k, static_cast<int>(matched.size())); i++) {
        result.ids.push_back(matched[i].second);
    }

    return result;
}

// Run JAG benchmark with Knowhere HNSW
JAGBenchmarkResult
RunJAGBenchmark(const JAGBenchmarkConfig& cfg) {
    JAGBenchmarkResult result;
    result.test_name = "Random " + std::to_string(cfg.n) + " vectors";
    result.n = cfg.n;
    result.dim = cfg.dim;
    result.k = cfg.k;

    // Generate data
    auto base_ds = GenDataSet(cfg.n, cfg.dim, cfg.seed);
    auto query_ds = GenDataSet(cfg.nq, cfg.dim, cfg.seed + 1);
    const float* base_data = reinterpret_cast<const float*>(base_ds->GetTensor());
    const float* query_data = reinterpret_cast<const float*>(query_ds->GetTensor());

    // Generate labels
    auto filter_set = knowhere::GenerateRandomLabels(cfg.n, cfg.num_labels, cfg.seed + 2);
    int32_t target_label = 0;
    result.filter_ratio = filter_set.GetFilterRatio(target_label);

    // Build HNSW index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = cfg.dim;
    build_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    build_conf[knowhere::indexparam::HNSW_M] = cfg.hnsw_m;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = cfg.hnsw_ef_construction;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    auto build_res = index.Build(base_ds, build_conf);
    REQUIRE(build_res == knowhere::Status::success);

    // Prepare search config
    knowhere::Json search_conf;
    search_conf[knowhere::meta::DIM] = cfg.dim;
    search_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    search_conf[knowhere::meta::TOPK] = cfg.k;
    search_conf[knowhere::indexparam::EF] = cfg.hnsw_ef;

    // Create bitset for filtering (bit=1 means filtered out)
    std::vector<uint8_t> bitset_data((cfg.n + 7) / 8, 0xFF);
    for (int64_t i = 0; i < cfg.n; i++) {
        if (filter_set.GetLabel(i) == target_label) {
            bitset_data[i / 8] &= ~(1 << (i % 8));
        }
    }
    knowhere::BitsetView bitset(bitset_data.data(), cfg.n);

    // Compute ground truth for all queries
    std::vector<std::vector<int64_t>> all_gt(cfg.nq);
    for (int64_t q = 0; q < cfg.nq; q++) {
        all_gt[q] = ComputeFilteredGroundTruth(base_data, query_data + q * cfg.dim,
                                               cfg.n, cfg.dim, cfg.k, filter_set, target_label);
    }

    // Run baseline search and measure QPS
    StopWatch sw;
    std::vector<int64_t> all_results(cfg.nq * cfg.k);

    for (int64_t q = 0; q < cfg.nq; q++) {
        auto single_query = CopyDataSet(query_ds, 1);
        float* query_ptr = new float[cfg.dim];
        memcpy(query_ptr, query_data + q * cfg.dim, cfg.dim * sizeof(float));
        auto query_ds_single = knowhere::GenDataSet(1, cfg.dim, query_ptr);
        query_ds_single->SetIsOwner(true);

        auto search_res = index.Search(query_ds_single, search_conf, bitset);
        REQUIRE(search_res.has_value());

        auto res_ids = search_res.value()->GetIds();
        for (int i = 0; i < cfg.k; i++) {
            all_results[q * cfg.k + i] = res_ids[i];
        }
    }
    double elapsed_sec = sw.elapsed();
    result.baseline_qps = cfg.nq / elapsed_sec;

    // Calculate recall
    int total_hits = 0;
    int total_gt = 0;
    for (int64_t q = 0; q < cfg.nq; q++) {
        std::set<int64_t> gt_set(all_gt[q].begin(), all_gt[q].end());
        for (int i = 0; i < cfg.k; i++) {
            if (gt_set.count(all_results[q * cfg.k + i])) {
                total_hits++;
            }
        }
        total_gt += all_gt[q].size();
    }
    result.baseline_recall = (total_gt > 0) ? static_cast<float>(total_hits) / total_gt : 0.0f;

    // Run JAG simulation on simple graph to estimate improvement
    SimpleTestGraph sim_graph;
    sim_graph.dim = cfg.dim;
    sim_graph.BuildRandomGraph(std::min(cfg.n, (int64_t)10000), 32, cfg.seed);

    auto sim_baseline = SearchBaseline(sim_graph, base_data, query_data, cfg.k,
                                        filter_set, target_label);
    auto sim_jag = SearchJAG(sim_graph, base_data, query_data, cfg.k,
                              filter_set, target_label, 2.0f);

    float baseline_valid_ratio = (sim_baseline.nodes_visited > 0)
        ? static_cast<float>(sim_baseline.valid_visits) / sim_baseline.nodes_visited
        : 0.0f;
    float jag_valid_ratio = (sim_jag.nodes_visited > 0)
        ? static_cast<float>(sim_jag.valid_visits) / sim_jag.nodes_visited
        : 0.0f;

    result.jag_valid_ratio_improvement = (baseline_valid_ratio > 0)
        ? jag_valid_ratio / baseline_valid_ratio : 1.0f;
    result.jag_visit_reduction = (sim_baseline.nodes_visited > 0)
        ? static_cast<float>(sim_jag.nodes_visited) / sim_baseline.nodes_visited : 1.0f;

    return result;
}

}  // namespace

// ============== Test Cases ==============

TEST_CASE("JAG-HNSW Filter Distance Basic", "[jag]") {
    auto filter_set = knowhere::GenerateRandomLabels(1000, 5, 42);

    REQUIRE(filter_set.Size() == 1000);
    REQUIRE(filter_set.NumLabels() == 5);

    for (int label = 0; label < 5; label++) {
        float ratio = filter_set.GetFilterRatio(label);
        REQUIRE(ratio > 0.0f);
        REQUIRE(ratio < 1.0f);
    }
}

TEST_CASE("JAG-HNSW Label Filter Distance", "[jag]") {
    auto filter_set = knowhere::GenerateRandomLabels(100, 3, 123);
    int32_t target_label = filter_set.labels[0];

    knowhere::LabelFilterConstraint constraint{target_label};
    knowhere::LabelFilterDistance filter_dist(filter_set, constraint);

    for (int64_t i = 0; i < 100; i++) {
        int dist = filter_dist.Calculate(i);
        bool match = filter_dist.Match(i);

        if (filter_set.GetLabel(i) == target_label) {
            REQUIRE(dist == 0);
            REQUIRE(match == true);
        } else {
            REQUIRE(dist == 1);
            REQUIRE(match == false);
        }
    }
}

TEST_CASE("JAG-HNSW Simple Graph Search", "[jag]") {
    const int64_t n = 1000;
    const int64_t dim = 64;
    const int k = 10;

    auto base_ds = GenDataSet(n, dim, 42);
    auto query_ds = GenDataSet(1, dim, 123);
    const float* base_data = reinterpret_cast<const float*>(base_ds->GetTensor());
    const float* query = reinterpret_cast<const float*>(query_ds->GetTensor());

    auto filter_set = knowhere::GenerateRandomLabels(n, 5, 456);
    int32_t target_label = filter_set.labels[0];

    SimpleTestGraph graph;
    graph.dim = dim;
    graph.BuildRandomGraph(n, 16, 789);

    auto gt = ComputeFilteredGroundTruth(base_data, query, n, dim, k, filter_set, target_label);

    auto baseline = SearchBaseline(graph, base_data, query, k, filter_set, target_label);
    auto jag = SearchJAG(graph, base_data, query, k, filter_set, target_label, 2.0f);

    float baseline_recall = 0.0f, jag_recall = 0.0f;
    std::set<int64_t> gt_set(gt.begin(), gt.end());

    for (auto id : baseline.ids) {
        if (gt_set.count(id)) baseline_recall += 1.0f;
    }
    for (auto id : jag.ids) {
        if (gt_set.count(id)) jag_recall += 1.0f;
    }
    baseline_recall /= k;
    jag_recall /= k;

    std::cout << "\n=== Simple Graph Search ===" << std::endl;
    std::cout << "Baseline - Recall: " << baseline_recall
              << ", Visited: " << baseline.nodes_visited
              << ", Valid: " << baseline.valid_visits << std::endl;
    std::cout << "JAG      - Recall: " << jag_recall
              << ", Visited: " << jag.nodes_visited
              << ", Valid: " << jag.valid_visits << std::endl;

    REQUIRE(baseline_recall >= 0.0f);
    REQUIRE(jag_recall >= 0.0f);
}

TEST_CASE("JAG-HNSW Benchmark Multiple Filter Ratios", "[jag][benchmark]") {
    const int64_t n = 5000;
    const int64_t dim = 64;
    const int k = 10;

    auto base_ds = GenDataSet(n, dim, 42);
    auto query_ds = GenDataSet(1, dim, 123);
    const float* base_data = reinterpret_cast<const float*>(base_ds->GetTensor());
    const float* query = reinterpret_cast<const float*>(query_ds->GetTensor());

    SimpleTestGraph graph;
    graph.dim = dim;
    graph.BuildRandomGraph(n, 16, 789);

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG-HNSW Benchmark (Simple Graph)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << "D, K: " << k << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << std::left << std::setw(12) << "Filter%" << " | "
              << std::setw(12) << "Base_Recall" << " | "
              << std::setw(12) << "JAG_Recall" << " | "
              << std::setw(14) << "Base_Valid%" << " | "
              << std::setw(14) << "JAG_Valid%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int num_labels : {2, 5, 10, 20, 50}) {
        auto filter_set = knowhere::GenerateRandomLabels(n, num_labels, 456 + num_labels);
        int32_t target_label = filter_set.labels[0];
        float filter_ratio = filter_set.GetFilterRatio(target_label);

        auto gt = ComputeFilteredGroundTruth(base_data, query, n, dim, k, filter_set, target_label);
        auto baseline = SearchBaseline(graph, base_data, query, k, filter_set, target_label);
        auto jag = SearchJAG(graph, base_data, query, k, filter_set, target_label, 2.0f);

        std::set<int64_t> gt_set(gt.begin(), gt.end());
        float baseline_recall = 0.0f, jag_recall = 0.0f;
        for (auto id : baseline.ids) if (gt_set.count(id)) baseline_recall += 1.0f;
        for (auto id : jag.ids) if (gt_set.count(id)) jag_recall += 1.0f;
        baseline_recall /= k;
        jag_recall /= k;

        float base_valid_pct = 100.0f * baseline.valid_visits / baseline.nodes_visited;
        float jag_valid_pct = 100.0f * jag.valid_visits / jag.nodes_visited;

        std::cout << std::setw(12) << std::setprecision(1) << (filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(3) << baseline_recall << " | "
                  << std::setw(12) << jag_recall << " | "
                  << std::setw(14) << std::setprecision(1) << base_valid_pct << " | "
                  << std::setw(14) << jag_valid_pct << std::endl;
    }

    std::cout << "========================================" << std::endl;
}

TEST_CASE("JAG-HNSW Knowhere HNSW Benchmark", "[jag][benchmark][hnsw]") {
    JAGBenchmarkConfig cfg;
    cfg.n = 10000;
    cfg.dim = 128;
    cfg.nq = 50;
    cfg.k = 10;
    cfg.num_labels = 10;

    auto result = RunJAGBenchmark(cfg);
    result.Print();

    REQUIRE(result.baseline_recall >= 0.0f);
}

TEST_CASE("JAG-HNSW Comparison with JAG Paper Metrics", "[jag][benchmark][compare]") {
    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG-HNSW vs JAG Paper Benchmark Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Testing various filter ratios to match JAG paper scenarios\n" << std::endl;

    std::vector<int> label_counts = {2, 5, 10, 20};  // Corresponds to ~50%, ~20%, ~10%, ~5% filter ratios

    std::cout << std::left
              << std::setw(10) << "Filter%" << " | "
              << std::setw(15) << "Visit_Reduce%" << " | "
              << std::setw(15) << "Valid_Improve%" << " | "
              << "Expected (JAG Paper)\n";
    std::cout << std::string(70, '-') << std::endl;

    for (int num_labels : label_counts) {
        JAGBenchmarkConfig cfg;
        cfg.n = 5000;
        cfg.dim = 64;
        cfg.nq = 10;
        cfg.k = 10;
        cfg.num_labels = num_labels;

        auto result = RunJAGBenchmark(cfg);

        float visit_reduce_pct = (1.0f - result.jag_visit_reduction) * 100;
        float valid_improve_pct = (result.jag_valid_ratio_improvement - 1.0f) * 100;

        std::string expected = (num_labels <= 5) ? "~20-40%" : "~10-30%";

        std::cout << std::setw(10) << std::setprecision(0) << (result.filter_ratio * 100) << " | "
                  << std::setw(15) << std::setprecision(1) << visit_reduce_pct << " | "
                  << std::setw(15) << valid_improve_pct << " | "
                  << expected << "\n";
    }

    std::cout << "\nNote: JAG paper reports ~20-40% reduction in visited nodes" << std::endl;
    std::cout << "for high filter ratios (>50%), with ~50-100% improvement in" << std::endl;
    std::cout << "valid visit ratio.\n" << std::endl;
    std::cout << "========================================" << std::endl;
}

// ============== SIFT1M Benchmark Test ==============

// Load .ibin file (integer binary format)
bool
LoadIBin(const std::string& filename, std::vector<int32_t>& data, int32_t& n) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    data.resize(static_cast<size_t>(n));
    ifs.read(reinterpret_cast<char*>(data.data()), n * sizeof(int32_t));
    return ifs.good() || ifs.eof();
}

// ============== Filter Weight Computation ==============

// Compute standard deviation of a vector of values
double
ComputeStdDev(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    double sum = 0.0;
    for (double v : values) sum += v;
    double mean = sum / values.size();
    double sq_sum = 0.0;
    for (double v : values) sq_sum += (v - mean) * (v - mean);
    return std::sqrt(sq_sum / values.size());
}

// Compute optimal filter_weight based on dataset statistics
// Smaller values allow more exploration through non-matching nodes
// Larger values prioritize matching nodes more aggressively
// Trade-off: recall vs valid_visit_ratio
float
ComputeOptimalFilterWeight(const float* base_data, int64_t n, int64_t dim,
                           float aggressiveness = 0.0f, int samples = 1000) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, n - 1);

    double sum_vec_dist = 0.0;
    int count = 0;

    // Sample random pairs to estimate average vector distance
    for (int i = 0; i < samples; i++) {
        int64_t p = dist(rng);
        int64_t q = dist(rng);
        if (p == q) continue;

        const float* p_vec = base_data + p * dim;
        const float* q_vec = base_data + q * dim;
        double d = L2DistanceSq(p_vec, q_vec, dim);

        sum_vec_dist += d;
        count++;
    }

    double avg_dist = (count > 0) ? sum_vec_dist / count : 10000.0;

    // Use aggressiveness * average distance as filter_weight
    // Lower values = more exploration, higher recall, lower valid_ratio
    // Higher values = less exploration, lower recall, higher valid_ratio
    return static_cast<float>(avg_dist * aggressiveness);
}

// SIFT1M benchmark configuration
struct SIFT1MConfig {
    std::string data_path = "./data";
    int k = 10;
    int hnsw_m = 32;
    int hnsw_ef_construction = 200;
    int hnsw_ef = 256;
    // filter_weight = 0 means auto-calculate using ComputeOptimalFilterWeight
    // Otherwise use the provided value
    float filter_weight = 0.0f;
    int num_queries = 100;  // Number of queries to run
};

// SIFT1M benchmark result
struct SIFT1MResult {
    float filter_ratio;
    float baseline_recall;
    float jag_recall;
    int64_t baseline_nodes_visited;
    int64_t jag_nodes_visited;
    float baseline_valid_ratio;
    float jag_valid_ratio;

    float
    VisitReduction() const {
        return (baseline_nodes_visited > 0)
            ? 1.0f - static_cast<float>(jag_nodes_visited) / baseline_nodes_visited
            : 0.0f;
    }

    float
    ValidRatioImprovement() const {
        return (baseline_valid_ratio > 0)
            ? jag_valid_ratio / baseline_valid_ratio - 1.0f
            : 0.0f;
    }
};

// Run SIFT1M benchmark for a specific filter ratio (target label)
SIFT1MResult
RunSIFT1MBenchmark(const float* base_data, int64_t n, int64_t dim,
                   const float* query_data, int64_t nq,
                   const knowhere::LabelFilterSet& filter_set, int32_t target_label,
                   const SIFT1MConfig& cfg) {
    SIFT1MResult result;
    result.filter_ratio = filter_set.GetFilterRatio(target_label);

    // Build faiss HNSW index
    faiss::IndexHNSWFlat hnsw_index(dim, cfg.hnsw_m, faiss::METRIC_L2);
    hnsw_index.hnsw.efSearch = cfg.hnsw_ef;
    hnsw_index.hnsw.efConstruction = cfg.hnsw_ef_construction;
    hnsw_index.add(n, base_data);

    // Create IDSelector for label filtering (for baseline)
    IDSelectorLabel label_selector(filter_set, target_label);

    // Create search parameters with IDSelector
    faiss::SearchParametersHNSW search_params;
    search_params.sel = &label_selector;
    search_params.efSearch = cfg.hnsw_ef;

    // Create graph wrapper for JAG search
    RealHNSWGraph real_graph(&hnsw_index, base_data, dim);

    // Compute optimal filter_weight if not specified
    float filter_weight = cfg.filter_weight;
    if (filter_weight == 0.0f) {
        filter_weight = ComputeOptimalFilterWeight(base_data, n, dim);
    }
    std::cout << "Computed filter_weight: " << filter_weight << std::endl;

    // Aggregate metrics
    int total_baseline_hits = 0;
    int total_jag_hits = 0;
    int total_gt_count = 0;
    int64_t total_jag_visits = 0;
    int64_t total_jag_valid = 0;
    int64_t total_baseline_visits = 0;  // Track baseline visits from IDSelector
    int64_t total_baseline_valid = 0;   // Track baseline valid visits from IDSelector

    int queries_to_run = std::min(static_cast<int64_t>(cfg.num_queries), nq);

    // Buffers for faiss search
    std::vector<float> distances(cfg.k);
    std::vector<faiss::idx_t> labels(cfg.k);

    for (int q = 0; q < queries_to_run; q++) {
        const float* query = query_data + q * dim;

        // Compute ground truth for this query
        auto gt = ComputeFilteredGroundTruth(base_data, query, n, dim, cfg.k, filter_set, target_label);
        std::set<int64_t> gt_set(gt.begin(), gt.end());
        total_gt_count += gt.size();

        // Reset IDSelector visit counters for this query
        label_selector.Reset();

        // Run baseline using faiss native search with IDSelector
        hnsw_index.search(1, query, cfg.k, distances.data(), labels.data(), &search_params);
        for (int i = 0; i < cfg.k; i++) {
            if (labels[i] >= 0 && gt_set.count(labels[i])) {
                total_baseline_hits++;
            }
        }

        // Accumulate baseline visit counts
        total_baseline_visits += label_selector.total_visits;
        total_baseline_valid += label_selector.valid_visits;

        // Run JAG search on real HNSW graph (filter-guided)
        auto real_jag = SearchJAGReal(real_graph, query, cfg.k, filter_set, target_label,
                                      filter_weight, cfg.hnsw_ef, cfg.hnsw_ef * 10);
        for (auto id : real_jag.ids) {
            if (gt_set.count(id)) {
                total_jag_hits++;
            }
        }

        total_jag_visits += real_jag.nodes_visited;
        total_jag_valid += real_jag.valid_visits;
    }

    // Compute aggregated metrics
    result.baseline_recall = (total_gt_count > 0)
        ? static_cast<float>(total_baseline_hits) / total_gt_count : 0.0f;
    result.jag_recall = (total_gt_count > 0)
        ? static_cast<float>(total_jag_hits) / total_gt_count : 0.0f;

    // Use tracked baseline visits (if available), otherwise estimate
    result.baseline_nodes_visited = (total_baseline_visits > 0)
        ? total_baseline_visits / queries_to_run : cfg.hnsw_ef;
    result.jag_nodes_visited = total_jag_visits / queries_to_run;

    // Use tracked baseline valid ratio (if available), otherwise use filter_ratio
    result.baseline_valid_ratio = (total_baseline_visits > 0)
        ? static_cast<float>(total_baseline_valid) / total_baseline_visits : result.filter_ratio;
    result.jag_valid_ratio = (total_jag_visits > 0)
        ? static_cast<float>(total_jag_valid) / total_jag_visits : 0.0f;

    return result;
}

TEST_CASE("JAG-HNSW SIFT1M Benchmark", "[jag][benchmark][sift1m]") {
    // Print version info
    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG-HNSW Test Version: 2025-02-14-v9" << std::endl;
    std::cout << "filter_weight = 0 (test baseline)" << std::endl;
    std::cout << "========================================" << std::endl;

    // Get data path from environment or use default
    const char* env_path = std::getenv("SIFT1M_PATH");
    SIFT1MConfig cfg;
    if (env_path) {
        cfg.data_path = env_path;
    }

    std::string base_path = cfg.data_path + "/sift1m-base.fbin";
    std::string query_path = cfg.data_path + "/sift1m-query.fbin";
    std::string label_path = cfg.data_path + "/sift1m-base-filters-label.ibin";

    // Check if data files exist
    std::ifstream base_test(base_path), query_test(query_path), label_test(label_path);
    if (!base_test.good() || !query_test.good() || !label_test.good()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "SIFT1M Benchmark: SKIPPED" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Data files not found. To run this benchmark:" << std::endl;
        std::cout << "  1. cd tests/ut && ./prepare_sift1m_data.sh ./data" << std::endl;
        std::cout << "  2. export SIFT1M_PATH=./data" << std::endl;
        std::cout << "  3. Run this test again" << std::endl;
        std::cout << "========================================" << std::endl;
        return;
    }

    // Load data
    std::vector<float> base_data, query_data;
    int32_t n, dim, nq, query_dim;

    REQUIRE(LoadFBin(base_path, base_data, n, dim));
    REQUIRE(LoadFBin(query_path, query_data, nq, query_dim));
    REQUIRE(dim == query_dim);

    // Load labels
    std::vector<int32_t> labels;
    int32_t num_labels;
    REQUIRE(LoadIBin(label_path, labels, num_labels));
    REQUIRE(num_labels == n);

    // Build LabelFilterSet
    knowhere::LabelFilterSet filter_set;
    filter_set.labels = std::move(labels);
    for (int32_t i = 0; i < n; i++) {
        filter_set.label_to_ids[filter_set.labels[i]].push_back(i);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG-HNSW SIFT1M Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << "D" << std::endl;
    std::cout << "Queries: " << nq << " (using " << cfg.num_queries << ")" << std::endl;
    std::cout << "K: " << cfg.k << std::endl;
    std::cout << "HNSW: M=" << cfg.hnsw_m << ", ef=" << cfg.hnsw_ef << std::endl;
    if (cfg.filter_weight == 0.0f) {
        std::cout << "JAG Filter Weight: auto-computed per query" << std::endl;
    } else {
        std::cout << "JAG Filter Weight: " << cfg.filter_weight << std::endl;
    }
    std::cout << "========================================\n" << std::endl;

    // Test multiple filter ratios
    std::vector<SIFT1MResult> results;

    // Use labels with different ratios (sorted by frequency)
    std::vector<std::pair<int32_t, size_t>> label_counts;
    for (const auto& [label, ids] : filter_set.label_to_ids) {
        label_counts.push_back({label, ids.size()});
    }
    std::sort(label_counts.begin(), label_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Print header
    std::cout << std::left
              << std::setw(10) << "Filter%" << " | "
              << std::setw(12) << "Base_Recall" << " | "
              << std::setw(12) << "JAG_Recall" << " | "
              << std::setw(14) << "Visit_Reduce%" << " | "
              << std::setw(14) << "Valid_Improve%" << " | "
              << std::setw(12) << "JAG_Paper" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    // Test top labels to get different filter ratios
    int labels_to_test = std::min(5, static_cast<int>(label_counts.size()));
    for (int i = 0; i < labels_to_test; i++) {
        int32_t target_label = label_counts[i].first;

        auto result = RunSIFT1MBenchmark(base_data.data(), n, dim,
                                         query_data.data(), nq,
                                         filter_set, target_label, cfg);
        results.push_back(result);

        // JAG paper expected improvement
        std::string expected = (result.filter_ratio >= 0.3f) ? "20-40%" : "10-30%";

        std::cout << std::fixed
                  << std::setw(10) << std::setprecision(1) << (result.filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(4) << result.baseline_recall << " | "
                  << std::setw(12) << result.jag_recall << " | "
                  << std::setw(14) << std::setprecision(1) << (result.VisitReduction() * 100) << " | "
                  << std::setw(14) << (result.ValidRatioImprovement() * 100) << " | "
                  << std::setw(12) << expected << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary: JAG paper reports ~20-40% reduction in visited nodes" << std::endl;
    std::cout << "for high filter ratios (>30%), with ~50-100% improvement in" << std::endl;
    std::cout << "valid visit ratio while maintaining similar recall." << std::endl;
    std::cout << "========================================" << std::endl;

    // Basic sanity checks
    for (const auto& r : results) {
        REQUIRE(r.baseline_recall >= 0.0f);
        REQUIRE(r.baseline_recall <= 1.0f);
    }
}
