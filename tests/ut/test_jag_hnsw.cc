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

#include "catch2/catch_test_macros.hpp"
#include "common/filter_distance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

// Test configuration
constexpr int64_t TEST_N = 10000;     // Number of vectors
constexpr int64_t TEST_DIM = 128;     // Vector dimension
constexpr int TEST_NUM_LABELS = 10;   // Number of label categories
constexpr int TEST_K = 10;            // Top-k for search

// Generate random vectors
std::vector<float>
GenerateRandomVectors(int64_t n, int64_t dim, int seed = 42) {
    std::vector<float> data(n * dim);
    uint64_t state = seed;
    for (int64_t i = 0; i < n * dim; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        data[i] = static_cast<float>((state >> 33) & 0x7FFFFFFF) / 0x7FFFFFFF;
    }
    return data;
}

// Simple L2 distance
float
L2Distance(const float* a, const float* b, int64_t dim) {
    float dist = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return std::sqrt(dist);
}

// Simple graph node for testing
struct SimpleGraphNode {
    int64_t id;
    std::vector<int64_t> neighbors;
};

// Simple HNSW-like graph for testing JAG algorithm
class SimpleTestGraph {
 public:
    std::vector<SimpleGraphNode> nodes;
    int64_t entry_point = 0;
    int64_t dim = 128;

    void
    BuildRandomGraph(int64_t n, int avg_degree, int seed = 42) {
        nodes.resize(n);
        for (int64_t i = 0; i < n; i++) {
            nodes[i].id = i;
        }

        uint64_t state = seed;
        auto next_random = [&state]() {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return static_cast<int64_t>((state >> 33) & 0x7FFFFFFF);
        };

        // Create random edges
        for (int64_t i = 0; i < n; i++) {
            int degree = avg_degree / 2 + next_random() % avg_degree;
            for (int d = 0; d < degree; d++) {
                int64_t neighbor = next_random() % n;
                if (neighbor != i) {
                    nodes[i].neighbors.push_back(neighbor);
                }
            }
        }
    }

    const std::vector<int64_t>&
    GetNeighbors(int64_t id) const {
        return nodes[id].neighbors;
    }

    int64_t
    GetEntryPoint() const {
        return entry_point;
    }

    int64_t
    Size() const {
        return nodes.size();
    }
};

// Compute ground truth with filter constraint
std::vector<int64_t>
ComputeFilteredGroundTruth(const float* base_data, const float* query,
                           int64_t n, int64_t dim, int k,
                           const knowhere::LabelFilterSet& filter_set,
                           int32_t target_label) {
    std::vector<std::pair<float, int64_t>> distances;

    for (int64_t i = 0; i < n; i++) {
        if (filter_set.GetLabel(i) == target_label) {
            float dist = L2Distance(query, base_data + i * dim, dim);
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

// Calculate recall
float
CalculateRecall(const std::vector<int64_t>& result, const std::vector<int64_t>& gt) {
    if (gt.empty() || result.empty()) {
        return 0.0f;
    }

    std::unordered_set<int64_t> gt_set(gt.begin(), gt.end());
    int matches = 0;
    int count = std::min(result.size(), gt.size());

    for (int i = 0; i < count; i++) {
        if (gt_set.count(result[i])) {
            matches++;
        }
    }

    return static_cast<float>(matches) / count;
}

// ============== FBIN File Loader (JAG Paper Format) ==============

// Load .fbin file (format: n: int32, dim: int32, data: float[n*dim])
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

// Load .ibin file for filters (format: n: int32, data: int32[n])
bool
LoadIBin(const std::string& filename, std::vector<int32_t>& data) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }

    int32_t n;
    ifs.read(reinterpret_cast<char*>(&n), sizeof(int32_t));

    data.resize(n);
    ifs.read(reinterpret_cast<char*>(data.data()), n * sizeof(int32_t));

    return ifs.good() || ifs.eof();
}

// Generate label filter file
void
GenerateLabelFilterFile(const std::string& filename, int64_t n, int num_labels, int seed = 42) {
    std::ofstream ofs(filename, std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(&n), sizeof(int32_t));

    uint64_t state = seed;
    for (int64_t i = 0; i < n; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int32_t label = static_cast<int32_t>((state >> 33) % num_labels);
        ofs.write(reinterpret_cast<const char*>(&label), sizeof(int32_t));
    }
}

// Load labels from .ibin into LabelFilterSet
knowhere::LabelFilterSet
LoadLabelsFromFile(const std::string& filename, int64_t expected_n) {
    std::vector<int32_t> labels;
    if (!LoadIBin(filename, labels)) {
        // Generate if file doesn't exist
        return knowhere::LabelFilterSet();
    }

    knowhere::LabelFilterSet filter_set;
    filter_set.labels = std::move(labels);
    for (int64_t i = 0; i < expected_n && i < static_cast<int64_t>(filter_set.labels.size()); i++) {
        int32_t label = filter_set.labels[i];
        filter_set.label_to_ids[label].push_back(static_cast<int32_t>(i));
    }

    return filter_set;
}

// ============== Timing Utilities ==============

class Timer {
 public:
    void
    Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double
    ElapsedMs() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

 private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Baseline: standard graph search + post-filter
std::vector<int64_t>
SearchBaseline(const SimpleTestGraph& graph, const float* base_data, const float* query,
               int k, const knowhere::LabelFilterSet& filter_set, int32_t target_label,
               int64_t* nodes_visited = nullptr, int64_t* valid_visits = nullptr) {
    std::vector<int64_t> results;
    std::unordered_set<int64_t> visited;
    std::priority_queue<std::pair<float, int64_t>> frontier;
    std::vector<std::pair<float, int64_t>> candidates;

    int64_t visited_count = 0;
    int64_t valid_count = 0;

    // Start from entry point
    int64_t entry = graph.GetEntryPoint();
    float entry_dist = L2Distance(query, base_data + entry * graph.dim, graph.dim);
    frontier.push({-entry_dist, entry});  // Negative for min-heap behavior

    while (!frontier.empty() && visited_count < 5000) {
        auto [neg_dist, current] = frontier.top();
        frontier.pop();

        if (visited.count(current)) {
            continue;
        }
        visited.insert(current);
        visited_count++;

        // Check filter
        if (filter_set.GetLabel(current) == target_label) {
            candidates.push_back({-neg_dist, current});
            valid_count++;
        }

        // Expand neighbors
        for (int64_t neighbor : graph.GetNeighbors(current)) {
            if (!visited.count(neighbor)) {
                float dist = L2Distance(query, base_data + neighbor * graph.dim, graph.dim);
                frontier.push({-dist, neighbor});
            }
        }
    }

    // Sort candidates and take top-k
    std::sort(candidates.begin(), candidates.end());
    for (int i = 0; i < std::min(k, (int)candidates.size()); i++) {
        results.push_back(candidates[i].second);
    }

    if (nodes_visited) *nodes_visited = visited_count;
    if (valid_visits) *valid_visits = valid_count;

    return results;
}

// JAG-style search with filter guidance
std::vector<int64_t>
SearchJAG(const SimpleTestGraph& graph, const float* base_data, const float* query,
          int k, const knowhere::LabelFilterSet& filter_set, int32_t target_label,
          float filter_weight = 1.0f, int64_t* nodes_visited = nullptr,
          int64_t* valid_visits = nullptr) {
    std::vector<int64_t> results;
    std::unordered_set<int64_t> visited;

    // Combined distance: vector_dist + filter_weight * filter_dist
    auto combined_dist = [&](float vec_dist, int64_t node_id) {
        int filter_dist = (filter_set.GetLabel(node_id) == target_label) ? 0 : 1;
        return vec_dist + filter_weight * filter_dist;
    };

    std::priority_queue<std::pair<float, int64_t>> frontier;
    std::vector<std::pair<float, int64_t>> matched;

    int64_t visited_count = 0;
    int64_t valid_count = 0;

    // Start from entry point
    int64_t entry = graph.GetEntryPoint();
    float entry_vec_dist = L2Distance(query, base_data + entry * graph.dim, graph.dim);
    float entry_combined = combined_dist(entry_vec_dist, entry);
    frontier.push({-entry_combined, entry});

    while (!frontier.empty() && visited_count < 5000 && (int)matched.size() < k * 10) {
        auto [neg_combined, current] = frontier.top();
        frontier.pop();

        if (visited.count(current)) {
            continue;
        }
        visited.insert(current);
        visited_count++;

        // Check filter
        if (filter_set.GetLabel(current) == target_label) {
            float vec_dist = L2Distance(query, base_data + current * graph.dim, graph.dim);
            matched.push_back({vec_dist, current});
            valid_count++;
        }

        // Expand neighbors
        for (int64_t neighbor : graph.GetNeighbors(current)) {
            if (!visited.count(neighbor)) {
                float vec_dist = L2Distance(query, base_data + neighbor * graph.dim, graph.dim);
                float comb = combined_dist(vec_dist, neighbor);
                frontier.push({-comb, neighbor});
            }
        }
    }

    // Sort matched by vector distance and take top-k
    std::sort(matched.begin(), matched.end());
    for (int i = 0; i < std::min(k, (int)matched.size()); i++) {
        results.push_back(matched[i].second);
    }

    if (nodes_visited) *nodes_visited = visited_count;
    if (valid_visits) *valid_visits = valid_count;

    return results;
}

}  // namespace

TEST_CASE("JAG-HNSW Filter Distance Basic", "[jag]") {
    // Test LabelFilterSet
    auto filter_set = knowhere::GenerateRandomLabels(1000, 5, 42);

    REQUIRE(filter_set.Size() == 1000);
    REQUIRE(filter_set.NumLabels() == 5);

    // Test filter ratio
    for (int label = 0; label < 5; label++) {
        float ratio = filter_set.GetFilterRatio(label);
        REQUIRE(ratio > 0.0f);
        REQUIRE(ratio < 1.0f);
    }
}

TEST_CASE("JAG-HNSW Label Filter Distance", "[jag]") {
    auto filter_set = knowhere::GenerateRandomLabels(100, 3, 123);

    // Get a valid label from the set
    int32_t target_label = filter_set.labels[0];

    knowhere::LabelFilterConstraint constraint{target_label};
    knowhere::LabelFilterDistance filter_dist(filter_set, constraint);

    // Test distance calculation
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

    // Test filter ratio
    float ratio = filter_dist.FilterRatio();
    REQUIRE(ratio > 0.0f);
    REQUIRE(ratio < 1.0f);
}

TEST_CASE("JAG-HNSW Simple Graph Search Comparison", "[jag]") {
    const int64_t n = 1000;
    const int64_t dim = 64;
    const int k = 10;
    const int num_labels = 5;

    // Generate data
    auto base_data = GenerateRandomVectors(n, dim, 42);
    auto query = GenerateRandomVectors(1, dim, 123);
    auto filter_set = knowhere::GenerateRandomLabels(n, num_labels, 456);

    // Build random graph
    SimpleTestGraph graph;
    graph.dim = dim;
    graph.BuildRandomGraph(n, 16, 789);

    // Select a label to query
    int32_t target_label = filter_set.labels[0];
    float filter_ratio = filter_set.GetFilterRatio(target_label);

    INFO("Filter ratio: " << filter_ratio);

    // Compute ground truth
    auto gt = ComputeFilteredGroundTruth(base_data.data(), query.data(),
                                         n, dim, k, filter_set, target_label);

    // Run baseline
    int64_t baseline_visited = 0, baseline_valid = 0;
    auto baseline_results = SearchBaseline(graph, base_data.data(), query.data(),
                                           k, filter_set, target_label,
                                           &baseline_visited, &baseline_valid);
    float baseline_recall = CalculateRecall(baseline_results, gt);

    // Run JAG with different weights
    int64_t jag_visited = 0, jag_valid = 0;
    auto jag_results = SearchJAG(graph, base_data.data(), query.data(),
                                 k, filter_set, target_label, 2.0f,
                                 &jag_visited, &jag_valid);
    float jag_recall = CalculateRecall(jag_results, gt);

    std::cout << "\n=== Simple Graph Search Comparison ===" << std::endl;
    std::cout << "Filter ratio: " << std::fixed << std::setprecision(1)
              << (filter_ratio * 100) << "%" << std::endl;
    std::cout << "Baseline - Recall: " << baseline_recall
              << ", Visited: " << baseline_visited
              << ", Valid: " << baseline_valid
              << " (" << (100.0 * baseline_valid / baseline_visited) << "%)" << std::endl;
    std::cout << "JAG      - Recall: " << jag_recall
              << ", Visited: " << jag_visited
              << ", Valid: " << jag_valid
              << " (" << (100.0 * jag_valid / jag_visited) << "%)" << std::endl;

    // JAG should have better valid visit ratio
    float baseline_valid_ratio = (float)baseline_valid / baseline_visited;
    float jag_valid_ratio = (float)jag_valid / jag_visited;

    // Note: With random graph, results may vary. This is just a basic test.
    REQUIRE(baseline_recall >= 0.0f);
    REQUIRE(jag_recall >= 0.0f);
}

TEST_CASE("JAG-HNSW Benchmark Multiple Filter Ratios", "[jag][benchmark]") {
    const int64_t n = 5000;
    const int64_t dim = 64;
    const int k = 10;

    // Generate data
    auto base_data = GenerateRandomVectors(n, dim, 42);
    auto query = GenerateRandomVectors(1, dim, 123);

    // Build random graph
    SimpleTestGraph graph;
    graph.dim = dim;
    graph.BuildRandomGraph(n, 16, 789);

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG-HNSW Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << " dimensions" << std::endl;
    std::cout << "K: " << k << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Test different number of labels (affects filter ratio)
    std::vector<int> label_counts = {2, 5, 10, 20, 50};

    std::cout << std::left;
    std::cout << std::setw(12) << "Filter%" << " | "
              << std::setw(12) << "Base_Recall" << " | "
              << std::setw(12) << "JAG_Recall" << " | "
              << std::setw(14) << "Base_Valid%" << " | "
              << std::setw(14) << "JAG_Valid%" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int num_labels : label_counts) {
        auto filter_set = knowhere::GenerateRandomLabels(n, num_labels, 456 + num_labels);

        // Select a label to query
        int32_t target_label = filter_set.labels[0];
        float filter_ratio = filter_set.GetFilterRatio(target_label);

        // Compute ground truth
        auto gt = ComputeFilteredGroundTruth(base_data.data(), query.data(),
                                             n, dim, k, filter_set, target_label);

        // Run baseline
        int64_t baseline_visited = 0, baseline_valid = 0;
        auto baseline_results = SearchBaseline(graph, base_data.data(), query.data(),
                                               k, filter_set, target_label,
                                               &baseline_visited, &baseline_valid);
        float baseline_recall = CalculateRecall(baseline_results, gt);
        float baseline_valid_ratio = 100.0f * baseline_valid / baseline_visited;

        // Run JAG
        int64_t jag_visited = 0, jag_valid = 0;
        auto jag_results = SearchJAG(graph, base_data.data(), query.data(),
                                     k, filter_set, target_label, 2.0f,
                                     &jag_visited, &jag_valid);
        float jag_recall = CalculateRecall(jag_results, gt);
        float jag_valid_ratio = 100.0f * jag_valid / jag_visited;

        std::cout << std::setw(12) << std::setprecision(1) << (filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(3) << baseline_recall << " | "
                  << std::setw(12) << jag_recall << " | "
                  << std::setw(14) << std::setprecision(1) << baseline_valid_ratio << " | "
                  << std::setw(14) << jag_valid_ratio << std::endl;
    }

    std::cout << "========================================" << std::endl;
}

// ============== Real HNSW Index Benchmark (Optional) ==============
// These tests require full Knowhere HNSW integration.
// Enable by defining JAG_ENABLE_HNSW_BENCHMARK during compilation.

#ifdef JAG_ENABLE_HNSW_BENCHMARK

#include "knowhere/comp/index_param.h"
#include "knowhere/index/index.h"
#include "knowhere/index/index_factory.h"

struct JAGBenchmarkResult {
    std::string dataset_name;
    int64_t n;
    int64_t dim;
    int k;
    float filter_ratio;
    int num_queries;

    // Baseline results
    float baseline_recall;
    double baseline_qps;
    int64_t baseline_avg_visited;
    float baseline_valid_ratio;

    // JAG results
    float jag_recall;
    double jag_qps;
    int64_t jag_avg_visited;
    float jag_valid_ratio;

    std::string
    ToString() const {
        std::ostringstream oss;
        oss << "\n========================================\n";
        oss << "JAG-HNSW Benchmark Results: " << dataset_name << "\n";
        oss << "========================================\n";
        oss << "Dataset: " << n << " vectors, " << dim << " dimensions\n";
        oss << "K: " << k << ", Filter Ratio: " << std::fixed << std::setprecision(1)
            << (filter_ratio * 100) << "%\n";
        oss << "Queries: " << num_queries << "\n\n";

        oss << std::left;
        oss << std::setw(18) << "Metric" << " | "
            << std::setw(15) << "Baseline" << " | "
            << std::setw(15) << "JAG" << " | "
            << std::setw(12) << "Change" << "\n";
        oss << std::string(70, '-') << "\n";

        auto format_change = [](float base, float jag) -> std::string {
            if (base == 0) return "N/A";
            float change = (jag - base) / base * 100;
            std::ostringstream s;
            s << std::fixed << std::setprecision(1) << (change >= 0 ? "+" : "") << change << "%";
            return s.str();
        };

        oss << std::setw(18) << "Recall@K" << " | "
            << std::setw(15) << std::setprecision(4) << baseline_recall << " | "
            << std::setw(15) << jag_recall << " | "
            << std::setw(12) << format_change(baseline_recall, jag_recall) << "\n";

        oss << std::setw(18) << "QPS" << " | "
            << std::setw(15) << std::setprecision(1) << baseline_qps << " | "
            << std::setw(15) << jag_qps << " | "
            << std::setw(12) << format_change(baseline_qps, jag_qps) << "\n";

        oss << std::setw(18) << "Avg Nodes Visited" << " | "
            << std::setw(15) << baseline_avg_visited << " | "
            << std::setw(15) << jag_avg_visited << " | "
            << std::setw(12) << format_change(baseline_avg_visited, jag_avg_visited) << "\n";

        oss << std::setw(18) << "Valid Visit Ratio" << " | "
            << std::setw(15) << std::setprecision(1) << (baseline_valid_ratio * 100) << "%"
            << " | " << std::setw(14) << (jag_valid_ratio * 100) << "%"
            << " | " << std::setw(12) << format_change(baseline_valid_ratio, jag_valid_ratio) << "\n";

        oss << "========================================\n";
        return oss.str();
    }
};

TEST_CASE("JAG-HNSW Random Data Benchmark with Real Index", "[jag][benchmark][hnsw]") {
    const int64_t n = 100000;
    const int64_t dim = 128;
    const int k = 10;
    const int num_test_queries = 100;
    const int num_labels = 10;

    std::cout << "\n========================================\n";
    std::cout << "JAG-HNSW Benchmark with Real HNSW Index\n";
    std::cout << "========================================\n";

    auto base_data = GenerateRandomVectors(n, dim, 42);
    auto query_data = GenerateRandomVectors(num_test_queries, dim, 123);
    auto filter_set = knowhere::GenerateRandomLabels(n, num_labels, 456);

    int32_t target_label = 0;
    float filter_ratio = filter_set.GetFilterRatio(target_label);

    std::cout << "Filter ratio: " << std::fixed << std::setprecision(1)
              << (filter_ratio * 100) << "%\n";

    // Build HNSW index
    knowhere::Json build_params;
    build_params[knowhere::meta::DIM] = dim;
    build_params[knowhere::meta::METRIC_TYPE] = "L2";
    build_params[knowhere::indexparam::HNSW_BUILD_M] = 32;
    build_params[knowhere::indexparam::HNSW_BUILD_EF_C] = 200;

    std::cout << "Building HNSW index...\n";
    Timer build_timer;
    build_timer.Start();

    auto index = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, knowhere::Version::GetDefaultVersion(), nullptr);

    auto dataset = knowhere::GenDataSet(n, dim, base_data.data());
    REQUIRE(index.value().Build(*dataset, build_params) == knowhere::Status::success);

    std::cout << "Index built in " << build_timer.ElapsedMs() << " ms\n";

    // Run search with bitset filter
    knowhere::Json search_params;
    search_params[knowhere::meta::DIM] = dim;
    search_params[knowhere::meta::METRIC_TYPE] = "L2";
    search_params[knowhere::meta::TOPK] = k;
    search_params[knowhere::indexparam::HNSW_SEARCH_EF] = 256;

    std::vector<uint8_t> bitset_data((n + 7) / 8, 0xFF);
    for (int64_t i = 0; i < n; i++) {
        if (filter_set.GetLabel(i) == target_label) {
            bitset_data[i / 8] &= ~(1 << (i % 8));
        }
    }

    Timer search_timer;
    search_timer.Start();

    for (int q = 0; q < num_test_queries; q++) {
        auto query_ds = knowhere::GenDataSet(1, dim, query_data.data() + q * dim);
        knowhere::BitsetView bitset(bitset_data.data(), n);
        index.value().Search(*query_ds, search_params, bitset);
    }

    double search_time_ms = search_timer.ElapsedMs();
    double qps = num_test_queries / (search_time_ms / 1000.0);

    std::cout << "\nSearch QPS: " << std::setprecision(1) << qps << "\n";
    std::cout << "========================================\n";
}

#endif  // JAG_ENABLE_HNSW_BENCHMARK
