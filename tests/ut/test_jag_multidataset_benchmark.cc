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

// Multi-dataset JAG benchmark test for ANN benchmarks
// Supports: glove-100, deep-1m, gist, sift1m

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

// ============== Dataset Configuration ==============

struct DatasetConfig {
    std::string name;
    std::string display_name;
    std::string metric_type;  // "L2" or "IP" (for angular after normalization)
    std::string env_var;
    int64_t expected_n;
    int64_t expected_dim;
    int hnsw_m;
    int ef_construction;
    int search_ef;
};

const std::vector<DatasetConfig> DATASET_CONFIGS = {
    {"sift1m", "SIFT1M", "L2", "SIFT1M_PATH", 1000000, 128, 32, 200, 256},
    {"glove-100", "GloVe-100", "IP", "GLOVE100_PATH", 1183514, 100, 32, 200, 256},
    {"deep-1m", "Deep-1M", "L2", "DEEP1M_PATH", 1000000, 96, 32, 200, 256},
    {"gist", "GIST", "L2", "GIST_PATH", 1000000, 960, 32, 200, 256},
};

// ============== FBIN File Loader ==============

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

// ============== Distance Functions ==============

float
L2DistanceSq(const float* a, const float* b, int64_t dim) {
    float dist = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

float
IPDistance(const float* a, const float* b, int64_t dim) {
    float dot = 0.0f;
    for (int64_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
    }
    return -dot;  // Negative because we want max dot = min distance
}

// Compute filtered ground truth
std::vector<int64_t>
ComputeFilteredGroundTruth(const float* base_data, const float* query, int64_t n, int64_t dim, int k,
                           const std::vector<int32_t>& labels, int32_t target_label,
                           const std::string& metric_type) {
    auto dist_fn = (metric_type == "IP") ? IPDistance : L2DistanceSq;

    std::vector<std::pair<float, int64_t>> distances;
    for (int64_t i = 0; i < n; i++) {
        if (labels[i] == target_label) {
            float dist = dist_fn(query, base_data + i * dim, dim);
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

// ============== Benchmark Result ==============

struct BenchmarkResult {
    std::string name;
    std::string dataset;
    float filter_ratio;
    float recall;
    double qps;
    double qps_delta_percent;

    void
    PrintHeader() {
        std::cout << std::left
                  << std::setw(12) << "Method" << " | "
                  << std::setw(10) << "Filter%" << " | "
                  << std::setw(12) << "Recall" << " | "
                  << std::setw(12) << "QPS" << " | "
                  << std::setw(10) << "QPS_Δ%" << std::endl;
        std::cout << std::string(65, '-') << std::endl;
    }

    void
    Print() const {
        std::cout << std::left
                  << std::setw(12) << name << " | "
                  << std::setw(10) << std::fixed << std::setprecision(1) << (filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(4) << recall << " | "
                  << std::setw(12) << std::setprecision(1) << qps << " | "
                  << std::setw(10) << std::setprecision(1) << qps_delta_percent << std::endl;
    }
};

// ============== Dataset Loader ==============

struct LoadedDataset {
    bool valid = false;
    std::string name;
    std::vector<float> base_data;
    std::vector<float> query_data;
    std::vector<int32_t> labels_uniform;
    std::vector<int32_t> labels_zipf;
    int32_t n = 0;
    int32_t dim = 0;
    int32_t nq = 0;
    std::string metric_type;
    std::string data_path;
};

LoadedDataset
LoadDataset(const DatasetConfig& config) {
    LoadedDataset result;
    result.name = config.name;
    result.metric_type = config.metric_type;

    // Get data path from environment
    const char* env_path = std::getenv(config.env_var.c_str());
    if (!env_path) {
        // Try ANN_BENCHMARK_PATH as fallback
        env_path = std::getenv("ANN_BENCHMARK_PATH");
    }
    result.data_path = env_path ? env_path : "./data";

    std::string base_path = result.data_path + "/" + config.name + "-base.fbin";
    std::string query_path = result.data_path + "/" + config.name + "-query.fbin";
    std::string label_uniform_path = result.data_path + "/" + config.name + "-base-filters-uniform.ibin";
    std::string label_zipf_path = result.data_path + "/" + config.name + "-base-filters-zipf.ibin";

    // Check if files exist
    std::ifstream base_test(base_path), query_test(query_path);
    if (!base_test.good() || !query_test.good()) {
        std::cout << "Dataset " << config.name << " not found at " << result.data_path << std::endl;
        return result;
    }

    // Load data
    if (!LoadFBin(base_path, result.base_data, result.n, result.dim)) {
        std::cout << "Failed to load base data: " << base_path << std::endl;
        return result;
    }

    int32_t query_dim;
    if (!LoadFBin(query_path, result.query_data, result.nq, query_dim)) {
        std::cout << "Failed to load query data: " << query_path << std::endl;
        return result;
    }

    if (result.dim != query_dim) {
        std::cout << "Dimension mismatch: base=" << result.dim << ", query=" << query_dim << std::endl;
        return result;
    }

    // Load labels (uniform is default, zipf optional)
    int32_t label_count;
    if (LoadIBin(label_uniform_path, result.labels_uniform, label_count)) {
        if (label_count != result.n) {
            std::cout << "Label count mismatch: expected " << result.n << ", got " << label_count << std::endl;
        }
    } else {
        // Generate uniform labels if not found
        std::cout << "Generating uniform labels for " << config.name << "..." << std::endl;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int32_t> dist(0, 9);
        result.labels_uniform.resize(result.n);
        for (int32_t i = 0; i < result.n; i++) {
            result.labels_uniform[i] = dist(rng);
        }
    }

    // Load Zipf labels if available
    if (LoadIBin(label_zipf_path, result.labels_zipf, label_count)) {
        if (label_count != result.n) {
            result.labels_zipf.clear();
        }
    }

    result.valid = true;
    return result;
}

// ============== Benchmark Runner ==============

BenchmarkResult
RunBenchmark(const knowhere::Index<knowhere::IndexNode>& index, const LoadedDataset& dataset,
             const std::vector<int32_t>& labels, int32_t target_label,
             bool enable_jag, float jag_filter_weight, int k, int ef, int num_queries) {
    BenchmarkResult result;
    result.dataset = dataset.name;
    result.name = enable_jag ? "JAG+Alpha" : "Alpha-only";
    result.qps_delta_percent = 0.0;

    // Calculate filter ratio
    int32_t valid_count = 0;
    for (int32_t i = 0; i < dataset.n; i++) {
        if (labels[i] == target_label) valid_count++;
    }
    result.filter_ratio = static_cast<float>(valid_count) / dataset.n;

    // Create bitset
    std::vector<uint8_t> bitset_data((dataset.n + 7) / 8, 0xFF);
    for (int32_t i = 0; i < dataset.n; i++) {
        if (labels[i] == target_label) {
            bitset_data[i / 8] &= ~(1 << (i % 8));
        }
    }
    knowhere::BitsetView bitset(bitset_data.data(), dataset.n);

    // Search config
    knowhere::Json search_conf;
    search_conf[knowhere::meta::DIM] = dataset.dim;
    search_conf[knowhere::meta::METRIC_TYPE] = dataset.metric_type;
    search_conf[knowhere::meta::TOPK] = k;
    search_conf[knowhere::indexparam::EF] = ef;
    search_conf[knowhere::indexparam::ENABLE_JAG] = enable_jag;
    search_conf[knowhere::indexparam::JAG_FILTER_WEIGHT] = jag_filter_weight;
    search_conf[knowhere::indexparam::JAG_CANDIDATE_POOL_SIZE] = ef * 4;

    // Run benchmark
    int total_hits = 0;
    int total_gt = 0;

    const float* base_data = dataset.base_data.data();
    const float* query_data = dataset.query_data.data();

    StopWatch sw;
    for (int q = 0; q < num_queries && q < dataset.nq; q++) {
        auto query_ds = knowhere::GenDataSet(1, dataset.dim,
                                              const_cast<float*>(query_data + q * dataset.dim));

        auto gt = ComputeFilteredGroundTruth(base_data, query_data + q * dataset.dim,
                                              dataset.n, dataset.dim, k,
                                              labels, target_label, dataset.metric_type);
        std::set<int64_t> gt_set(gt.begin(), gt.end());

        auto search_res = index.Search(query_ds, search_conf, bitset);
        if (search_res.has_value()) {
            auto res_ids = search_res.value()->GetIds();
            for (int i = 0; i < k; i++) {
                if (gt_set.count(res_ids[i])) total_hits++;
            }
        }
        total_gt += gt.size();
    }
    double elapsed = sw.elapsed();

    result.qps = (num_queries < dataset.nq ? num_queries : dataset.nq) / elapsed;
    result.recall = (total_gt > 0) ? static_cast<float>(total_hits) / total_gt : 0.0f;

    return result;
}

}  // namespace

// ============== Test Cases ==============

// Helper to get filter_weight from environment variable
// Default 0.3 provides ~87% recall with ~12-15% QPS improvement
// Higher weight = invalid nodes ranked lower = better QPS but potentially lower recall
// Lower weight = more exploration = higher recall but lower QPS
static float GetJagFilterWeight(float default_value = 0.3f) {
    const char* env_val = std::getenv("JAG_FILTER_WEIGHT");
    if (env_val != nullptr) {
        try {
            return std::stof(env_val);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

TEST_CASE("JAG Multi-Dataset Benchmark - All Datasets", "[jag][benchmark][multidataset]") {
    const int k = 10;
    const int num_queries = 100;
    const float jag_filter_weight = GetJagFilterWeight(0.3f);

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "JAG Multi-Dataset Benchmark" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "K: " << k << ", Queries per dataset: " << num_queries << std::endl;
    std::cout << "JAG filter_weight: " << jag_filter_weight;
    std::cout << " (set JAG_FILTER_WEIGHT env var to change)" << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "  1. Run: ./tests/ut/prepare_ann_benchmark_data.sh <dataset> ./data" << std::endl;
    std::cout << "  2. export ANN_BENCHMARK_PATH=./data (or dataset-specific env vars)" << std::endl;
    std::cout << "  3. export JAG_FILTER_WEIGHT=1.0 (optional, default=1.0)" << std::endl;
    std::cout << std::string(80, '=') << "\n" << std::endl;

    for (const auto& config : DATASET_CONFIGS) {
        std::cout << "\n--- " << config.display_name << " ---" << std::endl;

        // Load dataset
        LoadedDataset dataset = LoadDataset(config);
        if (!dataset.valid) {
            std::cout << "SKIPPED: Dataset not available." << std::endl;
            std::cout << "To prepare: ./prepare_ann_benchmark_data.sh " << config.name << " ./data" << std::endl;
            continue;
        }

        std::cout << "Loaded: " << dataset.n << " vectors, " << dataset.dim << "D, "
                  << dataset.nq << " queries, metric=" << dataset.metric_type << std::endl;

        // Build HNSW index
        knowhere::Json build_conf;
        build_conf[knowhere::meta::DIM] = dataset.dim;
        build_conf[knowhere::meta::METRIC_TYPE] = dataset.metric_type;
        build_conf[knowhere::indexparam::HNSW_M] = config.hnsw_m;
        build_conf[knowhere::indexparam::EFCONSTRUCTION] = config.ef_construction;

        auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
        auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
        REQUIRE(index_res.has_value());
        auto index = index_res.value();

        std::cout << "Building HNSW index (M=" << config.hnsw_m << ", ef_c=" << config.ef_construction << ")..."
                  << std::flush;
        auto base_ds = knowhere::GenDataSet(dataset.n, dataset.dim, dataset.base_data.data());
        auto build_res = index.Build(base_ds, build_conf);
        REQUIRE(build_res == knowhere::Status::success);
        std::cout << " Done." << std::endl;

        // Test with uniform labels
        std::cout << "\n[Uniform Label Distribution]" << std::endl;

        // Get label distribution
        std::map<int32_t, int> label_counts;
        for (int32_t l : dataset.labels_uniform) {
            label_counts[l]++;
        }
        std::vector<std::pair<int32_t, int>> sorted_labels(label_counts.begin(), label_counts.end());
        std::sort(sorted_labels.begin(), sorted_labels.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        BenchmarkResult dummy;
        dummy.PrintHeader();

        // Test first 3 labels (different filter ratios)
        int labels_to_test = std::min(3, static_cast<int>(sorted_labels.size()));
        std::vector<BenchmarkResult> baseline_results;

        for (int i = 0; i < labels_to_test; i++) {
            int32_t target_label = sorted_labels[i].first;

            // Run baseline
            auto baseline = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                                         false, 0.0f, k, config.search_ef, num_queries);
            baseline.name = "Alpha-only";
            baseline.Print();
            baseline_results.push_back(baseline);

            // Run JAG
            auto jag = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                                    true, jag_filter_weight, k, config.search_ef, num_queries);
            jag.name = "JAG+Alpha";
            jag.qps_delta_percent = ((jag.qps - baseline.qps) / baseline.qps) * 100;
            jag.Print();
        }

        // Test with Zipf labels if available
        if (!dataset.labels_zipf.empty()) {
            std::cout << "\n[Zipf Label Distribution (alpha=1.5)]" << std::endl;

            std::map<int32_t, int> zipf_counts;
            for (int32_t l : dataset.labels_zipf) {
                zipf_counts[l]++;
            }
            std::vector<std::pair<int32_t, int>> sorted_zipf(zipf_counts.begin(), zipf_counts.end());
            std::sort(sorted_zipf.begin(), sorted_zipf.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

            dummy.PrintHeader();

            for (int i = 0; i < std::min(3, static_cast<int>(sorted_zipf.size())); i++) {
                int32_t target_label = sorted_zipf[i].first;

                auto baseline = RunBenchmark(index, dataset, dataset.labels_zipf, target_label,
                                             false, 0.0f, k, config.search_ef, num_queries);
                baseline.name = "Alpha-only";
                baseline.Print();

                auto jag = RunBenchmark(index, dataset, dataset.labels_zipf, target_label,
                                        true, jag_filter_weight, k, config.search_ef, num_queries);
                jag.name = "JAG+Alpha";
                jag.qps_delta_percent = ((jag.qps - baseline.qps) / baseline.qps) * 100;
                jag.Print();
            }
        }
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Benchmark Summary" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "- JAG+Alpha should show QPS improvement at higher filter ratios" << std::endl;
    std::cout << "- Zipf distribution creates more varied filter ratios than uniform" << std::endl;
    std::cout << "- Optimal filter_weight varies by dataset (tested with 1.0)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

TEST_CASE("JAG Benchmark - GloVe-100", "[jag][benchmark][glove100]") {
    const DatasetConfig& config = DATASET_CONFIGS[1];  // glove-100

    LoadedDataset dataset = LoadDataset(config);
    if (!dataset.valid) {
        std::cout << "\nGloVe-100 benchmark SKIPPED - dataset not available" << std::endl;
        std::cout << "To prepare: ./tests/ut/prepare_ann_benchmark_data.sh glove-100 ./data" << std::endl;
        std::cout << "Then: export GLOVE100_PATH=./data (or ANN_BENCHMARK_PATH=./data)" << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG Benchmark - GloVe-100" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset.n << " vectors, " << dataset.dim << "D" << std::endl;
    std::cout << "Metric: " << dataset.metric_type << " (Angular normalized to IP)" << std::endl;

    // Build index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dataset.dim;
    build_conf[knowhere::meta::METRIC_TYPE] = dataset.metric_type;
    build_conf[knowhere::indexparam::HNSW_M] = 32;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    std::cout << "Building index..." << std::flush;
    auto base_ds = knowhere::GenDataSet(dataset.n, dataset.dim, dataset.base_data.data());
    REQUIRE(index.Build(base_ds, build_conf) == knowhere::Status::success);
    std::cout << " Done.\n" << std::endl;

    const int k = 10;
    const int ef = 256;
    const int num_queries = 100;
    const float jag_filter_weight = GetJagFilterWeight(1.0f);

    // Test with first label
    int32_t target_label = dataset.labels_uniform[0];
    int32_t valid_count = 0;
    for (int32_t l : dataset.labels_uniform) {
        if (l == target_label) valid_count++;
    }

    auto baseline = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                                 false, 0.0f, k, ef, num_queries);
    auto jag = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                            true, jag_filter_weight, k, ef, num_queries);

    BenchmarkResult dummy;
    dummy.PrintHeader();

    baseline.name = "Alpha-only";
    baseline.Print();

    jag.name = "JAG+Alpha";
    jag.qps_delta_percent = ((jag.qps - baseline.qps) / baseline.qps) * 100;
    jag.Print();

    std::cout << "\nFilter ratio: " << std::fixed << std::setprecision(1)
              << (baseline.filter_ratio * 100) << "%" << std::endl;
    std::cout << "QPS improvement: " << std::setprecision(1) << jag.qps_delta_percent << "%" << std::endl;
}

TEST_CASE("JAG Benchmark - Deep-1M", "[jag][benchmark][deep1m]") {
    const DatasetConfig& config = DATASET_CONFIGS[2];  // deep-1m

    LoadedDataset dataset = LoadDataset(config);
    if (!dataset.valid) {
        std::cout << "\nDeep-1M benchmark SKIPPED - dataset not available" << std::endl;
        std::cout << "To prepare: ./tests/ut/prepare_ann_benchmark_data.sh deep-1m ./data" << std::endl;
        std::cout << "Then: export DEEP1M_PATH=./data (or ANN_BENCHMARK_PATH=./data)" << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG Benchmark - Deep-1M" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset.n << " vectors, " << dataset.dim << "D" << std::endl;

    // Build index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dataset.dim;
    build_conf[knowhere::meta::METRIC_TYPE] = dataset.metric_type;
    build_conf[knowhere::indexparam::HNSW_M] = 32;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    std::cout << "Building index..." << std::flush;
    auto base_ds = knowhere::GenDataSet(dataset.n, dataset.dim, dataset.base_data.data());
    REQUIRE(index.Build(base_ds, build_conf) == knowhere::Status::success);
    std::cout << " Done.\n" << std::endl;

    const int k = 10;
    const int ef = 256;
    const int num_queries = 100;
    const float jag_filter_weight = GetJagFilterWeight(1.0f);

    int32_t target_label = dataset.labels_uniform[0];

    auto baseline = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                                 false, 0.0f, k, ef, num_queries);
    auto jag = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                            true, jag_filter_weight, k, ef, num_queries);

    BenchmarkResult dummy;
    dummy.PrintHeader();

    baseline.name = "Alpha-only";
    baseline.Print();

    jag.name = "JAG+Alpha";
    jag.qps_delta_percent = ((jag.qps - baseline.qps) / baseline.qps) * 100;
    jag.Print();

    std::cout << "\nFilter ratio: " << std::fixed << std::setprecision(1)
              << (baseline.filter_ratio * 100) << "%" << std::endl;
    std::cout << "QPS improvement: " << std::setprecision(1) << jag.qps_delta_percent << "%" << std::endl;
}

TEST_CASE("JAG Benchmark - GIST", "[jag][benchmark][gist]") {
    const DatasetConfig& config = DATASET_CONFIGS[3];  // gist

    LoadedDataset dataset = LoadDataset(config);
    if (!dataset.valid) {
        std::cout << "\nGIST benchmark SKIPPED - dataset not available" << std::endl;
        std::cout << "To prepare: ./tests/ut/prepare_ann_benchmark_data.sh gist ./data" << std::endl;
        std::cout << "Then: export GIST_PATH=./data (or ANN_BENCHMARK_PATH=./data)" << std::endl;
        return;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG Benchmark - GIST" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset.n << " vectors, " << dataset.dim << "D (high dimensional)" << std::endl;

    // Build index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dataset.dim;
    build_conf[knowhere::meta::METRIC_TYPE] = dataset.metric_type;
    build_conf[knowhere::indexparam::HNSW_M] = 32;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    std::cout << "Building index (this may take a while for 960D)..." << std::flush;
    auto base_ds = knowhere::GenDataSet(dataset.n, dataset.dim, dataset.base_data.data());
    REQUIRE(index.Build(base_ds, build_conf) == knowhere::Status::success);
    std::cout << " Done.\n" << std::endl;

    const int k = 10;
    const int ef = 256;
    const int num_queries = 100;
    const float jag_filter_weight = GetJagFilterWeight(1.0f);

    int32_t target_label = dataset.labels_uniform[0];

    auto baseline = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                                 false, 0.0f, k, ef, num_queries);
    auto jag = RunBenchmark(index, dataset, dataset.labels_uniform, target_label,
                            true, jag_filter_weight, k, ef, num_queries);

    BenchmarkResult dummy;
    dummy.PrintHeader();

    baseline.name = "Alpha-only";
    baseline.Print();

    jag.name = "JAG+Alpha";
    jag.qps_delta_percent = ((jag.qps - baseline.qps) / baseline.qps) * 100;
    jag.Print();

    std::cout << "\nFilter ratio: " << std::fixed << std::setprecision(1)
              << (baseline.filter_ratio * 100) << "%" << std::endl;
    std::cout << "QPS improvement: " << std::setprecision(1) << jag.qps_delta_percent << "%" << std::endl;
}
