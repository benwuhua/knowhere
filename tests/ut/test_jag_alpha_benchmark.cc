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

// ============== Benchmark Result ==============

struct BenchmarkResult {
    std::string name;
    float filter_ratio;
    float recall;
    double qps;

    void
    PrintHeader() {
        std::cout << std::left
                  << std::setw(15) << "Name" << " | "
                  << std::setw(10) << "Filter%" << " | "
                  << std::setw(12) << "Recall" << " | "
                  << std::setw(12) << "QPS" << std::endl;
        std::cout << std::string(55, '-') << std::endl;
    }

    void
    Print() const {
        std::cout << std::left
                  << std::setw(15) << name << " | "
                  << std::setw(10) << std::setprecision(1) << (filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(4) << recall << " | "
                  << std::setw(12) << std::setprecision(1) << qps << std::endl;
    }
};

}  // namespace

// ============== Benchmark Test Cases ==============

TEST_CASE("JAG+Alpha vs Alpha-only Benchmark - Random Data", "[jag][benchmark][compare]") {
    const int64_t n = 10000;
    const int64_t dim = 128;
    const int64_t nq = 100;
    const int k = 10;
    const int hnsw_ef = 256;

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG+Alpha vs Alpha-only Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << "D" << std::endl;
    std::cout << "Queries: " << nq << ", K: " << k << ", EF: " << hnsw_ef << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Generate data
    auto base_ds = GenDataSet(n, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 123);
    const float* base_data = reinterpret_cast<const float*>(base_ds->GetTensor());
    const float* query_data = reinterpret_cast<const float*>(query_ds->GetTensor());

    // Generate labels (10% filter ratio)
    auto filter_set = knowhere::GenerateRandomLabels(n, 10, 456);
    int32_t target_label = filter_set.labels[0];
    float filter_ratio = filter_set.GetFilterRatio(target_label);

    // Build HNSW index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dim;
    build_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    build_conf[knowhere::indexparam::HNSW_M] = 32;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    auto build_res = index.Build(base_ds, build_conf);
    REQUIRE(build_res == knowhere::Status::success);

    // Create bitset
    std::vector<uint8_t> bitset_data((n + 7) / 8, 0xFF);
    for (int64_t i = 0; i < n; i++) {
        if (filter_set.GetLabel(i) == target_label) {
            bitset_data[i / 8] &= ~(1 << (i % 8));
        }
    }
    knowhere::BitsetView bitset(bitset_data.data(), n);

    // Helper lambda to run search and measure performance
    auto run_search = [&](bool enable_jag, float jag_filter_weight) -> BenchmarkResult {
        BenchmarkResult result;
        result.filter_ratio = filter_ratio;
        result.name = enable_jag ? "JAG+Alpha" : "Alpha-only";

        knowhere::Json search_conf;
        search_conf[knowhere::meta::DIM] = dim;
        search_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        search_conf[knowhere::meta::TOPK] = k;
        search_conf[knowhere::indexparam::EF] = hnsw_ef;
        search_conf[knowhere::indexparam::ENABLE_JAG] = enable_jag;
        search_conf[knowhere::indexparam::JAG_FILTER_WEIGHT] = jag_filter_weight;
        search_conf[knowhere::indexparam::JAG_CANDIDATE_POOL_SIZE] = hnsw_ef * 4;

        // Warmup
        for (int i = 0; i < 5; i++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data + i * dim));
            index.Search(query_ds_single, search_conf, bitset);
        }

        // Measure QPS and recall
        int total_hits = 0;
        int total_gt = 0;

        StopWatch sw;
        for (int64_t q = 0; q < nq; q++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data + q * dim));

            auto gt = ComputeFilteredGroundTruth(base_data, query_data + q * dim, n, dim, k, filter_set, target_label);
            std::set<int64_t> gt_set(gt.begin(), gt.end());

            auto search_res = index.Search(query_ds_single, search_conf, bitset);
            if (!search_res.has_value()) {
                continue;
            }

            auto res_ids = search_res.value()->GetIds();
            for (int i = 0; i < k; i++) {
                if (gt_set.count(res_ids[i])) {
                    total_hits++;
                }
            }
            total_gt += gt.size();
        }
        double elapsed_sec = sw.elapsed();

        result.qps = nq / elapsed_sec;
        result.recall = (total_gt > 0) ? static_cast<float>(total_hits) / total_gt : 0.0f;

        return result;
    };

    // Run baseline (Alpha-only)
    BenchmarkResult baseline = run_search(false, 0.0f);

    baseline.PrintHeader();
    baseline.Print();

    // Run JAG+Alpha with different filter_weight values
    std::vector<float> filter_weights = {0.5f, 1.0f, 2.0f, 5.0f};

    std::cout << "\n--- JAG+Alpha with different filter_weight values ---" << std::endl;
    baseline.PrintHeader();

    for (float fw : filter_weights) {
        BenchmarkResult jag = run_search(true, fw);
        jag.name = "JAG(w=" + std::to_string(fw) + ")";
        jag.Print();
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Analysis:" << std::endl;
    std::cout << "- Higher filter_weight prioritizes valid nodes more aggressively" << std::endl;
    std::cout << "- Trade-off: recall vs valid_visit_ratio" << std::endl;
    std::cout << "- JAG paper recommends filter_weight = 1.0-2.0 for balanced results" << std::endl;
    std::cout << "========================================" << std::endl;
}

TEST_CASE("JAG+Alpha Multi-Filter-Ratio Comparison", "[jag][benchmark][multi]") {
    const int64_t n = 5000;
    const int64_t dim = 64;
    const int64_t nq = 50;
    const int k = 10;
    const int hnsw_ef = 128;

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG+Alpha Multi-Filter-Ratio Comparison" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << "D" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Generate data
    auto base_ds = GenDataSet(n, dim, 42);
    auto query_ds = GenDataSet(nq, dim, 123);
    const float* base_data = reinterpret_cast<const float*>(base_ds->GetTensor());
    const float* query_data = reinterpret_cast<const float*>(query_ds->GetTensor());

    // Build HNSW index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dim;
    build_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    build_conf[knowhere::indexparam::HNSW_M] = 16;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 100;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();
    REQUIRE(index.Build(base_ds, build_conf) == knowhere::Status::success);

    // Test different filter ratios
    std::vector<int> label_counts = {2, 5, 10, 20};  // ~50%, ~20%, ~10%, ~5%

    std::cout << std::left
              << std::setw(10) << "Filter%" << " | "
              << std::setw(12) << "Base_Rcl" << " | "
              << std::setw(12) << "JAG_Rcl" << " | "
              << std::setw(12) << "Base_QPS" << " | "
              << std::setw(12) << "JAG_QPS" << " | "
              << std::setw(10) << "QPS_Δ%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (int num_labels : label_counts) {
        auto filter_set = knowhere::GenerateRandomLabels(n, num_labels, 456 + num_labels);
        int32_t target_label = filter_set.labels[0];
        float filter_ratio = filter_set.GetFilterRatio(target_label);

        // Create bitset
        std::vector<uint8_t> bitset_data((n + 7) / 8, 0xFF);
        for (int64_t i = 0; i < n; i++) {
            if (filter_set.GetLabel(i) == target_label) {
                bitset_data[i / 8] &= ~(1 << (i % 8));
            }
        }
        knowhere::BitsetView bitset(bitset_data.data(), n);

        // Baseline search config
        knowhere::Json search_conf_base;
        search_conf_base[knowhere::meta::DIM] = dim;
        search_conf_base[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        search_conf_base[knowhere::meta::TOPK] = k;
        search_conf_base[knowhere::indexparam::EF] = hnsw_ef;
        search_conf_base[knowhere::indexparam::ENABLE_JAG] = false;

        // JAG search config
        knowhere::Json search_conf_jag;
        search_conf_jag[knowhere::meta::DIM] = dim;
        search_conf_jag[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        search_conf_jag[knowhere::meta::TOPK] = k;
        search_conf_jag[knowhere::indexparam::EF] = hnsw_ef;
        search_conf_jag[knowhere::indexparam::ENABLE_JAG] = true;
        search_conf_jag[knowhere::indexparam::JAG_FILTER_WEIGHT] = 2.0f;
        search_conf_jag[knowhere::indexparam::JAG_CANDIDATE_POOL_SIZE] = hnsw_ef * 4;

        // Run baseline
        int total_hits_base = 0, total_gt_base = 0;
        StopWatch sw_base;
        for (int64_t q = 0; q < nq; q++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data + q * dim));
            auto gt = ComputeFilteredGroundTruth(base_data, query_data + q * dim, n, dim, k, filter_set, target_label);
            std::set<int64_t> gt_set(gt.begin(), gt.end());

            auto search_res = index.Search(query_ds_single, search_conf_base, bitset);
            if (search_res.has_value()) {
                auto res_ids = search_res.value()->GetIds();
                for (int i = 0; i < k; i++) {
                    if (gt_set.count(res_ids[i])) total_hits_base++;
                }
            }
            total_gt_base += gt.size();
        }
        double elapsed_base = sw_base.elapsed();
        float recall_base = (total_gt_base > 0) ? (float)total_hits_base / total_gt_base : 0.0f;
        double qps_base = nq / elapsed_base;

        // Run JAG
        int total_hits_jag = 0, total_gt_jag = 0;
        StopWatch sw_jag;
        for (int64_t q = 0; q < nq; q++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data + q * dim));
            auto gt = ComputeFilteredGroundTruth(base_data, query_data + q * dim, n, dim, k, filter_set, target_label);
            std::set<int64_t> gt_set(gt.begin(), gt.end());

            auto search_res = index.Search(query_ds_single, search_conf_jag, bitset);
            if (search_res.has_value()) {
                auto res_ids = search_res.value()->GetIds();
                for (int i = 0; i < k; i++) {
                    if (gt_set.count(res_ids[i])) total_hits_jag++;
                }
            }
            total_gt_jag += gt.size();
        }
        double elapsed_jag = sw_jag.elapsed();
        float recall_jag = (total_gt_jag > 0) ? (float)total_hits_jag / total_gt_jag : 0.0f;
        double qps_jag = nq / elapsed_jag;

        double qps_delta = ((qps_jag - qps_base) / qps_base) * 100;

        std::cout << std::fixed
                  << std::setw(10) << std::setprecision(1) << (filter_ratio * 100) << " | "
                  << std::setw(12) << std::setprecision(4) << recall_base << " | "
                  << std::setw(12) << recall_jag << " | "
                  << std::setw(12) << std::setprecision(1) << qps_base << " | "
                  << std::setw(12) << qps_jag << " | "
                  << std::setw(10) << std::setprecision(1) << qps_delta << std::endl;
    }

    std::cout << "========================================" << std::endl;
}

TEST_CASE("JAG+Alpha vs Alpha-only Benchmark - SIFT1M", "[jag][benchmark][sift1m]") {
    // Get data path from environment or use default
    const char* env_path = std::getenv("SIFT1M_PATH");
    std::string data_path = env_path ? env_path : "./data";

    std::string base_path = data_path + "/sift1m-base.fbin";
    std::string query_path = data_path + "/sift1m-query.fbin";
    std::string label_path = data_path + "/sift1m-base-filters-label.ibin";

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

    const int k = 10;
    const int hnsw_ef = 256;
    const int num_queries = 100;

    std::cout << "\n========================================" << std::endl;
    std::cout << "JAG+Alpha vs Alpha-only Benchmark - SIFT1M" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << n << " vectors, " << dim << "D" << std::endl;
    std::cout << "Queries: " << num_queries << " (of " << nq << "), K: " << k << std::endl;
    std::cout << "HNSW: M=32, ef=" << hnsw_ef << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Build HNSW index
    knowhere::Json build_conf;
    build_conf[knowhere::meta::DIM] = dim;
    build_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
    build_conf[knowhere::indexparam::HNSW_M] = 32;
    build_conf[knowhere::indexparam::EFCONSTRUCTION] = 200;

    auto version = knowhere::Version::GetCurrentVersion().VersionNumber();
    auto index_res = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
        knowhere::IndexEnum::INDEX_HNSW, version, nullptr);
    REQUIRE(index_res.has_value());
    auto index = index_res.value();

    std::cout << "Building HNSW index..." << std::endl;
    auto base_ds = knowhere::GenDataSet(n, dim, base_data.data());
    auto build_res = index.Build(base_ds, build_conf);
    REQUIRE(build_res == knowhere::Status::success);
    std::cout << "Index built.\n" << std::endl;

    // Test different filter ratios by selecting labels with different frequencies
    std::vector<std::pair<int32_t, size_t>> label_counts;
    for (const auto& [label, ids] : filter_set.label_to_ids) {
        label_counts.push_back({label, ids.size()});
    }
    std::sort(label_counts.begin(), label_counts.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    BenchmarkResult dummy;
    dummy.PrintHeader();

    // Test top 3 labels
    int labels_to_test = std::min(3, static_cast<int>(label_counts.size()));
    for (int i = 0; i < labels_to_test; i++) {
        int32_t target_label = label_counts[i].first;
        float filter_ratio = filter_set.GetFilterRatio(target_label);

        // Create bitset
        std::vector<uint8_t> bitset_data((n + 7) / 8, 0xFF);
        for (int32_t j = 0; j < n; j++) {
            if (filter_set.GetLabel(j) == target_label) {
                bitset_data[j / 8] &= ~(1 << (j % 8));
            }
        }
        knowhere::BitsetView bitset(bitset_data.data(), n);

        std::cout << "\n--- Filter Ratio: " << std::fixed << std::setprecision(1)
                  << (filter_ratio * 100) << "% ---" << std::endl;
        dummy.PrintHeader();

        // Search configs
        knowhere::Json search_conf_base;
        search_conf_base[knowhere::meta::DIM] = dim;
        search_conf_base[knowhere::meta::METRIC_TYPE] = knowhere::metric::L2;
        search_conf_base[knowhere::meta::TOPK] = k;
        search_conf_base[knowhere::indexparam::EF] = hnsw_ef;
        search_conf_base[knowhere::indexparam::ENABLE_JAG] = false;

        knowhere::Json search_conf_jag = search_conf_base;
        search_conf_jag[knowhere::indexparam::ENABLE_JAG] = true;
        search_conf_jag[knowhere::indexparam::JAG_FILTER_WEIGHT] = 2.0f;
        search_conf_jag[knowhere::indexparam::JAG_CANDIDATE_POOL_SIZE] = hnsw_ef * 4;

        // Run baseline
        int total_hits_base = 0, total_gt_base = 0;
        StopWatch sw_base;
        for (int q = 0; q < num_queries; q++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data.data() + q * dim));
            auto gt = ComputeFilteredGroundTruth(base_data.data(), query_data.data() + q * dim, n, dim, k, filter_set, target_label);
            std::set<int64_t> gt_set(gt.begin(), gt.end());

            auto search_res = index.Search(query_ds_single, search_conf_base, bitset);
            if (search_res.has_value()) {
                auto res_ids = search_res.value()->GetIds();
                for (int j = 0; j < k; j++) {
                    if (gt_set.count(res_ids[j])) total_hits_base++;
                }
            }
            total_gt_base += gt.size();
        }
        double elapsed_base = sw_base.elapsed();
        BenchmarkResult baseline;
        baseline.name = "Alpha-only";
        baseline.filter_ratio = filter_ratio;
        baseline.recall = (total_gt_base > 0) ? (float)total_hits_base / total_gt_base : 0.0f;
        baseline.qps = num_queries / elapsed_base;
        baseline.Print();

        // Run JAG
        int total_hits_jag = 0, total_gt_jag = 0;
        StopWatch sw_jag;
        for (int q = 0; q < num_queries; q++) {
            auto query_ds_single = knowhere::GenDataSet(1, dim, const_cast<float*>(query_data.data() + q * dim));
            auto gt = ComputeFilteredGroundTruth(base_data.data(), query_data.data() + q * dim, n, dim, k, filter_set, target_label);
            std::set<int64_t> gt_set(gt.begin(), gt.end());

            auto search_res = index.Search(query_ds_single, search_conf_jag, bitset);
            if (search_res.has_value()) {
                auto res_ids = search_res.value()->GetIds();
                for (int j = 0; j < k; j++) {
                    if (gt_set.count(res_ids[j])) total_hits_jag++;
                }
            }
            total_gt_jag += gt.size();
        }
        double elapsed_jag = sw_jag.elapsed();
        BenchmarkResult jag;
        jag.name = "JAG+Alpha";
        jag.filter_ratio = filter_ratio;
        jag.recall = (total_gt_jag > 0) ? (float)total_hits_jag / total_gt_jag : 0.0f;
        jag.qps = num_queries / elapsed_jag;
        jag.Print();

        // Calculate improvement
        double qps_improvement = ((jag.qps - baseline.qps) / baseline.qps) * 100;
        std::cout << "\nQPS Improvement: " << std::fixed << std::setprecision(1)
                  << qps_improvement << "%" << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Summary:" << std::endl;
    std::cout << "- JAG+Alpha maintains similar recall while potentially improving QPS" << std::endl;
    std::cout << "- Higher filter ratios benefit more from JAG optimization" << std::endl;
    std::cout << "- Optimal filter_weight depends on dataset characteristics" << std::endl;
    std::cout << "========================================" << std::endl;
}
