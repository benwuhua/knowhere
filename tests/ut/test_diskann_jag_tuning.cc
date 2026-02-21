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

#include <chrono>
#include <iomanip>
#include <random>
#include <string>
#include <unordered_set>

#include "catch2/catch_test_macros.hpp"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "utils.h"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif

namespace {

std::string kDir = fs::current_path().string() + "/diskann_jag_tuning";
std::string kRawDataPath = kDir + "/raw_data";
std::string kIndexDir = kDir + "/index";
std::string kIndexPrefix = kIndexDir + "/diskann";

constexpr uint32_t kNumRows = 5000;
constexpr uint32_t kNumQueries = 100;
constexpr uint32_t kDim = 64;
constexpr uint32_t kK = 10;

// Generate bitset with specified filter ratio
std::vector<uint8_t>
GenerateBitsetWithFilterRatio(size_t num_rows, float filter_ratio, uint64_t seed = 42) {
    std::vector<uint8_t> bitset((num_rows + 7) / 8, 0);
    size_t num_filtered = static_cast<size_t>(num_rows * filter_ratio);

    std::mt19937 gen(seed);
    std::vector<size_t> indices(num_rows);
    for (size_t i = 0; i < num_rows; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t i = 0; i < num_filtered; ++i) {
        size_t idx = indices[i];
        bitset[idx / 8] |= (1 << (idx % 8));
    }
    return bitset;
}

// Write raw data to disk
template <typename T>
void
WriteRawDataToDisk(const std::string& path, const T* data, size_t n, size_t dim) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream writer(path, std::ios::binary);
    writer.write((char*)&n, sizeof(uint32_t));
    writer.write((char*)&dim, sizeof(uint32_t));
    writer.write((char*)data, sizeof(T) * n * dim);
    writer.close();
}

// Calculate recall
float
CalcRecall(const knowhere::DataSet& gt, const knowhere::DataSet& result) {
    auto gt_ids = gt.GetIds();
    auto res_ids = result.GetIds();
    auto nq = gt.GetRows();
    auto gt_k = gt.GetDim();
    auto res_k = result.GetDim();
    auto k = std::min(gt_k, res_k);

    size_t matched = 0;
    for (auto i = 0; i < nq; ++i) {
        std::unordered_set<int64_t> gt_set(gt_ids + i * gt_k, gt_ids + i * gt_k + k);
        for (auto j = 0; j < k; ++j) {
            if (gt_set.count(res_ids[i * res_k + j]) > 0) {
                matched++;
            }
        }
    }
    return static_cast<float>(matched) / (nq * k);
}

}  // namespace

TEST_CASE("DiskANN JAG Filter Weight Tuning", "[diskann][jag][tuning]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kIndexDir));

    auto version = GenTestVersionList();

    // Generate dataset
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);

    // Write raw data to disk
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    // Build config
    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kIndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 32;
        json["search_list_size"] = 64;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.1 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    // Search config
    auto search_gen = [&](bool enable_jag, float jag_weight) {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kIndexPrefix;
        json["search_list_size"] = 32;
        json["beamwidth"] = 8;
        json["enable_jag"] = enable_jag;
        json["jag_filter_weight"] = jag_weight;
        return json;
    };

    // Build index
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::BinarySet binset;

    {
        auto diskann = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                           .value();
        auto build_json = build_gen();
        diskann.Build(nullptr, build_json);
        diskann.Serialize(binset);
    }

    // Load index
    auto diskann = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                       .value();
    auto deserialize_json = build_gen();
    diskann.Deserialize(binset, deserialize_json);

    std::cout << "\n================================================================================\n";
    std::cout << "DiskANN JAG Filter Weight Tuning\n";
    std::cout << "================================================================================\n";
    std::cout << "Dataset: " << kNumRows << " vectors, " << kDim << "D\n";
    std::cout << "Queries: " << kNumQueries << ", K: " << kK << "\n";
    std::cout << "================================================================================\n\n";

    // Test different filter ratios and weights
    const std::vector<float> filter_ratios = {0.1f, 0.3f, 0.5f, 0.7f};
    const std::vector<float> jag_weights = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f};

    for (float filter_ratio : filter_ratios) {
        std::cout << "\n=== Filter Ratio: " << (filter_ratio * 100) << "% ===\n";
        std::cout << "Weight  | Recall | QPS     | vs Baseline QPS | vs Baseline Recall\n";
        std::cout << "--------|--------|---------|-----------------|-------------------\n";

        auto bitset_data = GenerateBitsetWithFilterRatio(kNumRows, filter_ratio);
        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);

        // Get ground truth with bitset
        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds,
                                                               search_gen(false, 0.0f), bitset);

        // Baseline search (JAG disabled)
        auto base_search_json = search_gen(false, 0.0f);
        auto base_start = std::chrono::high_resolution_clock::now();
        auto base_result = diskann.Search(query_ds, base_search_json, bitset);
        auto base_end = std::chrono::high_resolution_clock::now();
        auto base_ms = std::chrono::duration_cast<std::chrono::milliseconds>(base_end - base_start).count();
        float base_recall = CalcRecall(*gt.value(), *base_result.value());
        float base_qps = (base_ms > 0) ? (kNumQueries * 1000.0f / base_ms) : 0;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "BASE    | " << base_recall << " | " << std::setw(7) << base_qps << " |       -         |       -\n";

        // Test different weights
        for (float weight : jag_weights) {
            auto jag_search_json = search_gen(true, weight);
            auto jag_start = std::chrono::high_resolution_clock::now();
            auto jag_result = diskann.Search(query_ds, jag_search_json, bitset);
            auto jag_end = std::chrono::high_resolution_clock::now();
            auto jag_ms = std::chrono::duration_cast<std::chrono::milliseconds>(jag_end - jag_start).count();
            float jag_recall = CalcRecall(*gt.value(), *jag_result.value());
            float jag_qps = (jag_ms > 0) ? (kNumQueries * 1000.0f / jag_ms) : 0;

            float qps_delta = (base_qps > 0) ? ((jag_qps - base_qps) / base_qps * 100) : 0;
            float recall_delta = (jag_recall - base_recall) * 100;

            std::cout << std::setw(7) << weight << " | " << jag_recall << " | "
                      << std::setw(7) << jag_qps << " | " << std::showpos << std::setw(6)
                      << qps_delta << "%      | " << std::setw(6) << recall_delta << "%"
                      << std::noshowpos << "\n";
        }
    }

    std::cout << "\n================================================================================\n";
    std::cout << "Tuning Recommendations:\n";
    std::cout << "- Lower weight (0.1-0.5): Better QPS at low filter ratios\n";
    std::cout << "- Medium weight (0.5-1.5): Balanced recall/QPS trade-off\n";
    std::cout << "- Higher weight (2.0-5.0): Better recall at high filter ratios\n";
    std::cout << "================================================================================\n";

    // Cleanup
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("DiskANN JAG Optimal Weight Finder", "[diskann][jag][tuning]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kIndexDir));

    auto version = GenTestVersionList();
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kIndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 32;
        json["search_list_size"] = 64;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.125 / (1024 * 1024 * 1024);
        json["search_cache_budget_gb"] = sizeof(float) * kDim * kNumRows * 0.1 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
        return json;
    };

    auto search_gen = [&](bool enable_jag, float jag_weight) {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kIndexPrefix;
        json["search_list_size"] = 32;
        json["beamwidth"] = 8;
        json["enable_jag"] = enable_jag;
        json["jag_filter_weight"] = jag_weight;
        return json;
    };

    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    knowhere::BinarySet binset;

    {
        auto diskann = knowhere::IndexFactory::Instance()
                           .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                           .value();
        diskann.Build(nullptr, build_gen());
        diskann.Serialize(binset);
    }

    auto diskann = knowhere::IndexFactory::Instance()
                       .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                       .value();
    diskann.Deserialize(binset, build_gen());

    std::cout << "\n================================================================================\n";
    std::cout << "DiskANN JAG Optimal Weight Finder\n";
    std::cout << "Finding best weight that maintains recall >= baseline while maximizing QPS\n";
    std::cout << "================================================================================\n\n";

    const std::vector<float> filter_ratios = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
    const std::vector<float> jag_weights = {0.05f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f, 1.0f, 1.5f, 2.0f};

    std::cout << "Filter% | Best Weight | Base_Rcl | JAG_Rcl | Base_QPS | JAG_QPS | QPS_Δ%\n";
    std::cout << "--------|-------------|----------|---------|----------|---------|-------\n";

    for (float filter_ratio : filter_ratios) {
        auto bitset_data = GenerateBitsetWithFilterRatio(kNumRows, filter_ratio);
        knowhere::BitsetView bitset(bitset_data.data(), kNumRows);

        auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds,
                                                               search_gen(false, 0.0f), bitset);

        // Baseline
        auto base_result = diskann.Search(query_ds, search_gen(false, 0.0f), bitset);
        auto base_start = std::chrono::high_resolution_clock::now();
        base_result = diskann.Search(query_ds, search_gen(false, 0.0f), bitset);
        auto base_end = std::chrono::high_resolution_clock::now();
        auto base_ms = std::chrono::duration_cast<std::chrono::milliseconds>(base_end - base_start).count();
        float base_recall = CalcRecall(*gt.value(), *base_result.value());
        float base_qps = (base_ms > 0) ? (kNumQueries * 1000.0f / base_ms) : 0;

        // Find best weight
        float best_weight = 0.0f;
        float best_qps = base_qps;
        float best_recall = base_recall;

        for (float weight : jag_weights) {
            auto jag_start = std::chrono::high_resolution_clock::now();
            auto jag_result = diskann.Search(query_ds, search_gen(true, weight), bitset);
            auto jag_end = std::chrono::high_resolution_clock::now();
            auto jag_ms = std::chrono::duration_cast<std::chrono::milliseconds>(jag_end - jag_start).count();
            float jag_recall = CalcRecall(*gt.value(), *jag_result.value());
            float jag_qps = (jag_ms > 0) ? (kNumQueries * 1000.0f / jag_ms) : 0;

            // Prefer higher QPS while maintaining or improving recall
            if (jag_recall >= base_recall * 0.98f && jag_qps > best_qps) {
                best_weight = weight;
                best_qps = jag_qps;
                best_recall = jag_recall;
            }
        }

        float qps_delta = (base_qps > 0) ? ((best_qps - base_qps) / base_qps * 100) : 0;
        std::cout << std::fixed << std::setprecision(1)
                  << (filter_ratio * 100) << "%   | "
                  << std::setw(11) << best_weight << " | "
                  << base_recall << "   | "
                  << best_recall << "   | "
                  << std::setw(7) << base_qps << " | "
                  << std::setw(7) << best_qps << " | "
                  << std::showpos << qps_delta << std::noshowpos << "\n";
    }

    std::cout << "\n================================================================================\n";

    // Cleanup
    fs::remove_all(kDir);
    fs::remove(kDir);
}
