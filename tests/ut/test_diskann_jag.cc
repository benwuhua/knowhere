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

std::string kDir = fs::current_path().string() + "/diskann_jag_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kIndexDir = kDir + "/index";
std::string kIndexPrefix = kIndexDir + "/diskann";

constexpr uint32_t kNumRows = 5000;
constexpr uint32_t kNumQueries = 100;
constexpr uint32_t kDim = 64;
constexpr uint32_t kK = 10;
constexpr float kKnnRecall = 0.8f;

// Generate bitset with specified filter ratio (percentage of bits set = filtered out)
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

// Write raw data to disk for DiskANN
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

// Calculate recall (local version to avoid ambiguity)
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

TEST_CASE("DiskANN JAG Filtered Search Benchmark", "[diskann][jag]") {
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

    std::cout << "\n========================================\n";
    std::cout << "DiskANN JAG Filtered Search Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "Dataset: " << kNumRows << " vectors, " << kDim << "D\n";
    std::cout << "Queries: " << kNumQueries << ", K: " << kK << "\n";
    std::cout << "========================================\n\n";

    // Test different filter ratios
    const std::vector<float> filter_ratios = {0.1f, 0.3f, 0.5f, 0.7f};
    const std::vector<float> jag_weights = {0.3f, 0.5f, 1.0f, 2.0f};

    std::cout << "Filter% | Baseline_Rcl | JAG_Rcl | Base_QPS | JAG_QPS | QPS_Δ%\n";
    std::cout << "--------|--------------|---------|----------|---------|-------\n";

    for (float filter_ratio : filter_ratios) {
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

        // JAG search (with best weight)
        float best_jag_recall = 0;
        float best_jag_qps = 0;
        for (float weight : jag_weights) {
            auto jag_search_json = search_gen(true, weight);
            auto jag_start = std::chrono::high_resolution_clock::now();
            auto jag_result = diskann.Search(query_ds, jag_search_json, bitset);
            auto jag_end = std::chrono::high_resolution_clock::now();
            auto jag_ms = std::chrono::duration_cast<std::chrono::milliseconds>(jag_end - jag_start).count();
            float jag_recall = CalcRecall(*gt.value(), *jag_result.value());
            float jag_qps = (jag_ms > 0) ? (kNumQueries * 1000.0f / jag_ms) : 0;

            // Keep best result that maintains recall
            if (jag_recall >= base_recall * 0.98f && jag_qps > best_jag_qps) {
                best_jag_recall = jag_recall;
                best_jag_qps = jag_qps;
            }
        }

        float qps_delta = (base_qps > 0) ? ((best_jag_qps - base_qps) / base_qps * 100) : 0;
        std::cout << std::fixed << std::setprecision(1)
                  << (filter_ratio * 100) << "%   | "
                  << base_recall << "       | "
                  << best_jag_recall << "   | "
                  << base_qps << "   | "
                  << best_jag_qps << "  | "
                  << std::showpos << qps_delta << std::noshowpos << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Analysis:\n";
    std::cout << "- JAG should show QPS improvement at higher filter ratios\n";
    std::cout << "- Optimal jag_filter_weight varies (tested 0.3-2.0)\n";
    std::cout << "- Recall should remain >= baseline\n";
    std::cout << "========================================\n";

    // Cleanup
    fs::remove_all(kDir);
    fs::remove(kDir);
}

TEST_CASE("DiskANN JAG Parameter Validation", "[diskann][jag]") {
    fs::remove_all(kDir);
    fs::remove(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));
    REQUIRE_NOTHROW(fs::create_directory(kIndexDir));

    auto version = GenTestVersionList();
    auto base_ds = GenDataSet(kNumRows / 5, kDim, 30);
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows / 5, kDim);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = knowhere::metric::L2;
        json["k"] = kK;
        json["index_prefix"] = kIndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 24;
        json["search_list_size"] = 32;
        json["pq_code_budget_gb"] = sizeof(float) * kDim * (kNumRows / 5) * 0.125 / (1024 * 1024 * 1024);
        json["build_dram_budget_gb"] = 32.0;
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

    // Test that JAG parameters are accepted
    knowhere::Json jag_json = build_gen();
    jag_json["enable_jag"] = true;
    jag_json["jag_filter_weight"] = 1.5f;

    auto query_ds = GenDataSet(10, kDim, 42);
    auto bitset_data = GenerateBitsetWithFilterRatio(kNumRows / 5, 0.5f);
    knowhere::BitsetView bitset(bitset_data.data(), kNumRows / 5);

    auto result = diskann.Search(query_ds, jag_json, bitset);
    REQUIRE(result.has_value());

    std::cout << "\n=== DiskANN JAG Parameter Validation ===\n";
    std::cout << "enable_jag: " << jag_json["enable_jag"] << "\n";
    std::cout << "jag_filter_weight: " << jag_json["jag_filter_weight"] << "\n";
    std::cout << "Search completed successfully!\n";
    std::cout << "======================================\n";

    // Cleanup
    fs::remove_all(kDir);
    fs::remove(kDir);
}
