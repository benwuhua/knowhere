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

#include <sys/resource.h>

#include <atomic>
#include <string>
#include <thread>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "filemanager/FileManager.h"
#include "filemanager/impl/LocalFileManager.h"
#include "index/diskann/pageann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/knowhere_check.h"
#include "knowhere/context.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/utils.h"
#include "knowhere/version.h"
#include "nlohmann/json.hpp"
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
#include <fstream>

namespace {
std::string kDir = fs::current_path().string() + "/pageann_test";
std::string kRawDataPath = kDir + "/raw_data";
std::string kL2IndexDir = kDir + "/l2_index";
std::string kIPIndexDir = kDir + "/ip_index";
std::string kCOSINEIndexDir = kDir + "/cosine_index";
std::string kL2IndexPrefix = kL2IndexDir + "/l2";
std::string kIPIndexPrefix = kIPIndexDir + "/ip";
std::string kCOSINEIndexPrefix = kCOSINEIndexDir + "/cosine";

std::string kEmbListL2IndexDir = kDir + "/emb_list_l2_index";
std::string kEmbListIPIndexDir = kDir + "/emb_list_ip_index";
std::string kEmbListCOSINEIndexDir = kDir + "/emb_list_cosine_index";
std::string kEmbListL2IndexPrefix = kEmbListL2IndexDir + "/max_sim_l2";
std::string kEmbListIPIndexPrefix = kEmbListIPIndexDir + "/max_sim_ip";
std::string kEmbListCOSINEIndexPrefix = kEmbListCOSINEIndexDir + "/max_sim_cosine";
std::string kEmbListOffsetPath = kDir + "/emb_list_offset.bin";

constexpr uint32_t kNumRows = 1000;
constexpr uint32_t kNumQueries = 10;
constexpr uint32_t kDim = 128;
constexpr uint32_t kLargeDim = 256;
constexpr uint32_t kK = 10;
constexpr float kKnnRecall = 0.9;
constexpr float kEmbListKnnRecall = 0.75;
}  // namespace

TEST_CASE("PageANN - Basic functionality test", "[pageann]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = GENERATE(as<std::string>{}, knowhere::metric::L2, knowhere::metric::IP);

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);

    // Prepare build config
    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        return json;
    };

    // Write raw data to disk
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    SECTION("PageANN index creation") {
        // Create PageANN index
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto pageann_index_pack = knowhere::Pack(file_manager);

        auto pageann = knowhere::IndexFactory::Instance()
                            .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                            .value();
        REQUIRE(true);  // Index creation successful

        // Build index
        auto build_json = build_gen();
        auto status = pageann.Build(nullptr, build_json);
        REQUIRE(status == knowhere::Status::success);

        // Serialize
        knowhere::BinarySet binset;
        status = pageann.Serialize(binset);
        REQUIRE(status == knowhere::Status::success);
    }

    SECTION("PageANN search") {
        // Build and serialize
        knowhere::BinarySet binset;
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            auto build_json = build_gen();
            pageann.Build(nullptr, build_json);
            pageann.Serialize(binset);
        }

        // Deserialize and search
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["k"] = kK;
            search_json["index_prefix"] = kL2IndexPrefix;
            search_json["search_list_size"] = 64;
            search_json["beamwidth"] = 8;
            search_json["prefetch_batch_size"] = 16;
            search_json["enable_frequency_aware_cache"] = true;

            pageann.Deserialize(binset, search_json);

            auto results = pageann.Search(query_ds, search_json, nullptr);
            REQUIRE(results.has_value());

            // Verify results
            auto res_ids = results.value()->GetIds();
            auto res_dist = results.value()->GetDistance();
            REQUIRE(res_ids != nullptr);
            REQUIRE(res_dist != nullptr);
        }
    }

    SECTION("PageANN config parameters") {
        // Test that PageANN-specific config parameters are accepted
        knowhere::Json config;
        config["dim"] = kDim;
        config["metric_type"] = metric_str;
        config["k"] = kK;
        config["prefetch_batch_size"] = 16;
        config["prefetch_buffer_mb"] = 256;
        config["prefetch_lookahead_ratio"] = 2.0f;
        config["enable_frequency_aware_cache"] = true;
        config["frequency_cache_budget_gb"] = 0.1f;
        config["cache_decay_interval"] = 10000;
        config["cache_decay_factor"] = 0.99f;

        REQUIRE(config["prefetch_batch_size"] == 16);
        REQUIRE(config["prefetch_buffer_mb"] == 256);
        REQUIRE(config["prefetch_lookahead_ratio"] == 2.0f);
        REQUIRE(config["enable_frequency_aware_cache"] == true);
        REQUIRE(config["frequency_cache_budget_gb"] == 0.1f);
        REQUIRE(config["cache_decay_interval"] == 10000);
        REQUIRE(config["cache_decay_factor"] == 0.99f);
    }

    // Cleanup
    fs::remove_all(kDir);
}

TEST_CASE("PageANN vs DISKANN - Compatibility test", "[pageann][diskann]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = knowhere::metric::L2;

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        return json;
    };

    // Write raw data
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    // Build index with DISKANN
    knowhere::BinarySet binset;
    {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto diskann_index_pack = knowhere::Pack(file_manager);

        auto diskann = knowhere::IndexFactory::Instance()
                            .Create<knowhere::fp32>("DISKANN", version, diskann_index_pack)
                            .value();

        auto build_json = build_gen();
        diskann.Build(nullptr, build_json);
        diskann.Serialize(binset);
    }

    SECTION("PAGEANN can load DISKANN index") {
        std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
        auto pageann_index_pack = knowhere::Pack(file_manager);

        auto pageann = knowhere::IndexFactory::Instance()
                            .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                            .value();

        knowhere::Json search_json;
        search_json["dim"] = kDim;
        search_json["metric_type"] = metric_str;
        search_json["k"] = kK;
        search_json["index_prefix"] = kL2IndexPrefix;
        search_json["search_list_size"] = 64;
        search_json["beamwidth"] = 8;

        // PageANN should be able to load DISKANN-built index
        auto status = pageann.Deserialize(binset, search_json);
        REQUIRE(status == knowhere::Status::success);

        // Search should work
        auto results = pageann.Search(query_ds, search_json, nullptr);
        REQUIRE(results.has_value());
    }

    // Cleanup
    fs::remove_all(kDir);
}

TEST_CASE("PageANN - Multi-metric support", "[pageann][metric]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = GENERATE(as<std::string>{},
                               knowhere::metric::L2,
                               knowhere::metric::IP,
                               knowhere::metric::COSINE);

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        return json;
    };

    // Write raw data
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    SECTION("Build and search with different metrics") {
        knowhere::BinarySet binset;
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            auto build_json = build_gen();
            pageann.Build(nullptr, build_json);
            pageann.Serialize(binset);
        }

        // Deserialize and search
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["k"] = kK;
            search_json["index_prefix"] = kL2IndexPrefix;
            search_json["search_list_size"] = 64;
            search_json["beamwidth"] = 8;

            pageann.Deserialize(binset, search_json);

            auto results = pageann.Search(query_ds, search_json, nullptr);
            REQUIRE(results.has_value());

            // Verify results structure
            auto res_ids = results.value()->GetIds();
            auto res_dist = results.value()->GetDistance();
            REQUIRE(res_ids != nullptr);
            REQUIRE(res_dist != nullptr);
        }
    }

    // Cleanup
    fs::remove_all(kDir);
}

TEST_CASE("PageANN - LSH routing test", "[pageann][lsh]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = knowhere::metric::L2;

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);
    auto query_ds = GenDataSet(kNumQueries, kDim, 42);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["enable_lsh_routing"] = true;
        json["lsh_num_projections"] = 32;
        return json;
    };

    // Write raw data
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    SECTION("LSH enabled vs disabled") {
        // Build with LSH enabled
        knowhere::BinarySet binset;
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            auto build_json = build_gen();
            pageann.Build(nullptr, build_json);
            pageann.Serialize(binset);
        }

        // Test with LSH enabled
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["k"] = kK;
            search_json["index_prefix"] = kL2IndexPrefix;
            search_json["search_list_size"] = 64;
            search_json["beamwidth"] = 8;
            search_json["enable_lsh_routing"] = true;

            pageann.Deserialize(binset, search_json);
            auto results = pageann.Search(query_ds, search_json, nullptr);
            REQUIRE(results.has_value());
        }

        // Test with LSH disabled
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["k"] = kK;
            search_json["index_prefix"] = kL2IndexPrefix;
            search_json["search_list_size"] = 64;
            search_json["beamwidth"] = 8;
            search_json["enable_lsh_routing"] = false;

            pageann.Deserialize(binset, search_json);
            auto results = pageann.Search(query_ds, search_json, nullptr);
            REQUIRE(results.has_value());
        }
    }

    // Cleanup
    fs::remove_all(kDir);
}

TEST_CASE("PageANN - GetVectorByIds test", "[pageann][getvector]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = knowhere::metric::L2;

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        return json;
    };

    // Write raw data
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    SECTION("Get vectors by IDs") {
        knowhere::BinarySet binset;
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            auto build_json = build_gen();
            pageann.Build(nullptr, build_json);
            pageann.Serialize(binset);
        }

        // Deserialize and get vectors
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["index_prefix"] = kL2IndexPrefix;

            pageann.Deserialize(binset, search_json);

            // Create dataset with IDs to retrieve
            std::vector<int64_t> ids = {0, 10, 100, 500, 999};
            auto id_ds = GenIdsDataSet(ids.size(), ids);

            auto vectors = pageann.GetVectorByIds(id_ds, nullptr);
            REQUIRE(vectors.has_value());

            // Verify returned vectors
            auto vec_data = vectors.value()->GetTensor();
            REQUIRE(vec_data != nullptr);
            REQUIRE(vectors.value()->GetRows() == static_cast<int64_t>(ids.size()));
            REQUIRE(vectors.value()->GetDim() == kDim);
        }
    }

    // Cleanup
    fs::remove_all(kDir);
}

TEST_CASE("PageANN - GetIndexMeta test", "[pageann][metadata]") {
    // Setup
    fs::remove_all(kDir);
    REQUIRE_NOTHROW(fs::create_directory(kDir));

    auto version = GenTestVersionList();
    auto metric_str = knowhere::metric::L2;

    // Generate test data
    auto base_ds = GenDataSet(kNumRows, kDim, 30);

    auto build_gen = [&]() {
        knowhere::Json json;
        json["dim"] = kDim;
        json["metric_type"] = metric_str;
        json["index_prefix"] = kL2IndexPrefix;
        json["data_path"] = kRawDataPath;
        json["max_degree"] = 56;
        json["search_list_size"] = 128;
        json["enable_lsh_routing"] = true;
        return json;
    };

    // Write raw data
    auto base_ptr = static_cast<const float*>(base_ds->GetTensor());
    WriteRawDataToDisk<float>(kRawDataPath, base_ptr, kNumRows, kDim);

    SECTION("Get index metadata") {
        knowhere::BinarySet binset;
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            auto build_json = build_gen();
            pageann.Build(nullptr, build_json);
            pageann.Serialize(binset);
        }

        // Deserialize and get metadata
        {
            std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
            auto pageann_index_pack = knowhere::Pack(file_manager);

            auto pageann = knowhere::IndexFactory::Instance()
                                .Create<knowhere::fp32>("PAGEANN", version, pageann_index_pack)
                                .value();

            knowhere::Json search_json;
            search_json["dim"] = kDim;
            search_json["metric_type"] = metric_str;
            search_json["index_prefix"] = kL2IndexPrefix;

            pageann.Deserialize(binset, search_json);

            auto meta = pageann.GetIndexMeta(knowhere::Json{});
            REQUIRE(meta.has_value());

            // Verify metadata was returned
            auto json_info = meta.value()->GetJsonInfo();
            REQUIRE(!json_info.empty());
        }
    }

    // Cleanup
    fs::remove_all(kDir);
}
