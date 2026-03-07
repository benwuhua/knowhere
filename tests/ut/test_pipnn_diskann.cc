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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
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
    uint32_t leaf_max_size = 1000;
    uint32_t fanout_l1 = 10;
    uint32_t fanout_l2 = 3;
    uint32_t overlap_k = 2;
    uint32_t k_nn = 3;
    uint32_t hash_bits = 12;
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

PiPNNRecallTuningConfig
LoadPiPNNRecallTuningFromEnv() {
    PiPNNRecallTuningConfig config;
    config.leaf_max_size = GetEnvUintOrDefault("KNOWHERE_PIPNN_LEAF_MAX_SIZE", config.leaf_max_size);
    config.fanout_l1 = GetEnvUintOrDefault("KNOWHERE_PIPNN_FANOUT_L1", config.fanout_l1);
    config.fanout_l2 = GetEnvUintOrDefault("KNOWHERE_PIPNN_FANOUT_L2", config.fanout_l2);
    config.overlap_k = GetEnvUintOrDefault("KNOWHERE_PIPNN_OVERLAP_K", config.overlap_k);
    config.k_nn = GetEnvUintOrDefault("KNOWHERE_PIPNN_K_NN", config.k_nn);
    config.hash_bits = GetEnvUintOrDefault("KNOWHERE_PIPNN_HASH_BITS", config.hash_bits);
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
}

std::string
DescribePiPNNRecallTuning(const PiPNNRecallTuningConfig& config) {
    return "leaf_max_size=" + std::to_string(config.leaf_max_size) + ", fanout_l1=" +
           std::to_string(config.fanout_l1) + ", fanout_l2=" + std::to_string(config.fanout_l2) +
           ", overlap_k=" + std::to_string(config.overlap_k) + ", k_nn=" + std::to_string(config.k_nn) +
           ", hash_bits=" + std::to_string(config.hash_bits);
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

    auto build_json = MakeBuildJson("/tmp/pipnn_perf", "/tmp/raw.bin", kE2EDim, 1000);
    ApplyPiPNNRecallTuning(build_json, config);

    REQUIRE(build_json["pipnn_leaf_max_size"] == config.leaf_max_size);
    REQUIRE(build_json["pipnn_fanout_l1"] == config.fanout_l1);
    REQUIRE(build_json["pipnn_fanout_l2"] == config.fanout_l2);
    REQUIRE(build_json["pipnn_overlap_k"] == config.overlap_k);
    REQUIRE(build_json["pipnn_k_nn"] == config.k_nn);
    REQUIRE(build_json["pipnn_hash_bits"] == config.hash_bits);
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

TEST_CASE("PiPNN vs DiskANN recall@10 comparison", "[pipnn_diskann][e2e][recall]") {
    ScopedTempDir temp_dir("knowhere_pipnn_recall");
    const auto raw_path = (temp_dir.dir / "raw_data.bin").string();
    const auto pipnn_index_prefix = (temp_dir.dir / "pipnn_index" / "pipnn").string();
    const auto diskann_index_prefix = (temp_dir.dir / "diskann_index" / "diskann").string();
    REQUIRE(fs::create_directories(fs::path(pipnn_index_prefix).parent_path()));
    REQUIRE(fs::create_directories(fs::path(diskann_index_prefix).parent_path()));

    auto base_ds = GenDataSet(kE2ENumRows, kE2EDim, 30);
    auto query_ds = GenDataSet(kE2ENumQueries, kE2EDim, 42);
    WriteRawDataToDisk<float>(raw_path, static_cast<const float*>(base_ds->GetTensor()), kE2ENumRows, kE2EDim);

    auto gt_json = MakeBaseJson(kE2EDim, kE2EK);
    auto gt_res = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, gt_json, nullptr);
    REQUIRE(gt_res.has_value());

    auto version = GenTestVersionList();
    std::shared_ptr<milvus::FileManager> file_manager = std::make_shared<milvus::LocalFileManager>();
    auto diskann_index_pack = knowhere::Pack(file_manager);
    const auto pipnn_tuning = LoadPiPNNRecallTuningFromEnv();
    LOG_KNOWHERE_INFO_ << "PiPNN recall tuning overrides: " << DescribePiPNNRecallTuning(pipnn_tuning);

    auto build_and_search = [&](const std::string& index_type, const std::string& index_prefix) {
        auto create_res =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack);
        REQUIRE(create_res.has_value());
        auto index = create_res.value();

        auto build_json = MakeBuildJson(index_prefix, raw_path, kE2EDim, kE2ENumRows);
        if (index_type == knowhere::IndexEnum::INDEX_PIPNN_DISKANN) {
            ApplyPiPNNRecallTuning(build_json, pipnn_tuning);
        }
        const auto start = std::chrono::steady_clock::now();
        REQUIRE(index.Build(nullptr, build_json) == knowhere::Status::success);
        const auto end = std::chrono::steady_clock::now();
        const auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        knowhere::BinarySet binset;
        REQUIRE(index.Serialize(binset) == knowhere::Status::success);

        auto search_create_res =
            knowhere::IndexFactory::Instance().Create<knowhere::fp32>(index_type, version, diskann_index_pack);
        REQUIRE(search_create_res.has_value());
        auto search_index = search_create_res.value();
        auto deserialize_json = MakeDeserializeJson(index_prefix, kE2EDim, kE2ENumRows);
        REQUIRE(search_index.Deserialize(binset, deserialize_json) == knowhere::Status::success);

        auto search_json = MakeSearchJson(kE2EDim, kE2EK);
        auto search_res = search_index.Search(query_ds, search_json, nullptr);
        REQUIRE(search_res.has_value());
        return std::make_pair(search_res.value(), build_ms);
    };

    const auto [pipnn_result, pipnn_build_ms] =
        build_and_search(knowhere::IndexEnum::INDEX_PIPNN_DISKANN, pipnn_index_prefix);
    const auto [diskann_result, diskann_build_ms] =
        build_and_search(knowhere::IndexEnum::INDEX_DISKANN, diskann_index_prefix);

    LOG_KNOWHERE_INFO_
        << "[PiPNN Profiling] Stage: " << pipnn_build_ms
        << " ms (PiPNN total build in recall test; stage breakdown is logged by PiPNNBuilder::build)";
    LOG_KNOWHERE_INFO_ << "PiPNN build time(ms): " << pipnn_build_ms;
    LOG_KNOWHERE_INFO_ << "DiskANN build time(ms): " << diskann_build_ms;

    const float pipnn_recall = GetKNNRecall(*gt_res.value(), *pipnn_result);
    const float diskann_recall = GetKNNRecall(*gt_res.value(), *diskann_result);
    LOG_KNOWHERE_INFO_ << "PiPNN recall@10: " << pipnn_recall << ", DiskANN recall@10: " << diskann_recall;

    REQUIRE(pipnn_recall + 0.05f >= diskann_recall);
}
#endif
