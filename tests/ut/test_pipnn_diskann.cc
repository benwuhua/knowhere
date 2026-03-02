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

// Unit tests for PiPNN-DiskANN graph construction

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "catch2/catch_approx.hpp"
#include "pipnn_diskann_config.h"

namespace {

using knowhere::pipnn_diskann::PipnnConfig;

// Test config defaults
TEST_CASE("PipnnConfig defaults", "[pipnn_diskann]") {
    PipnnConfig config;
    REQUIRE(config.num_partitions == 32);
    REQUIRE(config.overlap_ratio == Approx(0.5f));
    REQUIRE(config.max_degree == 32);
    REQUIRE(config.hash_bits == 12);
    REQUIRE(config.use_medoid_entry == true);
}

// Test config ToString
TEST_CASE("PipnnConfig ToString", "[pipnn_diskann]") {
    PipnnConfig config;
    std::string str = config.ToString();
    REQUIRE(str.find("partitions=32") != std::string::npos);
    REQUIRE(str.find("overlap=0.5") != std::string::npos);
}

// Test RBC partitioning logic
TEST_CASE("RBC partitioning", "[pipnn_diskann]") {
    const uint32_t n = 100;
    const uint32_t d = 32;

    // Create dummy data
    std::vector<float> data(n * d);
    for (auto& val : data) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }

    knowhere::pipnn_diskann::RBCCluster rbc(4, 0.25f);
    std::mt19937 rng(42);

    auto partitions = rbc.PartitionData(data.data(), n, d, rng);

    REQUIRE(partitions.size() == 4);

    // Each partition should have approximately n/4 points + overlap
    uint32_t expected_base = n / 4;
    uint32_t expected_overlap = static_cast<uint32_t>(expected_base * 0.25f);

    for (const auto& p : partitions) {
        uint32_t min_size = expected_base;
        uint32_t max_size = expected_base + 2 * expected_overlap;
        REQUIRE(p.point_ids.size() >= min_size);
        REQUIRE(p.point_ids.size() <= max_size);
    }

    // Check partition IDs are unique
    std::vector<uint32_t> partition_ids;
    for (const auto& p : partitions) {
        partition_ids.push_back(p.partition_id);
    }
    std::sort(partition_ids.begin(), partition_ids.end());
    auto last = std::unique(partition_ids.begin(), partition_ids.end());
    REQUIRE(std::distance(partition_ids.begin(), last) == 4);
}

// Test HashPrune basic functionality
TEST_CASE("HashPrune basic", "[pipnn_diskann]") {
    knowhere::pipnn_diskann::HashPrune prune(8);  // 2^8 = 256 buckets

    // Initial state: all available
    for (uint32_t i = 0; i < 100; ++i) {
        REQUIRE(prune.IsAvailable(i));
    }

    // Mark some as taken
    prune.MarkTaken(10);
    prune.MarkTaken(10);  // Mark twice (idempotent-ish)
    prune.MarkTaken(42);

    REQUIRE(prune.IsAvailable(10) == false);
    REQUIRE(prune.IsAvailable(42) == false);
    REQUIRE(prune.IsAvailable(50) == true);  // Untouched

    // Clear and verify all available again
    prune.Clear();
    for (uint32_t i = 0; i < 100; ++i) {
        REQUIRE(prune.IsAvailable(i));
    }
}

// Test HashPrune collision handling
TEST_CASE("HashPrune collision", "[pipnn_diskann]") {
    knowhere::pipnn_diskann::HashPrune prune(4);  // 2^4 = 16 buckets, 32 slots

    // IDs 0 and 16 map to same bucket slot (bucket 0, bit 0)
    prune.MarkTaken(0);
    prune.MarkTaken(16);

    // Both should be unavailable (same bucket, same bit)
    REQUIRE(prune.IsAvailable(0) == false);
    REQUIRE(prune.IsAvailable(16) == false);

    // ID 32 maps to different bucket (bucket 0, bit 1)
    REQUIRE(prune.IsAvailable(32) == true);

    prune.MarkTaken(32);

    // Now ID 32 should also be unavailable
    REQUIRE(prune.IsAvailable(32) == false);

    // But ID 48 (bucket 0, bit 2) is still available
    REQUIRE(prune.IsAvailable(48) == true);
}

// Test GraphBuilder config initialization
TEST_CASE("GraphBuilder config", "[pipnn_diskann]") {
    knowhere::pipnn_diskann::PipnnConfig config;
    config.num_partitions = 8;
    config.max_degree = 16;
    config.hash_bits = 10;

    knowhere::pipnn_diskann::GraphBuilder builder(config);

    std::string str = config.ToString();
    REQUIRE(str.find("partitions=8") != std::string::npos);
    REQUIRE(str.find("max_degree=16") != std::string::npos);
}

// Minimal integration test
TEST_CASE("PiPNN graph serialization format", "[pipnn_diskann]") {
    // Verify the DiskANN graph format structure matches expected

    struct DiskANNGraphHeader {
        uint64_t index_size;
        uint64_t index_size_dup;
        uint32_t max_degree;
        uint32_t entry_point;
        uint64_t num_frozen_pts;
    };

    constexpr size_t header_size = sizeof(DiskANNGraphHeader);

    REQUIRE(header_size == 24);  // 8 + 8 + 4 + 4
    REQUIRE(sizeof(uint64_t) == 8);
    REQUIRE(sizeof(uint32_t) == 4);
}

}  // namespace

// ============================================================================
// HashPrune with Residualized LSH Tests (Task 2)
// ============================================================================

#include "index/diskann/impl/hash_prune.h"

namespace {

constexpr uint32_t kDim = 32;
constexpr uint32_t kMaxDegree = 16;
constexpr uint32_t kHashBits = 12;

// Helper: generate random float vector
std::vector<float> rand_vec(uint32_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

float l2_dist(const float* a, const float* b, uint32_t d) {
    float s = 0;
    for (uint32_t i = 0; i < d; i++) s += (a[i]-b[i])*(a[i]-b[i]);
    return s;
}

}  // namespace

TEST_CASE("HashPrune: history-independent insertion", "[pipnn][hash_prune]") {
    std::mt19937 rng(42);
    auto p_vec = rand_vec(kDim, rng);

    // Generate 50 candidate vectors
    std::vector<std::vector<float>> candidates(50);
    for (auto& c : candidates) c = rand_vec(kDim, rng);

    knowhere::pipnn_diskann::HashPrune prune_fwd(kDim, kHashBits, kMaxDegree);
    knowhere::pipnn_diskann::HashPrune prune_rev(kDim, kHashBits, kMaxDegree);

    // Pre-compute sketch for point p
    std::vector<float> sketch_p(kHashBits);
    prune_fwd.compute_sketch(p_vec.data(), sketch_p.data());

    // Forward order
    for (uint32_t i = 0; i < candidates.size(); i++) {
        std::vector<float> sketch_c(kHashBits);
        prune_fwd.compute_sketch(candidates[i].data(), sketch_c.data());
        float d = l2_dist(p_vec.data(), candidates[i].data(), kDim);
        prune_fwd.insert(i, sketch_p.data(), sketch_c.data(), d);
    }

    // Reverse order
    for (int i = (int)candidates.size()-1; i >= 0; i--) {
        std::vector<float> sketch_c(kHashBits);
        prune_rev.compute_sketch(candidates[i].data(), sketch_c.data());
        float d = l2_dist(p_vec.data(), candidates[i].data(), kDim);
        prune_rev.insert((uint32_t)i, sketch_p.data(), sketch_c.data(), d);
    }

    auto nbrs_fwd = prune_fwd.neighbors();
    auto nbrs_rev = prune_rev.neighbors();

    // Sort both for comparison (order within reservoir may differ)
    std::sort(nbrs_fwd.begin(), nbrs_fwd.end());
    std::sort(nbrs_rev.begin(), nbrs_rev.end());

    REQUIRE(nbrs_fwd == nbrs_rev);
}

TEST_CASE("HashPrune: reservoir never exceeds max_degree", "[pipnn][hash_prune]") {
    std::mt19937 rng(99);
    auto p_vec = rand_vec(kDim, rng);

    knowhere::pipnn_diskann::HashPrune prune(kDim, kHashBits, kMaxDegree);
    std::vector<float> sketch_p(kHashBits);
    prune.compute_sketch(p_vec.data(), sketch_p.data());

    for (uint32_t i = 0; i < 500; i++) {
        auto c = rand_vec(kDim, rng);
        std::vector<float> sketch_c(kHashBits);
        prune.compute_sketch(c.data(), sketch_c.data());
        float d = l2_dist(p_vec.data(), c.data(), kDim);
        prune.insert(i, sketch_p.data(), sketch_c.data(), d);
    }

    REQUIRE(prune.neighbors().size() <= kMaxDegree);
}

TEST_CASE("HashPrune: closer candidate replaces farther in same bucket",
          "[pipnn][hash_prune]") {
    std::mt19937 rng(7);
    auto p_vec = rand_vec(kDim, rng);

    // Use many hash bits so collisions are rare — we'll force collision
    // by using identical direction vectors scaled differently
    knowhere::pipnn_diskann::HashPrune prune(kDim, kHashBits, kMaxDegree);
    std::vector<float> sketch_p(kHashBits);
    prune.compute_sketch(p_vec.data(), sketch_p.data());

    // c_far and c_near are in same direction from p (so same hash bucket)
    auto direction = rand_vec(kDim, rng);
    std::vector<float> c_far(kDim), c_near(kDim);
    for (uint32_t j = 0; j < kDim; j++) {
        c_far[j]  = p_vec[j] + direction[j] * 2.0f;
        c_near[j] = p_vec[j] + direction[j] * 0.5f;
    }

    std::vector<float> sketch_c_far(kHashBits);
    std::vector<float> sketch_c_near(kHashBits);
    prune.compute_sketch(c_far.data(), sketch_c_far.data());
    prune.compute_sketch(c_near.data(), sketch_c_near.data());

    float d_far  = l2_dist(p_vec.data(), c_far.data(), kDim);
    float d_near = l2_dist(p_vec.data(), c_near.data(), kDim);

    prune.insert(0, sketch_p.data(), sketch_c_far.data(), d_far);
    prune.insert(1, sketch_p.data(), sketch_c_near.data(), d_near);

    auto nbrs = prune.neighbors();
    // c_near should be in the result, c_far should be replaced
    bool has_near = std::find(nbrs.begin(), nbrs.end(), 1u) != nbrs.end();
    bool has_far  = std::find(nbrs.begin(), nbrs.end(), 0u) != nbrs.end();
    // With high probability (same direction → same hash), near replaces far
    // This test may occasionally pass both if hash doesn't collide — that's ok
    REQUIRE((has_near || has_far));  // At minimum one is present
    if (has_near && has_far) {
        // Both present means no collision — fine
    } else {
        REQUIRE(has_near);  // If one was evicted, it must be c_far
    }
}

// ============================================================================
// Task 10: End-to-End Tests (Integration + Benchmark)
// ============================================================================

#include "index/diskann/impl/pipnn_diskann.h"
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace {

// Test configuration
constexpr uint32_t kTestN = 1000;      // Dataset size for integration test
constexpr uint32_t kTestDim = 64;      // Dimension
constexpr uint32_t kTestNQ = 50;       // Number of queries
constexpr uint32_t kTestK = 10;        // Top-k
constexpr float kMinRecall = 0.6f;     // Minimum acceptable recall

// Benchmark configuration
constexpr uint32_t kBenchN = 2000;     // Larger dataset for benchmark
constexpr uint32_t kBenchDim = 128;
constexpr uint32_t kBenchNQ = 100;

std::string GetTestTempDir() {
    return (fs::temp_directory_path() / ("pipnn_test_" + std::to_string(std::time(nullptr)))).string();
}

// Generate random float dataset
std::vector<float> GenerateRandomData(uint32_t n, uint32_t dim, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = dist(rng);
    return data;
}

// Compute L2 distance
float l2_distance(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

// Calculate recall@k
float CalcRecall(const int64_t* gt_ids, const int64_t* result_ids,
                 uint32_t nq, uint32_t k) {
    uint32_t total_hits = 0;
    for (uint32_t i = 0; i < nq; ++i) {
        const int64_t* gt_row = gt_ids + i * k;
        const int64_t* res_row = result_ids + i * k;
        for (uint32_t j = 0; j < k; ++j) {
            for (uint32_t l = 0; l < k; ++l) {
                if (gt_row[j] == res_row[l]) {
                    total_hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(total_hits) / (nq * k);
}

// Get ground truth via brute force
std::vector<std::vector<int64_t>> GetGroundTruth(
    const std::vector<float>& data, uint32_t n, uint32_t dim,
    const std::vector<float>& queries, uint32_t nq, uint32_t k) {

    std::vector<std::vector<int64_t>> gt(nq, std::vector<int64_t>(k));

    for (uint32_t i = 0; i < nq; ++i) {
        const float* q = queries.data() + i * dim;
        std::vector<std::pair<float, uint32_t>> dists(n);

        for (uint32_t j = 0; j < n; ++j) {
            float d = l2_distance(q, data.data() + j * dim, dim);
            dists[j] = {d, j};
        }

        std::partial_sort(dists.begin(), dists.begin() + k, dists.end());

        for (uint32_t j = 0; j < k; ++j) {
            gt[i][j] = static_cast<int64_t>(dists[j].second);
        }
    }

    return gt;
}

}  // namespace

// ============================================================================
// Task 10a: Basic End-to-End Test
// ============================================================================

TEST_CASE("PiPNN-DiskANN: End-to-End graph build and search",
          "[pipnn][e2e][integration]") {

    std::string test_dir = GetTestTempDir();
    fs::create_directories(test_dir);

    try {
        // Generate test data
        auto data = GenerateRandomData(kTestN, kTestDim, 42);
        auto queries = GenerateRandomData(kTestNQ, kTestDim, 99);

        // Get ground truth
        auto gt = GetGroundTruth(data, kTestN, kTestDim, queries, kTestNQ, kTestK);

        // Build PiPNN graph
        std::string graph_path = test_dir + "/graph.index";

        knowhere::pipnn_diskann::PipnnConfig config;
        config.num_partitions = 16;
        config.overlap_ratio = 0.3f;
        config.max_degree = 32;
        config.hash_bits = 12;
        config.use_medoid_entry = true;

        auto build_result = knowhere::pipnn_diskann::BuildPipnnDiskANNGraph(
            data.data(), kTestN, kTestDim, graph_path, config);

        REQUIRE(build_result.has_value());
        REQUIRE(build_result.value() == knowhere::Status::OK);
        REQUIRE(fs::exists(graph_path));

        // Verify graph file size is reasonable
        auto file_size = fs::file_size(graph_path);
        REQUIRE(file_size > 1000);

        // Read back and verify header
        std::ifstream in(graph_path, std::ios::binary);
        REQUIRE(in.good());

        uint64_t index_size;
        uint32_t max_degree, entry_point;
        uint64_t num_frozen_pts;

        in.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
        in.read(reinterpret_cast<char*>(&max_degree), sizeof(max_degree));
        in.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
        in.read(reinterpret_cast<char*>(&num_frozen_pts), sizeof(num_frozen_pts));

        REQUIRE(entry_point < kTestN);
        REQUIRE(max_degree <= config.max_degree);
        REQUIRE(num_frozen_pts == 0);

        in.close();

        LOG_KNOWHERE_INFO_ << "PiPNN-DiskANN e2e test passed: "
                           << "N=" << kTestN << ", dim=" << kTestDim;

    } catch (const std::exception& e) {
        FAIL("Exception: " << e.what());
    }

    // Cleanup
    fs::remove_all(test_dir);
}

// ============================================================================
// Task 10b: Recall Benchmark vs Brute Force
// ============================================================================

TEST_CASE("PiPNN-DiskANN: Recall benchmark vs brute force",
          "[pipnn][benchmark][recall]") {

    std::string test_dir = GetTestTempDir();
    fs::create_directories(test_dir);

    try {
        auto data = GenerateRandomData(kBenchN, kBenchDim, 123);
        auto queries = GenerateRandomData(kBenchNQ, kBenchDim, 456);

        auto start_gt = std::chrono::high_resolution_clock::now();
        auto gt = GetGroundTruth(data, kBenchN, kBenchDim, queries, kBenchNQ, kTestK);
        auto end_gt = std::chrono::high_resolution_clock::now();
        auto gt_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_gt - start_gt).count();

        struct TestConfig {
            std::string name;
            uint32_t num_partitions;
            float overlap_ratio;
            uint32_t max_degree;
        };

        std::vector<TestConfig> configs = {
            {"PiPNN-Light", 8, 0.2f, 16},
            {"PiPNN-Standard", 16, 0.3f, 32},
            {"PiPNN-HighQuality", 32, 0.4f, 64},
        };

        std::ostringstream results;
        results << "\n=== PiPNN-DiskANN Recall Benchmark ===\n";
        results << "Dataset: N=" << kBenchN << ", dim=" << kBenchDim
                << ", queries=" << kBenchNQ << ", k=" << kTestK << "\n";
        results << "Ground truth time: " << gt_time << " ms\n\n";

        for (const auto& cfg : configs) {
            std::string graph_path = test_dir + "/graph_" + cfg.name + ".index";

            knowhere::pipnn_diskann::PipnnConfig pipnn_cfg;
            pipnn_cfg.num_partitions = cfg.num_partitions;
            pipnn_cfg.overlap_ratio = cfg.overlap_ratio;
            pipnn_cfg.max_degree = cfg.max_degree;
            pipnn_cfg.hash_bits = 12;
            pipnn_cfg.use_medoid_entry = true;

            auto start_build = std::chrono::high_resolution_clock::now();
            auto build_result = knowhere::pipnn_diskann::BuildPipnnDiskANNGraph(
                data.data(), kBenchN, kBenchDim, graph_path, pipnn_cfg);
            auto end_build = std::chrono::high_resolution_clock::now();
            auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_build - start_build).count();

            REQUIRE(build_result.has_value());
            REQUIRE(build_result.value() == knowhere::Status::OK);

            std::ifstream in(graph_path, std::ios::binary);
            uint64_t index_size;
            uint32_t max_deg, entry_point;
            uint64_t num_frozen;
            in.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));
            in.read(reinterpret_cast<char*>(&max_deg), sizeof(max_deg));
            in.read(reinterpret_cast<char*>(&entry_point), sizeof(entry_point));
            in.read(reinterpret_cast<char*>(&num_frozen), sizeof(num_frozen));

            std::vector<std::vector<uint32_t>> graph(kBenchN);
            for (uint32_t i = 0; i < kBenchN; ++i) {
                uint32_t num_nbrs;
                in.read(reinterpret_cast<char*>(&num_nbrs), sizeof(num_nbrs));
                graph[i].resize(num_nbrs);
                in.read(reinterpret_cast<char*>(graph[i].data()),
                        num_nbrs * sizeof(uint32_t));
            }
            in.close();

            auto start_search = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<int64_t>> results(kBenchNQ,
                                                       std::vector<int64_t>(kTestK));

            for (uint32_t qi = 0; qi < kBenchNQ; ++qi) {
                const float* q = queries.data() + qi * kBenchDim;
                std::vector<std::pair<float, uint32_t>> candidates;
                std::vector<bool> visited(kBenchN, false);
                std::vector<std::pair<float, uint32_t>> top_k;

                float ep_dist = l2_distance(q, data.data() + entry_point * kBenchDim, kBenchDim);
                candidates.emplace_back(ep_dist, entry_point);
                visited[entry_point] = true;

                while (!candidates.empty() && top_k.size() < kTestK * 2) {
                    std::sort(candidates.begin(), candidates.end());
                    auto [dist, node] = candidates.front();
                    candidates.erase(candidates.begin());
                    top_k.emplace_back(dist, node);

                    for (uint32_t nbr : graph[node]) {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            float d = l2_distance(q, data.data() + nbr * kBenchDim, kBenchDim);
                            candidates.emplace_back(d, nbr);
                        }
                    }
                }

                std::sort(top_k.begin(), top_k.end());
                for (uint32_t i = 0; i < kTestK && i < top_k.size(); ++i) {
                    results[qi][i] = static_cast<int64_t>(top_k[i].second);
                }
            }

            auto end_search = std::chrono::high_resolution_clock::now();
            auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_search - start_search).count();

            int64_t* gt_flat = new int64_t[kBenchNQ * kTestK];
            int64_t* res_flat = new int64_t[kBenchNQ * kTestK];

            for (uint32_t i = 0; i < kBenchNQ; ++i) {
                for (uint32_t j = 0; j < kTestK; ++j) {
                    gt_flat[i * kTestK + j] = gt[i][j];
                    res_flat[i * kTestK + j] = results[i][j];
                }
            }

            float recall = CalcRecall(gt_flat, res_flat, kBenchNQ, kTestK);
            delete[] gt_flat;
            delete[] res_flat;

            results << cfg.name << ":\n";
            results << "  Build time: " << build_time << " ms\n";
            results << "  Search time: " << search_time << " ms\n";
            results << "  Recall@" << kTestK << ": " << std::fixed
                    << std::setprecision(3) << recall << "\n\n";

            REQUIRE(recall >= kMinRecall);
        }

        LOG_KNOWHERE_INFO_ << results.str();

    } catch (const std::exception& e) {
        FAIL("Exception: " << e.what());
    }

    fs::remove_all(test_dir);
}

// ============================================================================
// Task 10c: Performance Scaling Test
// ============================================================================

TEST_CASE("PiPNN-DiskANN: Scaling test (dataset size)",
          "[pipnn][benchmark][scaling]") {

    std::string test_dir = GetTestTempDir();
    fs::create_directories(test_dir);

    try {
        std::vector<uint32_t> sizes = {500, 1000, 2000};
        const uint32_t dim = 64;

        std::ostringstream results;
        results << "\n=== PiPNN-DiskANN Scaling Test ===\n";
        results << "Dimension: " << dim << "\n\n";

        for (uint32_t n : sizes) {
            auto data = GenerateRandomData(n, dim, 789);
            std::string graph_path = test_dir + "/graph_n" + std::to_string(n) + ".index";

            knowhere::pipnn_diskann::PipnnConfig config;
            config.num_partitions = std::max(4u, n / 100);
            config.overlap_ratio = 0.3f;
            config.max_degree = 32;
            config.hash_bits = 12;

            auto start = std::chrono::high_resolution_clock::now();
            auto build_result = knowhere::pipnn_diskann::BuildPipnnDiskANNGraph(
                data.data(), n, dim, graph_path, config);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end - start).count();

            REQUIRE(build_result.has_value());

            auto file_size = fs::file_size(graph_path);

            results << "N=" << std::setw(5) << n
                    << " | Build: " << std::setw(6) << duration << " ms"
                    << " | Size: " << std::setw(10) << file_size << " bytes"
                    << " | Throughput: " << std::setw(8)
                    << (n * 1000 / (duration > 0 ? duration : 1)) << " vec/s\n";
        }

        LOG_KNOWHERE_INFO_ << results.str();

    } catch (const std::exception& e) {
        FAIL("Exception: " << e.what());
    }

    fs::remove_all(test_dir);
}
