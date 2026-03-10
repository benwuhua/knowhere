// Copyright (C) 2019-2023 Zilliz. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "knowhere/log.h"

#include "index/diskann/impl/hash_prune.h"
#include "index/diskann/impl/rbc_partition.h"

namespace knowhere::pipnn_diskann {

class SpinMutex {
 public:
    SpinMutex() noexcept = default;

    SpinMutex(const SpinMutex&) = delete;
    SpinMutex&
    operator=(const SpinMutex&) = delete;

    bool
    try_lock() noexcept {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

    void
    lock() noexcept {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    }

    void
    unlock() noexcept {
        flag_.clear(std::memory_order_release);
    }

 private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class PiPNNBuilder {
 public:
    struct Config {
        uint32_t k_nn = 64;
        uint32_t hash_bits = 12;
        uint32_t max_degree = 32;
        float alpha = 1.2f;
        bool final_prune = true;
        uint32_t num_threads = 0;
        size_t leaf_max_size = 0;
        uint32_t fanout_l1 = 8;
        uint32_t fanout_l2 = 4;
        uint32_t fanout_rest = 2;
        uint32_t overlap_k = 2;
        bool enable_cross_leaf_union = true;
        uint64_t base_seed = 42;
    };

    PiPNNBuilder();
    explicit PiPNNBuilder(const Config& config);

    std::vector<std::vector<uint32_t>>
    build(const float* data, uint32_t n, uint32_t dim) const;

 private:
    struct RbcCoverageStats {
        uint32_t leaves = 0;
        uint32_t min_leaf_size = 0;
        uint32_t max_leaf_size = 0;
        double avg_leaf_size = 0.0;
        double avg_membership = 0.0;
        uint32_t min_membership = 0;
        uint32_t max_membership = 0;
        uint32_t single_membership_nodes = 0;
    };

    struct QualityProbeStats {
        std::atomic<int64_t> sampled_nodes{0};
        std::atomic<int64_t> candidate_overlap_hits{0};
        std::atomic<int64_t> hash_overlap_hits{0};
        std::atomic<int64_t> inserted_overlap_hits{0};
        std::atomic<int64_t> pre_final_retained_overlap_hits{0};
        std::atomic<int64_t> final_prune_overlap_hits{0};
        std::atomic<int64_t> candidate_total{0};
        std::atomic<int64_t> hash_total{0};
        std::atomic<int64_t> inserted_total{0};
        std::atomic<int64_t> pre_final_retained_total{0};
        std::atomic<int64_t> final_prune_total{0};
        std::atomic<int64_t> collision_replace{0};
        std::atomic<int64_t> collision_reject{0};
        std::atomic<int64_t> append_accept{0};
        std::atomic<int64_t> reservoir_replace{0};
        std::atomic<int64_t> reservoir_reject{0};
        std::atomic<int64_t> insert_append{0};
        std::atomic<int64_t> insert_duplicate{0};
        std::atomic<int64_t> insert_degree_full{0};
        std::atomic<int64_t> pre_final_retained_degree{0};
        std::atomic<int64_t> final_prune_degree_before{0};
        std::atomic<int64_t> final_prune_degree_after{0};
    };

    struct BuildContext {
        explicit BuildContext(uint32_t n)
            : adjacency(n),
              node_locks(std::make_unique<SpinMutex[]>(n)),
              leaf_total_ns(0),
              gemm_total_ns(0),
              hash_prune_total_ns(0),
              edge_insert_ns(0),
              edge_insert_count(0),
              edge_lock_wait_ns(0),
              edge_lock_contention_count(0),
              processed_leaves(0),
              membership_counts(n, 0) {
        }

        std::vector<std::vector<uint32_t>> adjacency;
        std::unique_ptr<SpinMutex[]> node_locks;
        std::atomic<int64_t> leaf_total_ns;
        std::atomic<int64_t> gemm_total_ns;
        std::atomic<int64_t> hash_prune_total_ns;
        std::atomic<int64_t> edge_insert_ns;
        std::atomic<int64_t> edge_insert_count;
        std::atomic<int64_t> edge_lock_wait_ns;
        std::atomic<int64_t> edge_lock_contention_count;
        std::atomic<uint32_t> processed_leaves;
        std::vector<uint32_t> membership_counts;

        // Global per-point sketches: flat layout [point_id * hash_bits + bit_idx]
        // Populated in build() before leaf processing. Size = n * hash_bits.
        std::vector<float> global_sketches_flat;

        // Global per-point reservoirs for the new streaming path.
        // Populated in build(). Size = n. Each reservoir is protected by node_locks[i].
        std::vector<HashReservoir> global_reservoirs;

        std::vector<std::vector<uint32_t>> point_to_leaf_ids;
        const std::vector<Leaf>* leaves = nullptr;
        RbcCoverageStats rbc_coverage;
        QualityProbeStats quality_probe;
    };

    static RbcCoverageStats
    collect_rbc_coverage_stats(const std::vector<Leaf>& leaves, uint32_t n);

    void
    collect_pre_final_retained_probe_stats(const float* data, uint32_t n, uint32_t dim, BuildContext& context) const;

    static void
    log_quality_probe_stats(const BuildContext& context, bool final_prune_enabled);

    void
    process_leaf(const float* data, uint32_t dim, const Leaf& leaf, BuildContext& context) const;

    void
    robust_prune_pass(const float* data, uint32_t n, uint32_t dim, std::vector<std::vector<uint32_t>>& adjacency,
                      BuildContext& context) const;

    static float
    l2sq(const float* a, const float* b, uint32_t dim);

 private:
    Config config_;
};

}  // namespace knowhere::pipnn_diskann
