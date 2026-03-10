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

#include "index/diskann/impl/pipnn_builder.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include "knowhere/log.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace knowhere::pipnn_diskann {
namespace {

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct EdgeLockStats {
    int64_t wait_ns = 0;
    bool contended = false;
};

enum class NeighborInsertResult {
    kAppend,
    kDuplicate,
    kDegreeFull,
};

struct BidirectionalInsertResult {
    NeighborInsertResult forward = NeighborInsertResult::kAppend;
    NeighborInsertResult reverse = NeighborInsertResult::kAppend;
};

class NodeLockGuard {
 public:
    NodeLockGuard(SpinMutex& first, SpinMutex* second, EdgeLockStats& stats)
        : first_(&first),
          second_(second) {
        using clock = std::chrono::steady_clock;
        const auto lock_start = clock::now();
        bool contended = false;

        if (!first_->try_lock()) {
            contended = true;
            first_->lock();
        }
        if (second_ != nullptr && second_ != first_) {
            if (!second_->try_lock()) {
                contended = true;
                second_->lock();
            }
        }

        if (contended) {
            stats.contended = true;
            stats.wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - lock_start).count();
        }
    }

    ~NodeLockGuard() {
        if (second_ != nullptr && second_ != first_) {
            second_->unlock();
        }
        first_->unlock();
    }

    NodeLockGuard(const NodeLockGuard&) = delete;
    NodeLockGuard&
    operator=(const NodeLockGuard&) = delete;

 private:
    SpinMutex* first_;
    SpinMutex* second_;
};

NeighborInsertResult
insert_neighbor(std::vector<uint32_t>& neighbors, uint32_t dst, uint32_t max_degree) {
    if (std::find(neighbors.begin(), neighbors.end(), dst) != neighbors.end()) {
        return NeighborInsertResult::kDuplicate;
    }
    if (neighbors.size() < max_degree) {
        neighbors.push_back(dst);
        return NeighborInsertResult::kAppend;
    }
    return NeighborInsertResult::kDegreeFull;
}

BidirectionalInsertResult
insert_bidirectional_edge(uint32_t u, uint32_t v, std::vector<std::vector<uint32_t>>& adjacency,
                          SpinMutex* node_locks, uint32_t max_degree,
                          EdgeLockStats* lock_stats = nullptr) {
    if (u == v) {
        return {};
    }

    EdgeLockStats local_stats;
    EdgeLockStats& stats = lock_stats == nullptr ? local_stats : *lock_stats;
    const uint32_t first_id = std::min(u, v);
    const uint32_t second_id = std::max(u, v);
    NodeLockGuard guard(node_locks[first_id], &node_locks[second_id], stats);
    BidirectionalInsertResult result;
    result.forward = insert_neighbor(adjacency[u], v, max_degree);
    result.reverse = insert_neighbor(adjacency[v], u, max_degree);
    return result;
}

void
dedup_and_drop_self(uint32_t self_id, std::vector<uint32_t>& neighbors) {
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), self_id), neighbors.end());
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
}

bool
should_probe_node(uint32_t global_id) {
    return global_id < 8 || global_id % 65536 == 0;
}

size_t
count_overlap_with_exact_topk(const std::vector<std::pair<float, uint32_t>>& exact_topk, const std::vector<uint32_t>& ids) {
    size_t hits = 0;
    for (uint32_t id : ids) {
        hits += std::find_if(exact_topk.begin(), exact_topk.end(), [&](const auto& item) { return item.second == id; }) !=
                        exact_topk.end()
                    ? 1
                    : 0;
    }
    return hits;
}

uint32_t
candidate_budget_for_leaf(const PiPNNBuilder::Config& config, uint32_t leaf_size) {
    if (leaf_size <= 1) {
        return 0;
    }

    const uint32_t leaf_cap = leaf_size - 1;
    const uint32_t overlap_multiplier = std::max<uint32_t>(2, config.overlap_k);
    const uint32_t expanded_budget = std::max<uint32_t>(config.k_nn, config.k_nn * overlap_multiplier);
    const uint32_t bounded_budget = std::min<uint32_t>(leaf_cap, std::max<uint32_t>(config.max_degree * 2, expanded_budget));
    return std::max<uint32_t>(config.k_nn, bounded_budget);
}

uint32_t
cross_leaf_budget(const PiPNNBuilder::Config& config) {
    return std::max<uint32_t>(config.k_nn, std::min<uint32_t>(config.max_degree, config.k_nn * 2));
}

bool
retained_degree_limit_enabled() {
    const char* raw = std::getenv("KNOWHERE_PIPNN_ENABLE_RETAINED_DEGREE_LIMIT");
    if (raw == nullptr || raw[0] == '\0') {
        return true;
    }
    return std::string_view(raw) != "0" && std::string_view(raw) != "false" && std::string_view(raw) != "FALSE";
}

uint32_t
retained_degree_limit(const PiPNNBuilder::Config& config) {
    if (!config.final_prune || !retained_degree_limit_enabled()) {
        return config.max_degree;
    }

    const uint64_t expanded_limit = static_cast<uint64_t>(config.max_degree) * 2;
    return static_cast<uint32_t>(std::max<uint64_t>(config.max_degree, expanded_limit));
}

bool
cross_leaf_union_enabled(const PiPNNBuilder::Config& config) {
    const char* raw = std::getenv("KNOWHERE_PIPNN_ENABLE_CROSS_LEAF_UNION");
    if (raw == nullptr || raw[0] == '\0') {
        return config.enable_cross_leaf_union;
    }
    return std::string_view(raw) != "0" && std::string_view(raw) != "false" && std::string_view(raw) != "FALSE";
}

bool
boundary_leader_bridge_enabled() {
    const char* raw = std::getenv("KNOWHERE_PIPNN_ENABLE_BOUNDARY_LEADER_BRIDGE");
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    return std::string_view(raw) != "0" && std::string_view(raw) != "false" && std::string_view(raw) != "FALSE";
}

uint32_t
boundary_leader_bridge_budget(const PiPNNBuilder::Config& config) {
    return std::max<uint32_t>(1, std::min<uint32_t>(config.max_degree / 2, config.k_nn));
}

constexpr uint32_t kLeafProgressLogInterval = 256;

}  // namespace

PiPNNBuilder::PiPNNBuilder()
    : config_{} {
}

PiPNNBuilder::PiPNNBuilder(const Config& config)
    : config_(config) {
}

std::vector<std::vector<uint32_t>>
PiPNNBuilder::build(const float* data, uint32_t n, uint32_t dim) const {
    if (data == nullptr || n == 0 || dim == 0) {
        return {};
    }

    using clock = std::chrono::steady_clock;
    const auto ns_to_ms = [](int64_t ns) { return static_cast<double>(ns) / 1e6; };

    BuildContext context(n);
    const uint32_t build_degree_limit = retained_degree_limit(config_);
    for (auto& neighbors : context.adjacency) {
        neighbors.reserve(build_degree_limit);
    }

    RBCPartitioner::Config rbc_config;
    const size_t default_leaf_max_size =
        std::max<size_t>(64, std::max<size_t>(config_.k_nn * 2, config_.max_degree * 2));
    rbc_config.leaf_max_size = config_.leaf_max_size == 0 ? default_leaf_max_size : config_.leaf_max_size;
    rbc_config.fanout_l1 = config_.fanout_l1;
    rbc_config.fanout_l2 = config_.fanout_l2;
    rbc_config.fanout_rest = config_.fanout_rest;
    rbc_config.overlap_k = config_.overlap_k;
    rbc_config.base_seed = config_.base_seed;
    RBCPartitioner partitioner(rbc_config);

    const auto rbc_start = clock::now();
    const auto leaves = partitioner.partition(data, n, dim);
    context.leaves = &leaves;
    context.point_to_leaf_ids.assign(n, {});
    for (uint32_t leaf_id = 0; leaf_id < leaves.size(); ++leaf_id) {
        for (uint32_t point_id : leaves[leaf_id].point_ids) {
            if (point_id < n) {
                context.point_to_leaf_ids[point_id].push_back(leaf_id);
            }
        }
    }
    context.rbc_coverage = collect_rbc_coverage_stats(leaves, n);
    const auto rbc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - rbc_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(rbc_ns)
                       << " ms (RBC partition, leaves=" << leaves.size() << ", num_points=" << n << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Graph Probe] RBC coverage leaves=" << context.rbc_coverage.leaves
                       << ", avg_leaf_size=" << context.rbc_coverage.avg_leaf_size
                       << ", min_leaf_size=" << context.rbc_coverage.min_leaf_size
                       << ", max_leaf_size=" << context.rbc_coverage.max_leaf_size
                       << ", avg_membership=" << context.rbc_coverage.avg_membership
                       << ", min_membership=" << context.rbc_coverage.min_membership
                       << ", max_membership=" << context.rbc_coverage.max_membership
                       << ", single_membership_nodes=" << context.rbc_coverage.single_membership_nodes;

    // Pre-compute all point sketches using one shared HashPrune instance.
    // HashPrune with max_degree=1 is used only for compute_sketch; the reservoir
    // is unused. All instances with same (dim, hash_bits) produce identical
    // hyperplanes (deterministic seed), so one instance suffices.
    {
        const auto sketch_start = clock::now();
        HashPrune sketch_engine(dim, config_.hash_bits, /*max_degree=*/1);
        context.global_sketches_flat.resize(static_cast<size_t>(n) * config_.hash_bits);
        for (uint32_t i = 0; i < n; ++i) {
            sketch_engine.compute_sketch(data + static_cast<size_t>(i) * dim,
                                         context.global_sketches_flat.data() + static_cast<size_t>(i) * config_.hash_bits);
        }
        const auto sketch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - sketch_start).count();
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(sketch_ns)
                           << " ms (Global sketch pre-computation, num_points=" << n
                           << ", sketch_bytes=" << (context.global_sketches_flat.size() * sizeof(float)) << ")";
    }

    // Allocate global per-point reservoirs.
    {
        context.global_reservoirs.assign(n, HashReservoir(config_.hash_bits, config_.max_degree));
        const size_t reservoir_bytes = static_cast<size_t>(n) * (static_cast<size_t>(config_.max_degree) * 8 + 16);
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: global reservoirs allocated"
                           << ", num_points=" << n
                           << ", reservoir_mb=" << (reservoir_bytes / (1024 * 1024));
    }

    const bool enable_cross_leaf_union = cross_leaf_union_enabled(config_);
    const bool enable_boundary_leader_bridge = boundary_leader_bridge_enabled();
    LOG_KNOWHERE_INFO_ << "[PiPNN Graph Probe] cross_leaf_union_enabled="
                       << (enable_cross_leaf_union ? "true" : "false")
                       << ", sibling_budget=" << (enable_cross_leaf_union ? cross_leaf_budget(config_) : 0)
                       << ", boundary_leader_bridge_enabled="
                       << (enable_boundary_leader_bridge ? "true" : "false")
                       << ", boundary_bridge_budget="
                       << (enable_boundary_leader_bridge ? boundary_leader_bridge_budget(config_) : 0);

    const auto leaf_process_start = clock::now();
#ifdef _OPENMP
    const int num_threads =
        config_.num_threads == 0 ? omp_get_max_threads() : static_cast<int>(config_.num_threads);
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int64_t i = 0; i < static_cast<int64_t>(leaves.size()); ++i) {
        process_leaf(data, dim, leaves[static_cast<size_t>(i)], context);
    }
#else
    for (const auto& leaf : leaves) {
        process_leaf(data, dim, leaf, context);
    }
#endif
    const auto leaf_process_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - leaf_process_start).count();
    const double leaf_avg_ms =
        leaves.empty() ? 0.0 : ns_to_ms(leaf_process_ns) / static_cast<double>(leaves.size());
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(leaf_process_ns)
                       << " ms (Parallel leaf processing, avg_per_leaf_ms=" << leaf_avg_ms
                       << ", leaves=" << leaves.size()
                       << ", accumulated_leaf_cpu_ms=" << ns_to_ms(context.leaf_total_ns.load()) << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.gemm_total_ns.load())
                       << " ms (GEMM distance matrix computation, leaves=" << leaves.size() << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.hash_prune_total_ns.load())
                       << " ms (HashPrune insertion, leaves=" << leaves.size() << ")";

    // Extract adjacency from global reservoirs.
    {
        const auto extract_start = clock::now();
        for (uint32_t i = 0; i < n; ++i) {
            context.adjacency[i] = context.global_reservoirs[i].neighbors();
        }
        const auto extract_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - extract_start).count();
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(extract_ns)
                           << " ms (Adjacency extraction from global reservoirs, num_points=" << n << ")";
    }

    if (n > 1) {
        for (uint32_t i = 0; i < n; ++i) {
            if (!context.adjacency[i].empty()) {
                continue;
            }

            uint32_t best_j = i == 0 ? 1 : 0;
            float best_dist = std::numeric_limits<float>::max();
            const float* point_i = data + static_cast<size_t>(i) * dim;

            for (uint32_t j = 0; j < n; ++j) {
                if (j == i) {
                    continue;
                }
                const float dist = l2sq(point_i, data + static_cast<size_t>(j) * dim, dim);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_j = j;
                }
            }

            const auto edge_insert_start = clock::now();
            EdgeLockStats lock_stats;
            insert_bidirectional_edge(
                i, best_j, context.adjacency, context.node_locks.get(), build_degree_limit, &lock_stats);
            const auto edge_insert_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - edge_insert_start).count();
            context.edge_insert_ns.fetch_add(edge_insert_ns, std::memory_order_relaxed);
            context.edge_insert_count.fetch_add(1, std::memory_order_relaxed);
            context.edge_lock_wait_ns.fetch_add(lock_stats.wait_ns, std::memory_order_relaxed);
            if (lock_stats.contended) {
                context.edge_lock_contention_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    collect_pre_final_retained_probe_stats(data, n, dim, context);
    if (config_.final_prune) {
        const auto robust_prune_start = clock::now();
        robust_prune_pass(data, n, dim, context.adjacency, context);
        const auto robust_prune_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - robust_prune_start).count();
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(robust_prune_ns)
                           << " ms (Robust prune pass, num_points=" << n << ")";
    }
    log_quality_probe_stats(context, config_.final_prune);
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.edge_insert_ns.load())
                       << " ms (Edge insertion, count=" << context.edge_insert_count.load() << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.edge_lock_wait_ns.load())
                       << " ms (Edge lock wait, lock_impl=spinlock, contended_inserts="
                       << context.edge_lock_contention_count.load() << ", lock_bytes="
                       << (static_cast<uint64_t>(sizeof(SpinMutex)) * n) << ")";

    return context.adjacency;
}

PiPNNBuilder::RbcCoverageStats
PiPNNBuilder::collect_rbc_coverage_stats(const std::vector<Leaf>& leaves, uint32_t n) {
    RbcCoverageStats stats;
    stats.leaves = static_cast<uint32_t>(leaves.size());
    if (leaves.empty() || n == 0) {
        return stats;
    }

    std::vector<uint32_t> membership(n, 0);
    stats.min_leaf_size = std::numeric_limits<uint32_t>::max();
    for (const auto& leaf : leaves) {
        const uint32_t leaf_size = static_cast<uint32_t>(leaf.point_ids.size());
        stats.min_leaf_size = std::min(stats.min_leaf_size, leaf_size);
        stats.max_leaf_size = std::max(stats.max_leaf_size, leaf_size);
        stats.avg_leaf_size += static_cast<double>(leaf_size);
        for (uint32_t id : leaf.point_ids) {
            if (id < n) {
                ++membership[id];
            }
        }
    }
    stats.avg_leaf_size /= static_cast<double>(leaves.size());
    stats.min_membership = std::numeric_limits<uint32_t>::max();
    for (uint32_t count : membership) {
        stats.avg_membership += static_cast<double>(count);
        stats.min_membership = std::min(stats.min_membership, count);
        stats.max_membership = std::max(stats.max_membership, count);
        stats.single_membership_nodes += count == 1 ? 1U : 0U;
    }
    stats.avg_membership /= static_cast<double>(n);
    return stats;
}

void
PiPNNBuilder::collect_pre_final_retained_probe_stats(const float* data, uint32_t n, uint32_t dim,
                                                     BuildContext& context) const {
    for (uint32_t i = 0; i < n; ++i) {
        if (!should_probe_node(i)) {
            continue;
        }

        const auto& neighbors = context.adjacency[i];
        context.quality_probe.pre_final_retained_degree.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                                  std::memory_order_relaxed);

        std::vector<std::pair<float, uint32_t>> exact_topk;
        exact_topk.reserve(config_.k_nn);
        const float* query = data + static_cast<size_t>(i) * dim;
        std::vector<std::pair<float, uint32_t>> global_candidates;
        global_candidates.reserve(n > 0 ? n - 1 : 0);
        for (uint32_t other = 0; other < n; ++other) {
            if (other == i) {
                continue;
            }
            global_candidates.emplace_back(l2sq(query, data + static_cast<size_t>(other) * dim, dim), other);
        }
        const uint32_t truth_k = std::min<uint32_t>(config_.k_nn, static_cast<uint32_t>(global_candidates.size()));
        if (global_candidates.size() > truth_k) {
            std::nth_element(global_candidates.begin(), global_candidates.begin() + truth_k, global_candidates.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            global_candidates.resize(truth_k);
        }
        std::sort(global_candidates.begin(), global_candidates.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        exact_topk = std::move(global_candidates);
        context.quality_probe.pre_final_retained_total.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                                 std::memory_order_relaxed);
        context.quality_probe.pre_final_retained_overlap_hits.fetch_add(
            static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, neighbors)), std::memory_order_relaxed);
    }
}

void
PiPNNBuilder::log_quality_probe_stats(const BuildContext& context, bool final_prune_enabled) {
    const double sampled_nodes = static_cast<double>(context.quality_probe.sampled_nodes.load());
    if (sampled_nodes == 0.0) {
        LOG_KNOWHERE_INFO_ << "[PiPNN Graph Probe] sampled_nodes=0";
        return;
    }

    const double candidate_total = static_cast<double>(context.quality_probe.candidate_total.load());
    const double hash_total = static_cast<double>(context.quality_probe.hash_total.load());
    const double inserted_total = static_cast<double>(context.quality_probe.inserted_total.load());
    const double pre_final_total = static_cast<double>(context.quality_probe.pre_final_retained_total.load());
    const double final_total = static_cast<double>(context.quality_probe.final_prune_total.load());
    LOG_KNOWHERE_INFO_ << "[PiPNN Graph Probe] sampled_nodes=" << sampled_nodes
                       << ", candidate_exact_topk_overlap="
                       << (candidate_total == 0.0 ? 0.0
                                                  : static_cast<double>(context.quality_probe.candidate_overlap_hits.load()) /
                                                        candidate_total)
                       << ", hash_exact_topk_overlap="
                       << (hash_total == 0.0 ? 0.0
                                             : static_cast<double>(context.quality_probe.hash_overlap_hits.load()) /
                                                   hash_total)
                       << ", inserted_exact_topk_overlap="
                       << (inserted_total == 0.0 ? 0.0
                                                 : static_cast<double>(context.quality_probe.inserted_overlap_hits.load()) /
                                                       inserted_total)
                       << ", pre_final_retained_exact_topk_overlap="
                       << (pre_final_total == 0.0 ? 0.0
                                                  : static_cast<double>(context.quality_probe.pre_final_retained_overlap_hits.load()) /
                                                        pre_final_total)
                       << ", final_prune_exact_topk_overlap="
                       << (final_total == 0.0 ? 0.0
                                              : static_cast<double>(context.quality_probe.final_prune_overlap_hits.load()) /
                                                    final_total)
                       << ", hash_collision_replace=" << context.quality_probe.collision_replace.load()
                       << ", hash_collision_reject=" << context.quality_probe.collision_reject.load()
                       << ", hash_append=" << context.quality_probe.append_accept.load()
                       << ", hash_reservoir_replace=" << context.quality_probe.reservoir_replace.load()
                       << ", hash_reservoir_reject=" << context.quality_probe.reservoir_reject.load()
                       << ", insert_append=" << context.quality_probe.insert_append.load()
                       << ", insert_duplicate=" << context.quality_probe.insert_duplicate.load()
                       << ", insert_degree_full=" << context.quality_probe.insert_degree_full.load()
                       << ", pre_final_retained_avg_degree="
                       << (sampled_nodes == 0.0 ? 0.0
                                                : static_cast<double>(context.quality_probe.pre_final_retained_degree.load()) /
                                                      sampled_nodes)
                       << ", final_prune_enabled=" << (final_prune_enabled ? "true" : "false")
                       << ", final_prune_avg_degree_before="
                       << (sampled_nodes == 0.0 ? 0.0
                                                : static_cast<double>(context.quality_probe.final_prune_degree_before.load()) /
                                                      sampled_nodes)
                       << ", final_prune_avg_degree_after="
                       << (sampled_nodes == 0.0 ? 0.0
                                                : static_cast<double>(context.quality_probe.final_prune_degree_after.load()) /
                                                      sampled_nodes);
}

void
PiPNNBuilder::process_leaf(const float* data, uint32_t dim, const Leaf& leaf, BuildContext& context) const {
    if (leaf.point_ids.size() < 2) {
        return;
    }

    using clock = std::chrono::steady_clock;
    const auto leaf_start = clock::now();
    const uint32_t leaf_size = static_cast<uint32_t>(leaf.point_ids.size());
    RowMajorMatrixXf x(leaf_size, dim);
    for (uint32_t local_id = 0; local_id < leaf_size; ++local_id) {
        const uint32_t global_id = leaf.point_ids[local_id];
        const float* src = data + static_cast<size_t>(global_id) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            x(local_id, d) = src[d];
        }
    }

    const auto gemm_start = clock::now();
    Eigen::VectorXf norms = x.rowwise().squaredNorm();
    Eigen::MatrixXf d_mat = -2.0f * x * x.transpose();
    d_mat.colwise() += norms;
    d_mat.rowwise() += norms.transpose();
    for (uint32_t i = 0; i < leaf_size; ++i) {
        d_mat(i, i) = std::numeric_limits<float>::max();
    }
    const auto gemm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - gemm_start).count();
    context.gemm_total_ns.fetch_add(gemm_ns, std::memory_order_relaxed);

    const bool enable_cross_leaf_union = cross_leaf_union_enabled(config_);

    for (uint32_t i = 0; i < leaf_size; ++i) {
        const uint32_t global_i = leaf.point_ids[i];
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(leaf_size - 1);
        for (uint32_t j = 0; j < leaf_size; ++j) {
            if (i == j) {
                continue;
            }
            candidates.emplace_back(d_mat(i, j), leaf.point_ids[j]);
        }

        const uint32_t local_target_k = std::min<uint32_t>(candidate_budget_for_leaf(config_, leaf_size),
                                                           static_cast<uint32_t>(candidates.size()));
        if (local_target_k > 0 && candidates.size() > local_target_k) {
            std::nth_element(candidates.begin(),
                             candidates.begin() + local_target_k,
                             candidates.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            candidates.resize(local_target_k);
        }

        const auto& shared_leaf_ids = context.point_to_leaf_ids[global_i];
        const uint32_t sibling_budget = enable_cross_leaf_union ? cross_leaf_budget(config_) : 0;
        const uint32_t bridge_budget = boundary_leader_bridge_enabled() ? boundary_leader_bridge_budget(config_) : 0;
        const float* query = data + static_cast<size_t>(global_i) * dim;
        if (sibling_budget > 0) {
            for (uint32_t shared_leaf_id : shared_leaf_ids) {
                const auto& sibling_leaf = (*context.leaves)[shared_leaf_id];
                if (&sibling_leaf == &leaf || sibling_leaf.point_ids.size() <= 1) {
                    continue;
                }

                std::vector<std::pair<float, uint32_t>> sibling_candidates;
                sibling_candidates.reserve(sibling_leaf.point_ids.size() - 1);
                for (uint32_t sibling_point : sibling_leaf.point_ids) {
                    if (sibling_point == global_i) {
                        continue;
                    }
                    sibling_candidates.emplace_back(
                        l2sq(query, data + static_cast<size_t>(sibling_point) * dim, dim), sibling_point);
                }
                const uint32_t take = std::min<uint32_t>(sibling_budget, static_cast<uint32_t>(sibling_candidates.size()));
                if (take == 0) {
                    continue;
                }
                if (sibling_candidates.size() > take) {
                    std::nth_element(sibling_candidates.begin(),
                                     sibling_candidates.begin() + take,
                                     sibling_candidates.end(),
                                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                    sibling_candidates.resize(take);
                }
                candidates.insert(candidates.end(), sibling_candidates.begin(), sibling_candidates.end());
            }
        }

        if (bridge_budget > 0) {
            const uint32_t local_bridge_seed = std::min<uint32_t>(bridge_budget, static_cast<uint32_t>(candidates.size()));
            for (uint32_t seed_idx = 0; seed_idx < local_bridge_seed; ++seed_idx) {
                const uint32_t bridge_seed = candidates[seed_idx].second;
                for (uint32_t bridge_leaf_id : context.point_to_leaf_ids[bridge_seed]) {
                    const auto& bridge_leaf = (*context.leaves)[bridge_leaf_id];
                    if (&bridge_leaf == &leaf || bridge_leaf.point_ids.size() <= 1) {
                        continue;
                    }

                    uint32_t best_bridge_point = global_i;
                    float best_bridge_dist = std::numeric_limits<float>::max();
                    for (uint32_t bridge_point : bridge_leaf.point_ids) {
                        if (bridge_point == global_i || bridge_point == bridge_seed) {
                            continue;
                        }
                        const float dist = l2sq(query, data + static_cast<size_t>(bridge_point) * dim, dim);
                        if (dist < best_bridge_dist) {
                            best_bridge_dist = dist;
                            best_bridge_point = bridge_point;
                        }
                    }
                    if (best_bridge_point != global_i) {
                        candidates.emplace_back(best_bridge_dist, best_bridge_point);
                    }
                }
            }
        }

        if (candidates.empty()) {
            continue;
        }

        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
        candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
                             return lhs.second == rhs.second;
                         }),
                         candidates.end());

        const uint32_t target_k = std::min<uint32_t>(std::max<uint32_t>(local_target_k, sibling_budget),
                                                     static_cast<uint32_t>(candidates.size()));
        if (target_k == 0) {
            continue;
        }
        if (candidates.size() > target_k) {
            candidates.resize(target_k);
        }

        const bool probe_node = should_probe_node(global_i);
        std::vector<std::pair<float, uint32_t>> exact_topk;
        if (probe_node) {
            exact_topk.reserve(target_k);
            const float* query = data + static_cast<size_t>(global_i) * dim;
            std::vector<std::pair<float, uint32_t>> global_candidates;
            const uint32_t total_points = static_cast<uint32_t>(context.adjacency.size());
            global_candidates.reserve(total_points > 0 ? total_points - 1 : 0);
            for (uint32_t other = 0; other < total_points; ++other) {
                if (other == global_i) {
                    continue;
                }
                global_candidates.emplace_back(l2sq(query, data + static_cast<size_t>(other) * dim, dim), other);
            }
            if (global_candidates.size() > target_k) {
                std::nth_element(global_candidates.begin(),
                                 global_candidates.begin() + target_k,
                                 global_candidates.end(),
                                 [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                global_candidates.resize(target_k);
            }
            std::sort(global_candidates.begin(), global_candidates.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.first < rhs.first;
            });
            exact_topk = std::move(global_candidates);

            std::vector<uint32_t> candidate_ids;
            candidate_ids.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                candidate_ids.push_back(candidate.second);
            }
            context.quality_probe.sampled_nodes.fetch_add(1, std::memory_order_relaxed);
            context.quality_probe.candidate_total.fetch_add(static_cast<int64_t>(target_k), std::memory_order_relaxed);
            context.quality_probe.candidate_overlap_hits.fetch_add(
                static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, candidate_ids)), std::memory_order_relaxed);
        }

        // Stream candidates into global_reservoirs[global_i] under per-point lock.
        // Bidirectionality emerges naturally: when point j is processed as point i
        // in another iteration of this same loop, it also streams its candidates
        // (including i) into global_reservoirs[j].
        const float* sketch_i = context.global_sketches_flat.data() +
                                static_cast<size_t>(global_i) * config_.hash_bits;
        const auto hash_prune_insert_start = clock::now();
        for (const auto& candidate : candidates) {
            const uint32_t global_j = candidate.second;
            const float dist = std::max(0.0f, candidate.first);
            const float* sketch_j = context.global_sketches_flat.data() +
                                    static_cast<size_t>(global_j) * config_.hash_bits;
            const uint16_t h = HashPrune::residual_hash(sketch_i, sketch_j, config_.hash_bits);

            std::lock_guard<SpinMutex> lock(context.node_locks[global_i]);
            context.global_reservoirs[global_i].insert(global_j, h, dist);
        }
        const auto hash_prune_insert_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - hash_prune_insert_start).count();
        context.hash_prune_total_ns.fetch_add(hash_prune_insert_ns, std::memory_order_relaxed);

        if (probe_node) {
            // Count how many candidates were accepted into the global reservoir so far
            // (approximate — read under no lock, benign for diagnostics)
            const auto current_neighbors = context.global_reservoirs[global_i].neighbors();
            context.quality_probe.inserted_total.fetch_add(
                static_cast<int64_t>(current_neighbors.size()), std::memory_order_relaxed);
            if (!exact_topk.empty()) {
                context.quality_probe.inserted_overlap_hits.fetch_add(
                    static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, current_neighbors)),
                    std::memory_order_relaxed);
            }
        }
    }
    const auto leaf_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - leaf_start).count();
    context.leaf_total_ns.fetch_add(leaf_ns, std::memory_order_relaxed);

    const uint32_t processed_leaves = context.processed_leaves.fetch_add(1, std::memory_order_relaxed) + 1;
    if (processed_leaves <= 4 || processed_leaves % kLeafProgressLogInterval == 0) {
        const auto leaf_elapsed_ms = static_cast<double>(leaf_ns) / 1e6;
        const auto accumulated_leaf_cpu_ms = static_cast<double>(context.leaf_total_ns.load()) / 1e6;
        const auto accumulated_gemm_ms = static_cast<double>(context.gemm_total_ns.load()) / 1e6;
        const auto accumulated_hash_prune_ms = static_cast<double>(context.hash_prune_total_ns.load()) / 1e6;
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Leaf progress processed=" << processed_leaves
                           << "/" << context.rbc_coverage.leaves
                           << ", leaf_size=" << leaf_size
                           << ", leaf_elapsed_ms=" << leaf_elapsed_ms
                           << ", accumulated_leaf_cpu_ms=" << accumulated_leaf_cpu_ms
                           << ", accumulated_gemm_ms=" << accumulated_gemm_ms
                           << ", accumulated_hash_prune_ms=" << accumulated_hash_prune_ms
                           << ", edge_insert_count=" << context.edge_insert_count.load();
    }
}

void
PiPNNBuilder::robust_prune_pass(const float* data, uint32_t n, uint32_t dim,
                                std::vector<std::vector<uint32_t>>& adjacency, BuildContext& context) const {
    for (uint32_t i = 0; i < n; ++i) {
        auto& neighbors = adjacency[i];
        dedup_and_drop_self(i, neighbors);
        const auto degree_before = neighbors.size();

        if (should_probe_node(i)) {
            context.quality_probe.final_prune_degree_before.fetch_add(static_cast<int64_t>(degree_before),
                                                                     std::memory_order_relaxed);
        }

        if (neighbors.size() <= config_.max_degree) {
            if (should_probe_node(i)) {
                context.quality_probe.final_prune_degree_after.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                                        std::memory_order_relaxed);
                std::vector<std::pair<float, uint32_t>> exact_topk;
                exact_topk.reserve(config_.max_degree);
                const float* query = data + static_cast<size_t>(i) * dim;
                std::vector<std::pair<float, uint32_t>> global_candidates;
                global_candidates.reserve(n > 0 ? n - 1 : 0);
                for (uint32_t other = 0; other < n; ++other) {
                    if (other == i) {
                        continue;
                    }
                    global_candidates.emplace_back(l2sq(query, data + static_cast<size_t>(other) * dim, dim), other);
                }
                const uint32_t truth_k = std::min<uint32_t>(config_.k_nn, static_cast<uint32_t>(global_candidates.size()));
                if (global_candidates.size() > truth_k) {
                    std::nth_element(global_candidates.begin(), global_candidates.begin() + truth_k, global_candidates.end(),
                                     [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                    global_candidates.resize(truth_k);
                }
                std::sort(global_candidates.begin(), global_candidates.end(),
                          [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                exact_topk = std::move(global_candidates);
                context.quality_probe.final_prune_total.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                                  std::memory_order_relaxed);
                context.quality_probe.final_prune_overlap_hits.fetch_add(
                    static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, neighbors)), std::memory_order_relaxed);
            }
            continue;
        }

        HashPrune prune(dim, config_.hash_bits, config_.max_degree);
        std::vector<float> sketch_i(config_.hash_bits);
        prune.compute_sketch(data + static_cast<size_t>(i) * dim, sketch_i.data());

        for (uint32_t neighbor : neighbors) {
            std::vector<float> sketch_j(config_.hash_bits);
            prune.compute_sketch(data + static_cast<size_t>(neighbor) * dim, sketch_j.data());
            const float dist = l2sq(data + static_cast<size_t>(i) * dim, data + static_cast<size_t>(neighbor) * dim, dim);
            prune.insert(neighbor, sketch_i.data(), sketch_j.data(), dist);
        }
        neighbors = prune.neighbors();

        if (should_probe_node(i)) {
            context.quality_probe.final_prune_degree_after.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                                    std::memory_order_relaxed);
            std::vector<std::pair<float, uint32_t>> exact_topk;
            exact_topk.reserve(config_.max_degree);
            const float* query = data + static_cast<size_t>(i) * dim;
            std::vector<std::pair<float, uint32_t>> global_candidates;
            global_candidates.reserve(n > 0 ? n - 1 : 0);
            for (uint32_t other = 0; other < n; ++other) {
                if (other == i) {
                    continue;
                }
                global_candidates.emplace_back(l2sq(query, data + static_cast<size_t>(other) * dim, dim), other);
            }
            const uint32_t truth_k = std::min<uint32_t>(config_.k_nn, static_cast<uint32_t>(global_candidates.size()));
            if (global_candidates.size() > truth_k) {
                std::nth_element(global_candidates.begin(), global_candidates.begin() + truth_k, global_candidates.end(),
                                 [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
                global_candidates.resize(truth_k);
            }
            std::sort(global_candidates.begin(), global_candidates.end(),
                      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            exact_topk = std::move(global_candidates);
            context.quality_probe.final_prune_total.fetch_add(static_cast<int64_t>(neighbors.size()),
                                                              std::memory_order_relaxed);
            context.quality_probe.final_prune_overlap_hits.fetch_add(
                static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, neighbors)), std::memory_order_relaxed);
        }
    }
}

float
PiPNNBuilder::l2sq(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

}  // namespace knowhere::pipnn_diskann
