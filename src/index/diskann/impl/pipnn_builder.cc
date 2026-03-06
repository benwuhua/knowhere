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
#include <limits>
#include <numeric>
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

void
insert_neighbor(std::vector<uint32_t>& neighbors, uint32_t dst, uint32_t max_degree) {
    if (std::find(neighbors.begin(), neighbors.end(), dst) != neighbors.end()) {
        return;
    }
    if (neighbors.size() < max_degree) {
        neighbors.push_back(dst);
    }
}

void
insert_bidirectional_edge(uint32_t u, uint32_t v, std::vector<std::vector<uint32_t>>& adjacency,
                          SpinMutex* node_locks, uint32_t max_degree,
                          EdgeLockStats* lock_stats = nullptr) {
    if (u == v) {
        return;
    }

    EdgeLockStats local_stats;
    EdgeLockStats& stats = lock_stats == nullptr ? local_stats : *lock_stats;
    const uint32_t first_id = std::min(u, v);
    const uint32_t second_id = std::max(u, v);
    NodeLockGuard guard(node_locks[first_id], &node_locks[second_id], stats);
    insert_neighbor(adjacency[u], v, max_degree);
    insert_neighbor(adjacency[v], u, max_degree);
}

void
dedup_and_drop_self(uint32_t self_id, std::vector<uint32_t>& neighbors) {
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), self_id), neighbors.end());
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
}

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
    for (auto& neighbors : context.adjacency) {
        neighbors.reserve(config_.max_degree);
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
    const auto rbc_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - rbc_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(rbc_ns)
                       << " ms (RBC partition, leaves=" << leaves.size() << ", num_points=" << n << ")";

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
                i, best_j, context.adjacency, context.node_locks.get(), config_.max_degree, &lock_stats);
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

    if (config_.final_prune) {
        const auto robust_prune_start = clock::now();
        robust_prune_pass(data, n, dim, context.adjacency);
        const auto robust_prune_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - robust_prune_start).count();
        LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(robust_prune_ns)
                           << " ms (Robust prune pass, num_points=" << n << ")";
    }
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.edge_insert_ns.load())
                       << " ms (Edge insertion, count=" << context.edge_insert_count.load() << ")";
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(context.edge_lock_wait_ns.load())
                       << " ms (Edge lock wait, lock_impl=spinlock, contended_inserts="
                       << context.edge_lock_contention_count.load() << ", lock_bytes="
                       << (static_cast<uint64_t>(sizeof(SpinMutex)) * n) << ")";

    return context.adjacency;
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

    HashPrune sketcher(dim, config_.hash_bits, config_.max_degree);
    std::vector<std::vector<float>> sketches(leaf_size, std::vector<float>(config_.hash_bits));
    for (uint32_t local_id = 0; local_id < leaf_size; ++local_id) {
        const uint32_t global_id = leaf.point_ids[local_id];
        const float* vec = data + static_cast<size_t>(global_id) * dim;
        sketcher.compute_sketch(vec, sketches[local_id].data());
    }

    for (uint32_t i = 0; i < leaf_size; ++i) {
        std::vector<std::pair<float, uint32_t>> candidates;
        candidates.reserve(leaf_size - 1);
        for (uint32_t j = 0; j < leaf_size; ++j) {
            if (i == j) {
                continue;
            }
            candidates.emplace_back(d_mat(i, j), j);
        }

        const uint32_t target_k = std::min<uint32_t>(config_.k_nn, static_cast<uint32_t>(candidates.size()));
        if (target_k == 0) {
            continue;
        }

        if (candidates.size() > target_k) {
            std::nth_element(candidates.begin(),
                             candidates.begin() + target_k,
                             candidates.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            candidates.resize(target_k);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        HashPrune prune(dim, config_.hash_bits, config_.max_degree);
        const auto hash_prune_insert_start = clock::now();
        for (const auto& candidate : candidates) {
            const uint32_t local_j = candidate.second;
            const uint32_t global_j = leaf.point_ids[local_j];
            const float dist = std::max(0.0f, candidate.first);
            prune.insert(global_j, sketches[i].data(), sketches[local_j].data(), dist);
        }
        const auto hash_prune_insert_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - hash_prune_insert_start).count();
        context.hash_prune_total_ns.fetch_add(hash_prune_insert_ns, std::memory_order_relaxed);

        const uint32_t global_i = leaf.point_ids[i];
        for (uint32_t neighbor : prune.neighbors()) {
            const auto edge_insert_start = clock::now();
            EdgeLockStats lock_stats;
            insert_bidirectional_edge(global_i,
                                      neighbor,
                                      context.adjacency,
                                      context.node_locks.get(),
                                      config_.max_degree,
                                      &lock_stats);
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
    const auto leaf_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - leaf_start).count();
    context.leaf_total_ns.fetch_add(leaf_ns, std::memory_order_relaxed);
}

void
PiPNNBuilder::robust_prune_pass(const float* data, uint32_t n, uint32_t dim,
                                std::vector<std::vector<uint32_t>>& adjacency) const {
    for (uint32_t i = 0; i < n; ++i) {
        auto& neighbors = adjacency[i];
        dedup_and_drop_self(i, neighbors);

        if (neighbors.size() <= config_.max_degree) {
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
