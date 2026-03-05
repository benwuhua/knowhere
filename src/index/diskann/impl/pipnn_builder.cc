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
#include <limits>
#include <mutex>
#include <numeric>
#include <vector>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace knowhere::pipnn_diskann {
namespace {

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

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
                          std::vector<std::mutex>& node_mutexes, uint32_t max_degree) {
    if (u == v) {
        return;
    }

    if (u < v) {
        std::scoped_lock guard(node_mutexes[u], node_mutexes[v]);
        insert_neighbor(adjacency[u], v, max_degree);
        insert_neighbor(adjacency[v], u, max_degree);
    } else {
        std::scoped_lock guard(node_mutexes[v], node_mutexes[u]);
        insert_neighbor(adjacency[u], v, max_degree);
        insert_neighbor(adjacency[v], u, max_degree);
    }
}

void
dedup_and_drop_self(uint32_t self_id, std::vector<uint32_t>& neighbors) {
    neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), self_id), neighbors.end());
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
}

}  // namespace

struct PiPNNBuilder::BuildContext {
    explicit BuildContext(uint32_t n)
        : adjacency(n), node_mutexes(n) {
    }

    std::vector<std::vector<uint32_t>> adjacency;
    std::vector<std::mutex> node_mutexes;
};

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

    const auto leaves = partitioner.partition(data, n, dim);

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

            insert_bidirectional_edge(i, best_j, context.adjacency, context.node_mutexes, config_.max_degree);
        }
    }

    if (config_.final_prune) {
        robust_prune_pass(data, n, dim, context.adjacency);
    }

    return context.adjacency;
}

void
PiPNNBuilder::process_leaf(const float* data, uint32_t dim, const Leaf& leaf, BuildContext& context) const {
    if (leaf.point_ids.size() < 2) {
        return;
    }

    const uint32_t leaf_size = static_cast<uint32_t>(leaf.point_ids.size());
    RowMajorMatrixXf x(leaf_size, dim);
    for (uint32_t local_id = 0; local_id < leaf_size; ++local_id) {
        const uint32_t global_id = leaf.point_ids[local_id];
        const float* src = data + static_cast<size_t>(global_id) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            x(local_id, d) = src[d];
        }
    }

    Eigen::VectorXf norms = x.rowwise().squaredNorm();
    Eigen::MatrixXf d_mat = -2.0f * x * x.transpose();
    d_mat.colwise() += norms;
    d_mat.rowwise() += norms.transpose();
    for (uint32_t i = 0; i < leaf_size; ++i) {
        d_mat(i, i) = std::numeric_limits<float>::max();
    }

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
        for (const auto& candidate : candidates) {
            const uint32_t local_j = candidate.second;
            const uint32_t global_j = leaf.point_ids[local_j];
            const float dist = std::max(0.0f, candidate.first);
            prune.insert(global_j, sketches[i].data(), sketches[local_j].data(), dist);
        }

        const uint32_t global_i = leaf.point_ids[i];
        for (uint32_t neighbor : prune.neighbors()) {
            insert_bidirectional_edge(global_i, neighbor, context.adjacency, context.node_mutexes, config_.max_degree);
        }
    }
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
