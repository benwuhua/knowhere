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

#include "index/diskann/impl/rbc_partition.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <utility>

namespace knowhere::pipnn_diskann {

RBCPartitioner::RBCPartitioner()
    : config_{} {}

RBCPartitioner::RBCPartitioner(const Config& config)
    : config_(config) {}

std::vector<Leaf>
RBCPartitioner::partition(const float* data, uint32_t n, uint32_t dim) const {
    std::vector<Leaf> leaves;
    if (data == nullptr || n == 0 || dim == 0) {
        return leaves;
    }

    std::vector<uint32_t> point_ids(n);
    std::iota(point_ids.begin(), point_ids.end(), 0);
    partition_recursive(data, dim, point_ids, 0, leaves);
    return leaves;
}

void
RBCPartitioner::partition_recursive(const float* data, uint32_t dim, const std::vector<uint32_t>& point_ids, size_t depth,
                                    std::vector<Leaf>& leaves) const {
    if (point_ids.empty()) {
        return;
    }

    if (point_ids.size() <= config_.leaf_max_size) {
        leaves.push_back(Leaf{point_ids});
        return;
    }

    const uint32_t fanout = std::min<uint32_t>(fanout_for_depth(depth), static_cast<uint32_t>(point_ids.size()));
    if (fanout <= 1) {
        leaves.push_back(Leaf{point_ids});
        return;
    }

    uint32_t leaders_to_sample = std::max<uint32_t>(fanout, effective_num_leaders(point_ids.size()));
    if (point_ids.size() > 1) {
        leaders_to_sample =
            std::min<uint32_t>(leaders_to_sample, static_cast<uint32_t>(point_ids.size() - 1));
    }
    // Use all effective_num_leaders leaders as partition centers (RBC guarantee requires
    // ~√n leaders so that true neighbors co-occur in the same ball with high probability).
    // Do NOT truncate to fanout — the fanout parameter controls the split_evenly fallback only.
    auto leaders = sample_leaders(point_ids, leaders_to_sample, config_.base_seed + static_cast<uint32_t>(depth));
    if (leaders.empty()) {
        leaves.push_back(Leaf{point_ids});
        return;
    }

    const uint32_t k = std::max<uint32_t>(1, std::min<uint32_t>(config_.overlap_k, static_cast<uint32_t>(leaders.size())));
    auto buckets = assign_to_k_leaders(data, dim, point_ids, leaders, k);

    bool has_progress = false;
    for (const auto& bucket : buckets) {
        if (!bucket.empty() && bucket.size() < point_ids.size()) {
            has_progress = true;
            break;
        }
    }
    if (!has_progress) {
        buckets = split_evenly(point_ids, fanout);
    }

    for (const auto& bucket : buckets) {
        if (!bucket.empty()) {
            partition_recursive(data, dim, bucket, depth + 1, leaves);
        }
    }
}

std::vector<uint32_t>
RBCPartitioner::sample_leaders(const std::vector<uint32_t>& point_ids, uint32_t num_leaders, uint32_t seed) const {
    if (point_ids.empty() || num_leaders == 0) {
        return {};
    }

    std::vector<uint32_t> sampled = point_ids;
    std::mt19937 rng(seed);
    std::shuffle(sampled.begin(), sampled.end(), rng);

    const size_t keep = std::min<size_t>(num_leaders, sampled.size());
    sampled.resize(keep);
    return sampled;
}

std::vector<std::vector<uint32_t>>
RBCPartitioner::assign_to_k_leaders(const float* data, uint32_t dim, const std::vector<uint32_t>& point_ids,
                                    const std::vector<uint32_t>& leaders, uint32_t k) const {
    std::vector<std::vector<uint32_t>> buckets(leaders.size());
    if (leaders.empty() || k == 0) {
        return buckets;
    }

    const uint32_t effective_k = std::min<uint32_t>(k, static_cast<uint32_t>(leaders.size()));

    for (auto point_id : point_ids) {
        const float* point = data + static_cast<size_t>(point_id) * dim;

        std::vector<std::pair<float, uint32_t>> dist_to_leaders;
        dist_to_leaders.reserve(leaders.size());

        for (size_t leader_idx = 0; leader_idx < leaders.size(); ++leader_idx) {
            const float* leader = data + static_cast<size_t>(leaders[leader_idx]) * dim;
            dist_to_leaders.emplace_back(l2sq(point, leader, dim), static_cast<uint32_t>(leader_idx));
        }

        if (dist_to_leaders.size() > effective_k) {
            std::nth_element(dist_to_leaders.begin(), dist_to_leaders.begin() + effective_k, dist_to_leaders.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
            dist_to_leaders.resize(effective_k);
        }

        for (const auto& pair : dist_to_leaders) {
            buckets[pair.second].push_back(point_id);
        }
    }

    return buckets;
}

std::vector<std::vector<uint32_t>>
RBCPartitioner::split_evenly(const std::vector<uint32_t>& point_ids, uint32_t fanout) const {
    std::vector<std::vector<uint32_t>> buckets(fanout);
    if (fanout == 0) {
        return buckets;
    }

    for (size_t i = 0; i < point_ids.size(); ++i) {
        buckets[i % fanout].push_back(point_ids[i]);
    }

    return buckets;
}

uint32_t
RBCPartitioner::fanout_for_depth(size_t depth) const {
    if (depth == 0) {
        return std::max<uint32_t>(1, config_.fanout_l1);
    }
    if (depth == 1) {
        return std::max<uint32_t>(1, config_.fanout_l2);
    }
    return std::max<uint32_t>(1, config_.fanout_rest);
}

uint32_t
RBCPartitioner::effective_num_leaders(size_t subset_size) const {
    if (subset_size == 0) {
        return 0;
    }
    if (subset_size == 1) {
        return 1;
    }

    if (config_.num_leaders > 0) {
        return std::min<uint32_t>(config_.num_leaders, static_cast<uint32_t>(subset_size - 1));
    }

    const auto by_sqrt = static_cast<uint32_t>(std::sqrt(static_cast<double>(subset_size)));
    const uint32_t default_leaders = std::max<uint32_t>(1, std::min<uint32_t>(by_sqrt, 1000));
    return std::min<uint32_t>(default_leaders, static_cast<uint32_t>(subset_size - 1));
}

float
RBCPartitioner::l2sq(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

}  // namespace knowhere::pipnn_diskann
