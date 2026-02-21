// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

// standard headers
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>

// Faiss-specific headers
#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/ordered_key_value.h>

// Knowhere-specific headers
#include <faiss/cppcontrib/knowhere/impl/Neighbor.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

namespace {

// whether to track statistics
constexpr bool track_hnsw_stats = true;

} // namespace

// Accomodates all the search logic and variables.
/// * DistanceComputerT is responsible for computing distances
/// * GraphVisitorT records visited edges
/// * VisitedT is responsible for tracking visited nodes
/// * FilterT is resposible for filtering unneeded nodes
/// Interfaces of all templates are tweaked to accept standard Faiss structures
///   with dynamic dispatching. Custom Knowhere structures are also accepted.
template <
        typename DistanceComputerT,
        typename GraphVisitorT,
        typename VisitedT,
        typename FilterT>
struct v2_hnsw_searcher {
    using storage_idx_t = faiss::HNSW::storage_idx_t;
    using idx_t = faiss::idx_t;

    // hnsw structure.
    // the reference is not owned.
    const faiss::HNSW& hnsw;

    // computes distances. it already knows the query vector.
    // the reference is not owned.
    DistanceComputerT& qdis;

    // records visited edges.
    // the reference is not owned.
    GraphVisitorT& graph_visitor;

    // tracks the nodes that have been visited already.
    // the reference is not owned.
    VisitedT& visited_nodes;

    // a filter for disabled nodes.
    // the reference is not owned.
    const FilterT& filter;

    // parameter for the filtering
    const float kAlpha;

    // custom parameters of HNSW search.
    // the pointer is not owned.
    const faiss::SearchParametersHNSW* params;

    //
    v2_hnsw_searcher(
            const faiss::HNSW& hnsw_,
            DistanceComputerT& qdis_,
            GraphVisitorT& graph_visitor_,
            VisitedT& visited_nodes_,
            const FilterT& filter_,
            const float kAlpha_,
            const faiss::SearchParametersHNSW* params_)
            : hnsw{hnsw_},
              qdis{qdis_},
              graph_visitor{graph_visitor_},
              visited_nodes{visited_nodes_},
              filter{filter_},
              kAlpha{kAlpha_},
              params{params_} {}

    v2_hnsw_searcher(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher(v2_hnsw_searcher&&) = delete;
    v2_hnsw_searcher& operator=(const v2_hnsw_searcher&) = delete;
    v2_hnsw_searcher& operator=(v2_hnsw_searcher&&) = delete;

    // greedily update a nearest vector at a given level.
    // * the update starts from the value in 'nearest'.
    faiss::HNSWStats greedy_update_nearest(
            const int level,
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        for (;;) {
            storage_idx_t prev_nearest = nearest;

            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            // prefetch and eval the size
            size_t count = 0;
            for (size_t i = begin; i < end; i++) {
                storage_idx_t v = hnsw.neighbors[i];
                if (v < 0) {
                    break;
                }

                // qdis.prefetch(v);
                count += 1;
            }

            // visit neighbors
            for (size_t i = begin; i < begin + count; i++) {
                storage_idx_t v = hnsw.neighbors[i];

                // compute the distance
                const float dis = qdis(v);

                // record a traversed edge
                graph_visitor.visit_edge(level, prev_nearest, nearest, dis);

                // check if an update is needed
                if (dis < d_nearest) {
                    nearest = v;
                    d_nearest = dis;
                }
            }

            // update stats
            if (track_hnsw_stats) {
                stats.ndis += count;
                stats.nhops += 1;
            }

            // we're done if there we no changes
            if (nearest == prev_nearest) {
                return stats;
            }
        }
    }

    // no loops, just check neighbors of a single node.
    template <typename FuncAddCandidate>
    faiss::HNSWStats evaluate_single_node(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        // // unused
        // bool do_dis_check = params ? params->check_relative_distance
        //                            : hnsw.check_relative_distance;

        faiss::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        // todo: add prefetch
        size_t counter = 0;
        size_t saved_indices[4];
        int saved_statuses[4];

        size_t ndis = 0;
        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                // no more neighbors
                break;
            }

            // already visited?
            if (visited_nodes.get(v1)) {
                // yes, visited.
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            // not visited. mark as visited.
            visited_nodes.set(v1);

            // is the node disabled?
            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                // yes, disabled
                status = knowhere::Neighbor::kInvalid;

                // sometimes, disabled nodes are allowed to be used
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }

                accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            counter += 1;

            ndis += 1;

            if (counter == 4) {
                // evaluate 4x distances at once
                float dis[4] = {0, 0, 0, 0};
                qdis.distances_batch_4(
                        saved_indices[0],
                        saved_indices[1],
                        saved_indices[2],
                        saved_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    // record a traversed edge
                    graph_visitor.visit_edge(
                            level, node_id, saved_indices[id4], dis[id4]);

                    // add a record of visited nodes
                    knowhere::Neighbor nn(
                            saved_indices[id4], dis[id4], saved_statuses[id4]);
                    if (func_add_candidate(nn)) {
#if defined(USE_PREFETCH)
                        // TODO
                        // _mm_prefetch(get_linklist0(v), _MM_HINT_T0);
#endif
                    }
                }

                counter = 0;
            }
        }

        // process leftovers
        for (size_t id4 = 0; id4 < counter; id4++) {
            // evaluate a single distance
            const float dis = qdis(saved_indices[id4]);

            // record a traversed edge
            graph_visitor.visit_edge(level, node_id, saved_indices[id4], dis);

            // add a record of visited
            knowhere::Neighbor nn(saved_indices[id4], dis, saved_statuses[id4]);
            if (func_add_candidate(nn)) {
#if defined(USE_PREFETCH)
                // TODO
                // _mm_prefetch(get_linklist0(v), _MM_HINT_T0);
#endif
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        // done
        return stats;
    }

    // perform the search on a given level.
    // it is assumed that retset is initialized and contains the initial nodes.
    template <typename RetSetT>
    faiss::HNSWStats search_on_a_level(
            RetSetT& retset,
            const int level,
            knowhere::IteratorMinHeap* const __restrict disqualified = nullptr,
            const float initial_accumulated_alpha = 1.0f) {
        faiss::HNSWStats stats;

        //
        float accumulated_alpha = initial_accumulated_alpha;

        // what to do with a accepted candidate
        auto add_search_candidate = [&](const knowhere::Neighbor n) {
            return retset.insert(n, disqualified);
        };

        // iterate while possible
        while (retset.has_next()) {
            // get a node to be processed
            const knowhere::Neighbor neighbor = retset.pop();

            // analyze its neighbors
            faiss::HNSWStats local_stats = evaluate_single_node(
                    neighbor.id,
                    level,
                    accumulated_alpha,
                    add_search_candidate);

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        // done
        return stats;
    }

    // traverse down to the level 0
    faiss::HNSWStats greedy_search_top_levels(
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        // iterate through upper levels
        for (int level = hnsw.max_level; level >= 1; level--) {
            // update the visitor
            graph_visitor.visit_level(level);

            // alter the value of 'nearest'
            faiss::HNSWStats local_stats =
                    greedy_update_nearest(level, nearest, d_nearest);

            // update stats
            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        return stats;
    }

    // perform the search.
    faiss::HNSWStats search(
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels) {
        faiss::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = params ? params->efSearch : hnsw.efSearch;

        // greedy search on upper levels?
        if (hnsw.upper_beam != 1) {
            FAISS_THROW_MSG("Not implemented");
            return stats;
        }

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = std::max((idx_t)efSearch, k);
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // perform the search of the level 0.
        faiss::HNSWStats local_stats = search_on_a_level(retset, 0);

        // todo: switch to brute-force in case of (retset.size() < k)

        // populate the result
        const idx_t len = std::min((idx_t)retset.size(), k);
        for (idx_t i = 0; i < len; i++) {
            distances[i] = retset[i].distance;
            labels[i] = (idx_t)retset[i].id;
        }
        if (len < k) {
            for (idx_t idx = len; idx < k; idx++) {
                labels[idx] = -1;
                distances[idx] = std::numeric_limits<float>::max();
            }
        }
        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // done
        return stats;
    }

    faiss::HNSWStats range_search(
            const float radius,
            typename faiss::RangeSearchBlockResultHandler<
                    faiss::CMax<float, int64_t>>::
                    SingleResultHandler* const __restrict rres) {
        faiss::HNSWStats stats;

        // is the graph empty?
        if (hnsw.entry_point == -1) {
            return stats;
        }

        // grab some needed parameters
        const int efSearch = params ? params->efSearch : hnsw.efSearch;

        // greedy search on upper levels?
        if (hnsw.upper_beam != 1) {
            FAISS_THROW_MSG("Not implemented");
            return stats;
        }

        // yes.
        // greedy search on upper levels.

        // initialize the starting point.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        // iterate through upper levels
        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search

        // update the visitor
        graph_visitor.visit_level(0);

        // initialize the container for candidates
        const idx_t n_candidates = efSearch;
        knowhere::NeighborSetDoublePopList retset(n_candidates);

        // initialize retset with a single 'nearest' point
        {
            if (!filter.is_member(nearest)) {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kInvalid));
            } else {
                retset.insert(knowhere::Neighbor(
                        nearest, d_nearest, knowhere::Neighbor::kValid));
            }

            visited_nodes[nearest] = true;
        }

        // perform the search of the level 0.
        faiss::HNSWStats local_stats = search_on_a_level(retset, 0);

        // update stats
        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        // select candidates that match our criteria
        faiss::HNSWStats pick_stats;

        visited_nodes.clear();

        std::queue<std::pair<float, int64_t>> radius_queue;
        for (size_t i = retset.size(); (i--) > 0;) {
            const auto candidate = retset[i];
            if (candidate.distance < radius) {
                radius_queue.push({candidate.distance, candidate.id});
                rres->add_result(candidate.distance, candidate.id);

                visited_nodes[candidate.id] = true;
            }
        }

        while (!radius_queue.empty()) {
            auto current = radius_queue.front();
            radius_queue.pop();

            size_t id_begin = 0;
            size_t id_end = 0;
            hnsw.neighbor_range(current.second, 0, &id_begin, &id_end);

            for (size_t id = id_begin; id < id_end; id++) {
                const auto ngb = hnsw.neighbors[id];
                if (ngb == -1) {
                    break;
                }

                if (visited_nodes[ngb]) {
                    continue;
                }

                visited_nodes[ngb] = true;

                if (filter.is_member(ngb)) {
                    const float dis = qdis(ngb);
                    if (dis < radius) {
                        radius_queue.push({dis, ngb});
                        rres->add_result(dis, ngb);
                    }

                    if (track_hnsw_stats) {
                        pick_stats.ndis += 1;
                    }
                }
            }
        }

        // update stats
        if (track_hnsw_stats) {
            stats.combine(pick_stats);
        }

        return stats;
    }
};

// Optimized JAG-HNSW Searcher: Uses combined distance for ranking while
// maintaining compatibility with efficient NeighborSetDoublePopList
// Key optimization: Store combined_dist in Neighbor.distance for ranking,
//                  collect all candidates, then filter and sort by actual distance
// v3: Added adaptive filter_weight based on observed filter ratio
template <
        typename DistanceComputerT,
        typename GraphVisitorT,
        typename VisitedT,
        typename FilterT>
struct v2_hnsw_jag_searcher {
    using storage_idx_t = faiss::HNSW::storage_idx_t;
    using idx_t = faiss::idx_t;

    // hnsw structure.
    const faiss::HNSW& hnsw;

    // computes distances.
    DistanceComputerT& qdis;

    // records visited edges.
    GraphVisitorT& graph_visitor;

    // tracks visited nodes.
    VisitedT& visited_nodes;

    // filter for disabled nodes.
    const FilterT& filter;

    // Alpha parameter for probabilistic filtered node admission
    const float kAlpha;

    // JAG parameters
    const float base_filter_weight;    // Base weight for filter distance
    const bool adaptive_weight;         // Enable adaptive weight adjustment

    // custom parameters of HNSW search.
    const faiss::SearchParametersHNSW* params;

    //
    v2_hnsw_jag_searcher(
            const faiss::HNSW& hnsw_,
            DistanceComputerT& qdis_,
            GraphVisitorT& graph_visitor_,
            VisitedT& visited_nodes_,
            const FilterT& filter_,
            const float kAlpha_,
            const float filter_weight_,
            const int /*candidate_pool_size_*/,  // Not used in optimized version
            const faiss::SearchParametersHNSW* params_)
            : hnsw{hnsw_},
              qdis{qdis_},
              graph_visitor{graph_visitor_},
              visited_nodes{visited_nodes_},
              filter{filter_},
              kAlpha{kAlpha_},
              base_filter_weight{filter_weight_},
              adaptive_weight{filter_weight_ > 0},  // Enable adaptive if weight > 0
              params{params_} {}

    v2_hnsw_jag_searcher(const v2_hnsw_jag_searcher&) = delete;
    v2_hnsw_jag_searcher(v2_hnsw_jag_searcher&&) = delete;
    v2_hnsw_jag_searcher& operator=(const v2_hnsw_jag_searcher&) = delete;
    v2_hnsw_jag_searcher& operator=(v2_hnsw_jag_searcher&&) = delete;

    // Compute combined distance for JAG ranking
    // Goal: invalid nodes should have WORSE priority than valid nodes
    // For L2 (positive): larger distance = worse, so add penalty
    // For IP (negative): more negative = better, so we need to make invalid LESS negative (closer to 0)
    // Therefore: penalty direction is the same (always positive), but effect on ranking differs
    // Solution: Use absolute value comparison in NeighborSetDoublePopList instead
    // Here we keep combined distance such that abs(invalid_combined) > abs(valid_combined)
    inline float
    compute_combined_dist(float vec_dist, bool is_filtered, float current_weight) const {
        if (!is_filtered) {
            return vec_dist;  // No penalty for valid nodes
        }
        // For both L2 and IP, we want invalid nodes to be deprioritized
        // L2: dist > 0, combined = dist + weight (larger = worse) ✓
        // IP: dist < 0, combined = dist + weight (closer to 0 = worse in min-heap) ✓
        // So we always add the weight
        return vec_dist + current_weight;
    }

    // Compute combined distance with normalized_h adjustment (JAG paper formula)
    // combined = vec_dist + weight * normalized_h * filter_dist
    // For binary filter: filter_dist = 0 (valid) or 1 (invalid)
    inline float
    compute_combined_dist_with_normalized_h(
            float vec_dist, bool is_filtered, float current_weight, float normalized_h) const {
        if (!is_filtered) {
            return vec_dist;  // No penalty for valid nodes
        }
        // Apply the JAG paper formula: combined = vec_dist + weight * normalized_h * filter_dist
        // filter_dist = 1 for invalid nodes (binary filter)
        return vec_dist + current_weight * normalized_h;
    }

    // Adaptive weight calculation based on observed filter ratio
    // Uses logarithmic scaling like the JAG paper: weight = log(1/p)
    // where p is the fraction of valid nodes
    // This gives higher weights when fewer nodes are valid (higher filter ratio)
    inline float
    get_adaptive_weight(int total_seen, int invalid_seen) const {
        if (!adaptive_weight || total_seen < 10) {
            return base_filter_weight;
        }

        // Calculate valid ratio (fraction of nodes that match the filter)
        int valid_seen = total_seen - invalid_seen;
        float valid_ratio = static_cast<float>(valid_seen) / total_seen;

        // Avoid log(0) by using minimum ratio
        valid_ratio = std::max(valid_ratio, 0.01f);

        // Logarithmic weight scaling (inspired by JAG paper)
        // weight = base_weight * log(1/valid_ratio)
        float log_weight = std::log(1.0f / valid_ratio);

        return base_filter_weight * log_weight;
    }

    // Auto-select optimal weight based on estimated filter ratio
    // Based on JAG paper findings: different filter ratios need different weights
    // Lower filter ratio -> lower weight (QPS priority)
    // Higher filter ratio -> higher weight (recall priority)
    inline float
    get_auto_weight_for_filter_ratio(float filter_ratio) const {
        // Filter ratio = fraction of points that are filtered OUT (invalid)
        // valid_ratio = 1 - filter_ratio

        if (filter_ratio <= 0.15f) {
            // Low filter ratio: prioritize QPS with low weight
            // Paper shows 0.1 is optimal here
            return 0.1f;
        } else if (filter_ratio <= 0.30f) {
            // Medium filter ratio: balanced
            return 0.3f;
        } else if (filter_ratio <= 0.50f) {
            // High filter ratio: prioritize recall
            return 0.5f;
        } else {
            // Very high filter ratio: need aggressive weight for recall
            // Paper uses up to 10.0 for extreme cases
            return 1.0f;
        }
    }

    // Online estimation of normalized_h factor (inspired by JAG paper)
    // The paper uses: normalized_h[p] = std_vec_dist / std_filter_dist for each point
    // RWalksVamana variant uses a global: normalized_h = 0.1 * avg_vec_dist / avg_filter_dist
    // Here we estimate it online from observed distances during search
    // This avoids O(n) storage and O(n*100) precomputation cost
    inline float
    estimate_normalized_h(float sum_vec_dist, float sum_filter_dist, int sample_count) const {
        if (sample_count < 10 || sum_filter_dist < 0.001f) {
            return 1.0f;  // Default fallback
        }
        // Use ratio of sums (equivalent to ratio of averages)
        float raw_ratio = sum_vec_dist / sum_filter_dist;
        // Apply scaling factor of 0.1 as suggested by RWalksVamana
        // This prevents over-weighting filter distance
        return 0.1f * raw_ratio;
    }

    // Early pruning check - with high safety margin for high recall
    // Only prune if filter penalty is MUCH larger than worst distance
    // This balances recall and QPS
    inline bool
    should_prune_by_filter(float filter_dist, float current_weight, float worst_dist) const {
        // For binary filter: filter_dist is 0 (valid) or 1 (invalid)
        // Use 4x safety margin - only prune hopeless candidates
        // This maintains good recall while still providing QPS benefit
        return (filter_dist * current_weight) > (worst_dist * 4.0f);
    }

    // greedily update a nearest vector at a given level.
    faiss::HNSWStats greedy_update_nearest(
            const int level,
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        for (;;) {
            storage_idx_t prev_nearest = nearest;

            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(nearest, level, &begin, &end);

            size_t count = 0;
            for (size_t i = begin; i < end; i++) {
                storage_idx_t v = hnsw.neighbors[i];
                if (v < 0) {
                    break;
                }
                count += 1;
            }

            for (size_t i = begin; i < begin + count; i++) {
                storage_idx_t v = hnsw.neighbors[i];

                const float dis = qdis(v);

                graph_visitor.visit_edge(level, prev_nearest, nearest, dis);

                if (dis < d_nearest) {
                    nearest = v;
                    d_nearest = dis;
                }
            }

            if (track_hnsw_stats) {
                stats.ndis += count;
                stats.nhops += 1;
            }

            if (nearest == prev_nearest) {
                return stats;
            }
        }
    }

    // Optimized: evaluate neighbors and add candidates with JAG ranking
    // Stores combined_dist in Neighbor.distance for efficient ranking
    template <typename FuncAddCandidate>
    faiss::HNSWStats evaluate_single_node_jag(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        faiss::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        size_t counter = 0;
        size_t saved_indices[4];
        int saved_statuses[4];
        float saved_distances[4];  // Store actual distances

        size_t ndis = 0;
        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }

            if (visited_nodes.get(v1)) {
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            visited_nodes.set(v1);

            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                status = knowhere::Neighbor::kInvalid;

                // Alpha mechanism: probabilistic admission
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }
                accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            counter += 1;
            ndis += 1;

            if (counter == 4) {
                float dis[4] = {0, 0, 0, 0};
                qdis.distances_batch_4(
                        saved_indices[0],
                        saved_indices[1],
                        saved_indices[2],
                        saved_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    graph_visitor.visit_edge(
                            level, node_id, saved_indices[id4], dis[id4]);

                    // JAG: Use combined distance for ranking
                    bool is_filtered = (saved_statuses[id4] == knowhere::Neighbor::kInvalid);
                    float combined = compute_combined_dist(dis[id4], is_filtered, base_filter_weight);

                    // Store combined distance in distance field for ranking
                    knowhere::Neighbor nn(saved_indices[id4], combined, saved_statuses[id4]);
                    // Store actual distance in a separate structure or recover later
                    func_add_candidate(nn);
                }

                counter = 0;
            }
        }

        // Process leftovers
        for (size_t id4 = 0; id4 < counter; id4++) {
            const float dis = qdis(saved_indices[id4]);

            graph_visitor.visit_edge(level, node_id, saved_indices[id4], dis);

            bool is_filtered = (saved_statuses[id4] == knowhere::Neighbor::kInvalid);
            float combined = compute_combined_dist(dis, is_filtered, base_filter_weight);

            knowhere::Neighbor nn(saved_indices[id4], combined, saved_statuses[id4]);
            func_add_candidate(nn);
        }

        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        return stats;
    }

    // Search on a level using JAG ranking with adaptive weight
    // v3: Track filter ratio and adapt weight dynamically
    // v4: Add early pruning based on filter distance
    // v5: Add multi-tier search with refinement pass
    faiss::HNSWStats search_on_a_level_jag_v2(
            knowhere::NeighborSetDoublePopListJAG& retset,
            const int level,
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels,
            float initial_accumulated_alpha = 1.0f) {
        faiss::HNSWStats stats;

        float accumulated_alpha = initial_accumulated_alpha;

        // Track filter ratio for adaptive weight
        int total_nodes_seen = 0;
        int invalid_nodes_seen = 0;
        float current_weight = base_filter_weight;

        // Track normalized_h estimation (online estimation from observed distances)
        // Inspired by JAG paper: normalized_h = std_vec_dist / std_filter_dist
        // We use: normalized_h = 0.1 * sum_vec_dist / sum_filter_dist
        float sum_vec_dist = 0.0f;
        float sum_filter_dist = 0.0f;
        float estimated_normalized_h = 1.0f;  // Default fallback

        auto add_search_candidate = [&](const knowhere::Neighbor& n) {
            return retset.insert(n);
        };

        // ============ TIER 1: Weighted JAG Search ============
        // Use combined distance to prioritize valid nodes
        while (retset.has_next()) {
            const knowhere::Neighbor neighbor = retset.pop();

            // Update adaptive weight periodically
            if (adaptive_weight && total_nodes_seen > 0 && total_nodes_seen % 50 == 0) {
                current_weight = get_adaptive_weight(total_nodes_seen, invalid_nodes_seen);
            }

            // Update normalized_h estimation periodically
            if (total_nodes_seen > 0 && total_nodes_seen % 50 == 0) {
                estimated_normalized_h = estimate_normalized_h(sum_vec_dist, sum_filter_dist, total_nodes_seen);
            }

            // Get worst distance in result set for early pruning
            float worst_dist = retset.at_search_back_dist();

            faiss::HNSWStats local_stats = evaluate_single_node_jag_v2(
                    neighbor.id,
                    level,
                    accumulated_alpha,
                    current_weight,
                    estimated_normalized_h,
                    worst_dist,
                    total_nodes_seen,
                    invalid_nodes_seen,
                    sum_vec_dist,
                    sum_filter_dist,
                    add_search_candidate);

            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        // Note: Multi-tier refinement was considered but removed for simplicity.
        // The adaptive weight mechanism already provides good balance between
        // QPS (low filter ratio) and recall (high filter ratio).
        // Future: Could implement proper multi-tier search if needed.

        // Extract results directly from retset's valid_ns_
        // Note: valid neighbors have actual distance stored (combined = actual for valid nodes)
        // No need to recompute - nbr.distance is already the correct value
        const idx_t len = std::min((idx_t)retset.size(), k);
        for (idx_t i = 0; i < len; i++) {
            const auto& nbr = retset[i];
            // For valid nodes, combined_dist = actual_dist, so no recomputation needed
            distances[i] = nbr.distance;
            labels[i] = nbr.id;
        }

        // Fill remaining slots
        for (idx_t i = len; i < k; i++) {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
        }

        return stats;
    }

    // Optimized v2: For valid neighbors, store actual distance
    // For invalid neighbors, store combined distance (for ranking in invalid_ns_)
    // v3: Accept current_weight and track filter ratio
    // v4: Add early pruning based on filter distance
    // v6: Add normalized_h estimation for more accurate combined distance
    template <typename FuncAddCandidate>
    faiss::HNSWStats evaluate_single_node_jag_v2(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            const float current_weight,
            const float estimated_normalized_h,
            const float worst_dist,
            int& total_nodes_seen,
            int& invalid_nodes_seen,
            float& sum_vec_dist,
            float& sum_filter_dist,
            FuncAddCandidate func_add_candidate) {
        faiss::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        size_t counter = 0;
        size_t saved_indices[4];
        bool saved_is_valid[4];  // true = valid, false = invalid

        size_t ndis = 0;
        int early_pruned = 0;  // Count of nodes pruned by early filter check

        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }

            if (visited_nodes.get(v1)) {
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            visited_nodes.set(v1);

            bool is_valid = filter.is_member(v1);
            total_nodes_seen++;
            if (!is_valid) {
                invalid_nodes_seen++;

                // Early pruning: skip vector distance computation if filter distance alone
                // exceeds the worst distance in the result set
                // Inspired by JAG paper WeightJAG lines 422-427
                // Use normalized_h-adjusted weight for pruning decision
                float effective_weight = current_weight * estimated_normalized_h;
                if (should_prune_by_filter(1.0f, effective_weight, worst_dist)) {
                    early_pruned++;
                    continue;  // Skip computing vector distance
                }

                // JAG: Skip Alpha mechanism for invalid nodes to allow full graph exploration
                // The weight-based ranking will handle prioritization
                // Original Alpha mechanism commented out:
                // accumulated_alpha += kAlpha;
                // if (accumulated_alpha < 1.0f) {
                //     continue;
                // }
                // accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_is_valid[counter] = is_valid;
            counter += 1;
            ndis += 1;

            if (counter == 4) {
                float dis[4] = {0, 0, 0, 0};
                qdis.distances_batch_4(
                        saved_indices[0],
                        saved_indices[1],
                        saved_indices[2],
                        saved_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    graph_visitor.visit_edge(
                            level, node_id, saved_indices[id4], dis[id4]);

                    // Track distances for normalized_h estimation
                    // For binary filter: filter_dist = 0 (valid) or 1 (invalid)
                    float filter_dist = saved_is_valid[id4] ? 0.0f : 1.0f;
                    sum_vec_dist += std::fabs(dis[id4]);
                    sum_filter_dist += filter_dist;

                    // JAG v6: Use combined distance with normalized_h adjustment
                    // Formula: combined = vec_dist + weight * normalized_h * filter_dist
                    // For valid nodes: combined = dis (no penalty)
                    // For invalid nodes: combined = dis + weight * normalized_h
                    float combined = compute_combined_dist_with_normalized_h(
                            dis[id4], !saved_is_valid[id4], current_weight, estimated_normalized_h);
                    int status = saved_is_valid[id4] ? knowhere::Neighbor::kValid : knowhere::Neighbor::kInvalid;
                    knowhere::Neighbor nn(saved_indices[id4], combined, status);
                    func_add_candidate(nn);
                }

                counter = 0;
            }
        }

        // Process leftovers
        for (size_t id4 = 0; id4 < counter; id4++) {
            const float dis = qdis(saved_indices[id4]);

            graph_visitor.visit_edge(level, node_id, saved_indices[id4], dis);

            // Track distances for normalized_h estimation
            float filter_dist = saved_is_valid[id4] ? 0.0f : 1.0f;
            sum_vec_dist += std::fabs(dis);
            sum_filter_dist += filter_dist;

            // JAG v6: Use combined distance with normalized_h adjustment
            float combined = compute_combined_dist_with_normalized_h(
                    dis, !saved_is_valid[id4], current_weight, estimated_normalized_h);
            int status = saved_is_valid[id4] ? knowhere::Neighbor::kValid : knowhere::Neighbor::kInvalid;
            knowhere::Neighbor nn(saved_indices[id4], combined, status);
            func_add_candidate(nn);
        }

        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        return stats;
    }

    // Optimized: evaluate neighbors and add candidates with JAG ranking
    // Also passes actual distance to callback to avoid recomputation
    template <typename FuncAddCandidate>
    faiss::HNSWStats evaluate_single_node_jag_with_actual_dist(
            const idx_t node_id,
            const int level,
            float& accumulated_alpha,
            FuncAddCandidate func_add_candidate) {
        faiss::HNSWStats stats;

        size_t begin = 0;
        size_t end = 0;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        size_t counter = 0;
        size_t saved_indices[4];
        int saved_statuses[4];

        size_t ndis = 0;
        for (size_t j = begin; j < end; j++) {
            const storage_idx_t v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }

            if (visited_nodes.get(v1)) {
                graph_visitor.visit_edge(level, node_id, v1, -1);
                continue;
            }

            visited_nodes.set(v1);

            int status = knowhere::Neighbor::kValid;
            if (!filter.is_member(v1)) {
                status = knowhere::Neighbor::kInvalid;

                // Alpha mechanism: probabilistic admission
                accumulated_alpha += kAlpha;
                if (accumulated_alpha < 1.0f) {
                    continue;
                }
                accumulated_alpha -= 1.0f;
            }

            saved_indices[counter] = v1;
            saved_statuses[counter] = status;
            counter += 1;
            ndis += 1;

            if (counter == 4) {
                float dis[4] = {0, 0, 0, 0};
                qdis.distances_batch_4(
                        saved_indices[0],
                        saved_indices[1],
                        saved_indices[2],
                        saved_indices[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    graph_visitor.visit_edge(
                            level, node_id, saved_indices[id4], dis[id4]);

                    // JAG: Use combined distance for ranking
                    bool is_filtered = (saved_statuses[id4] == knowhere::Neighbor::kInvalid);
                    float combined = compute_combined_dist(dis[id4], is_filtered, base_filter_weight);

                    // Store combined distance for ranking, pass actual distance to callback
                    knowhere::Neighbor nn(saved_indices[id4], combined, saved_statuses[id4]);
                    func_add_candidate(nn, dis[id4]);  // Pass actual distance
                }

                counter = 0;
            }
        }

        // Process leftovers
        for (size_t id4 = 0; id4 < counter; id4++) {
            const float dis = qdis(saved_indices[id4]);

            graph_visitor.visit_edge(level, node_id, saved_indices[id4], dis);

            bool is_filtered = (saved_statuses[id4] == knowhere::Neighbor::kInvalid);
            float combined = compute_combined_dist(dis, is_filtered, base_filter_weight);

            knowhere::Neighbor nn(saved_indices[id4], combined, saved_statuses[id4]);
            func_add_candidate(nn, dis);  // Pass actual distance
        }

        if (track_hnsw_stats) {
            stats.ndis = ndis;
            stats.nhops = 1;
        }

        return stats;
    }

    // traverse down to the level 0
    faiss::HNSWStats greedy_search_top_levels(
            storage_idx_t& nearest,
            float& d_nearest) {
        faiss::HNSWStats stats;

        for (int level = hnsw.max_level; level >= 1; level--) {
            graph_visitor.visit_level(level);

            faiss::HNSWStats local_stats =
                    greedy_update_nearest(level, nearest, d_nearest);

            if (track_hnsw_stats) {
                stats.combine(local_stats);
            }
        }

        return stats;
    }

    // perform the search.
    faiss::HNSWStats search(
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels) {
        faiss::HNSWStats stats;

        if (hnsw.entry_point == -1) {
            return stats;
        }

        const int efSearch = params ? params->efSearch : hnsw.efSearch;

        if (hnsw.upper_beam != 1) {
            FAISS_THROW_MSG("Not implemented");
            return stats;
        }

        // greedy search on upper levels.
        storage_idx_t nearest = hnsw.entry_point;
        float d_nearest = qdis(nearest);

        auto bottom_levels_stats = greedy_search_top_levels(nearest, d_nearest);

        if (track_hnsw_stats) {
            stats.combine(bottom_levels_stats);
        }

        // level 0 search
        graph_visitor.visit_level(0);

        // For JAG v2: Use NeighborSetDoublePopListJAG (absolute value comparison for IP distance)
        const idx_t n_candidates = std::max((idx_t)efSearch, k);
        knowhere::NeighborSetDoublePopListJAG retset(n_candidates);

        // Initialize retset with entry point
        {
            bool is_valid = filter.is_member(nearest);
            float dist_to_store = is_valid ? d_nearest : compute_combined_dist(d_nearest, true, base_filter_weight);
            int status = is_valid ? knowhere::Neighbor::kValid : knowhere::Neighbor::kInvalid;

            retset.insert(knowhere::Neighbor(nearest, dist_to_store, status));
            visited_nodes[nearest] = true;
        }

        // Perform JAG search v2 - results extracted directly in the function
        faiss::HNSWStats local_stats = search_on_a_level_jag_v2(
                retset, 0, k, distances, labels);

        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        return stats;
    }
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
