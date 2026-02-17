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
    faiss::HNSWStats search_on_a_level(
            knowhere::NeighborSetDoublePopList& retset,
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

// JAG-HNSW Searcher: Combines JAG (filter-guided traversal) with Alpha (probabilistic admission)
// Phase 1: Use combined_dist (vec_dist + filter_weight * filter_dist) for frontier sorting
//          Apply Alpha mechanism for filtered nodes to maintain graph connectivity
// Phase 2: Sort candidates by vec_dist and filter for top-k matches
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
    const float filter_weight;          // Weight for filter distance in combined score
    const int candidate_pool_size;      // Size of candidate pool for final ranking

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
            const int candidate_pool_size_,
            const faiss::SearchParametersHNSW* params_)
            : hnsw{hnsw_},
              qdis{qdis_},
              graph_visitor{graph_visitor_},
              visited_nodes{visited_nodes_},
              filter{filter_},
              kAlpha{kAlpha_},
              filter_weight{filter_weight_},
              candidate_pool_size{candidate_pool_size_},
              params{params_} {}

    v2_hnsw_jag_searcher(const v2_hnsw_jag_searcher&) = delete;
    v2_hnsw_jag_searcher(v2_hnsw_jag_searcher&&) = delete;
    v2_hnsw_jag_searcher& operator=(const v2_hnsw_jag_searcher&) = delete;
    v2_hnsw_jag_searcher& operator=(v2_hnsw_jag_searcher&&) = delete;

    // Compute combined distance for JAG ranking
    // combined_dist = vec_dist + filter_weight * filter_dist
    // filter_dist = 0 if node matches filter, 1 otherwise
    inline float
    compute_combined_dist(float vec_dist, bool is_filtered) const {
        return vec_dist + filter_weight * (is_filtered ? 1.0f : 0.0f);
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

    // Candidate for JAG search with candidate pool management
    struct JagCandidate {
        storage_idx_t id;
        float vec_dist;
        bool is_filtered;
        float combined_dist;

        // For min-heap by combined distance
        bool operator<(const JagCandidate& other) const {
            return combined_dist > other.combined_dist;
        }
    };

    // JAG search on level 0 with combined Alpha mechanism and candidate pool
    faiss::HNSWStats search_on_level_jag(
            const idx_t k,
            float* __restrict distances,
            idx_t* __restrict labels) {
        faiss::HNSWStats stats;

        const int efSearch = params ? params->efSearch : hnsw.efSearch;
        const int pool_size = (candidate_pool_size > 0) ? candidate_pool_size : efSearch * 4;

        // Priority queue for frontier (sorted by combined_dist)
        std::priority_queue<JagCandidate> frontier;

        // Candidate pool for final ranking (stores all visited nodes)
        std::vector<JagCandidate> candidate_pool;
        candidate_pool.reserve(pool_size);

        // Alpha accumulator for probabilistic filtered node admission
        float accumulated_alpha = 1.0f;

        // Initialize with entry point
        {
            storage_idx_t nearest = hnsw.entry_point;
            if (nearest < 0) {
                return stats;
            }

            float d_nearest = qdis(nearest);
            bool is_filtered = !filter.is_member(nearest);
            float combined = compute_combined_dist(d_nearest, is_filtered);

            frontier.push({nearest, d_nearest, is_filtered, combined});
            visited_nodes[nearest] = true;

            // Add to candidate pool
            candidate_pool.push_back({nearest, d_nearest, is_filtered, combined});
        }

        // Main search loop with JAG ranking and Alpha admission
        while (!frontier.empty()) {
            JagCandidate current = frontier.top();
            frontier.pop();

            // Process neighbors
            size_t begin = 0;
            size_t end = 0;
            hnsw.neighbor_range(current.id, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                const storage_idx_t v1 = hnsw.neighbors[j];

                if (v1 < 0) {
                    break;
                }

                if (visited_nodes.get(v1)) {
                    graph_visitor.visit_edge(0, current.id, v1, -1);
                    continue;
                }

                // Mark as visited
                visited_nodes.set(v1);

                // Check filter status
                bool is_filtered = !filter.is_member(v1);

                // Alpha mechanism: probabilistic admission for filtered nodes
                if (is_filtered) {
                    accumulated_alpha += kAlpha;
                    if (accumulated_alpha < 1.0f) {
                        // Skip this filtered node (but still add to candidate pool)
                        // This maintains graph connectivity while prioritizing valid nodes
                    }
                    // Allow this node to be added to frontier
                    accumulated_alpha -= 1.0f;
                }

                // Compute distance
                const float dis = qdis(v1);
                float combined = compute_combined_dist(dis, is_filtered);

                graph_visitor.visit_edge(0, current.id, v1, dis);

                // Add to candidate pool (all nodes, for final ranking)
                candidate_pool.push_back({v1, dis, is_filtered, combined});

                // Add to frontier (JAG ranking by combined_dist)
                frontier.push({v1, dis, is_filtered, combined});

                if (track_hnsw_stats) {
                    stats.ndis += 1;
                }

                // Early termination if candidate pool is full and frontier is exhausted
                // or if we've visited enough nodes
                if (static_cast<int>(candidate_pool.size()) >= pool_size) {
                    // Check if frontier's best combined_dist is worse than
                    // the k-th best vec_dist among valid candidates
                    int valid_count = 0;
                    float kth_valid_dist = std::numeric_limits<float>::max();
                    for (const auto& c : candidate_pool) {
                        if (!c.is_filtered) {
                            valid_count++;
                            if (valid_count >= k) {
                                kth_valid_dist = c.vec_dist;
                                break;
                            }
                        }
                    }
                    // If frontier is exhausted or best combined > kth valid + margin
                    if (frontier.empty() ||
                        (!frontier.empty() && frontier.top().combined_dist > kth_valid_dist + filter_weight)) {
                        break;
                    }
                }
            }

            stats.nhops += 1;
        }

        // Phase 2: Sort candidates by vector distance (not combined distance)
        // and select top-k valid (non-filtered) nodes
        std::sort(candidate_pool.begin(), candidate_pool.end(),
                  [](const JagCandidate& a, const JagCandidate& b) {
                      return a.vec_dist < b.vec_dist;
                  });

        // Select top-k valid candidates
        idx_t result_count = 0;
        for (const auto& candidate : candidate_pool) {
            if (!candidate.is_filtered) {
                distances[result_count] = candidate.vec_dist;
                labels[result_count] = candidate.id;
                result_count++;
                if (result_count >= k) {
                    break;
                }
            }
        }

        // Fill remaining slots if not enough valid candidates
        for (idx_t i = result_count; i < k; i++) {
            labels[i] = -1;
            distances[i] = std::numeric_limits<float>::max();
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

        // level 0 search with JAG
        graph_visitor.visit_level(0);

        faiss::HNSWStats local_stats = search_on_level_jag(k, distances, labels);

        if (track_hnsw_stats) {
            stats.combine(local_stats);
        }

        return stats;
    }
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
