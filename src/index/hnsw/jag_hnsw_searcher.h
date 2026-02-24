// Copyright 2025 The Knowhere Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KNOWHERE_INDEX_HNSW_JAG_HNSW_SEARCHER_H_
#define KNOWHERE_INDEX_HNSW_JAG_HNSW_SEARCHER_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "common/filter_distance.h"

namespace knowhere {

// JAG-HNSW Searcher
// Implements JAG (Join-And-AGgregate) search strategy on HNSW index
// Uses filter distance to guide graph traversal
template <typename DistanceType, typename IndexType>
class JagHnswSearcher {
 public:
    struct Config {
        int beam_width = 16;         // Max candidates in frontier
        float filter_weight = 1.0f;  // Weight for filter distance in combined score
        int max_visits = 10000;      // Max nodes to visit per query
        int ef_search = 64;          // EF parameter for HNSW search
    };

    struct Metrics {
        int64_t nodes_visited = 0;
        int64_t valid_visits = 0;      // Visits to filter-matching nodes
        int64_t distance_computations = 0;
        int64_t filter_checks = 0;
    };

    // Search result: (node_id, distance)
    using SearchResult = std::pair<int64_t, DistanceType>;

    // Internal search state for priority queue
    struct SearchState {
        int64_t node_id;
        DistanceType vector_dist;
        int filter_dist;
        DistanceType combined_dist;

        // Higher priority = lower combined distance
        bool
        operator<(const SearchState& other) const {
            if (combined_dist != other.combined_dist) {
                return combined_dist > other.combined_dist;  // Min-heap by combined distance
            }
            // Tie-breaker: prefer lower filter distance
            if (filter_dist != other.filter_dist) {
                return filter_dist > other.filter_dist;
            }
            return node_id > other.node_id;
        }
    };

    JagHnswSearcher() = default;

    // JAG-style beam search with filter guidance
    // Returns top-k results that match the filter
    template <typename HnswIndex, typename DistanceFunc>
    std::vector<SearchResult>
    Search(const HnswIndex& index, const float* query, int k,
           const FilterDistanceCalculator& filter_calc, const DistanceFunc& dist_func,
           const Config& config, Metrics* metrics = nullptr) {
        // Use vector-backed min-heap so we can scan all elements for the worst.
        // SearchState::operator< is inverted (a < b ↔ a.combined > b.combined),
        // so std::make/push/pop_heap give min-heap semantics: front() = best.
        std::vector<SearchState> frontier;
        frontier.reserve(config.beam_width * 2 + 1);
        std::unordered_set<int64_t> visited;

        // All valid candidates found; sorted and truncated to k at the end.
        // This mirrors the ef_search approach: explore fully, then pick top-k
        // by vector distance. Stopping early (results.size() >= k) was the bug:
        // the first k valid nodes found are not necessarily the k nearest ones.
        std::vector<SearchResult> all_candidates;

        // Initialize metrics
        Metrics local_metrics;
        Metrics* m = metrics ? metrics : &local_metrics;

        // Get entry point
        int64_t entry_point = GetEntryPoint(index);
        if (entry_point < 0) {
            return {};
        }

        // Initialize with entry point
        {
            DistanceType entry_dist = dist_func(query, entry_point);
            int entry_filter = filter_calc.Calculate(entry_point);

            m->distance_computations++;
            m->filter_checks++;

            frontier.push_back({entry_point, entry_dist, entry_filter,
                                 entry_dist + config.filter_weight * entry_filter});
            std::push_heap(frontier.begin(), frontier.end());
        }

        // Main search loop: explore until frontier is empty or budget is exhausted.
        // Do NOT terminate early on all_candidates.size() >= k — that is what
        // caused the recall bug. We collect all reachable valid nodes and pick
        // the best k afterwards.
        while (!frontier.empty() &&
               static_cast<int>(visited.size()) < config.max_visits) {
            // Pop best candidate (minimum combined distance)
            std::pop_heap(frontier.begin(), frontier.end());
            SearchState current = frontier.back();
            frontier.pop_back();

            // Skip if already visited
            if (visited.count(current.node_id)) {
                continue;
            }
            visited.insert(current.node_id);
            m->nodes_visited++;

            // Collect valid candidates; keep exploring regardless
            if (current.filter_dist == 0) {
                all_candidates.push_back({current.node_id, current.vector_dist});
                m->valid_visits++;
            }

            // Expand neighbors
            auto neighbors = GetNeighbors(index, current.node_id);
            for (int64_t neighbor : neighbors) {
                if (visited.count(neighbor)) {
                    continue;
                }

                // Calculate filter distance first (cheaper)
                int filter_dist = filter_calc.Calculate(neighbor);
                m->filter_checks++;

                // Early pruning based on filter distance alone:
                // if even the best possible vector distance (0) still exceeds
                // the worst combined distance in the frontier, skip.
                if (frontier.size() >= static_cast<size_t>(config.beam_width) &&
                    filter_dist > 0 && config.filter_weight > 0) {
                    DistanceType worst_combined = GetWorstCombinedDistance(frontier);
                    if (config.filter_weight * filter_dist > worst_combined) {
                        continue;  // Prune this neighbor
                    }
                }

                // Calculate vector distance
                DistanceType dist = dist_func(query, neighbor);
                m->distance_computations++;

                DistanceType combined = dist + config.filter_weight * filter_dist;

                // Add to frontier if it has room or beats the current worst
                if (static_cast<int>(frontier.size()) < config.beam_width ||
                    combined < GetWorstCombinedDistance(frontier)) {
                    frontier.push_back({neighbor, dist, filter_dist, combined});
                    std::push_heap(frontier.begin(), frontier.end());

                    // Keep frontier size bounded: discard the worst candidates
                    if (static_cast<int>(frontier.size()) > config.beam_width * 2) {
                        // Sort ascending by combined_dist; best (lowest) first
                        std::sort(frontier.begin(), frontier.end(),
                                  [](const SearchState& a, const SearchState& b) {
                                      return a.combined_dist < b.combined_dist;
                                  });
                        frontier.resize(config.beam_width);
                        std::make_heap(frontier.begin(), frontier.end());
                    }
                }
            }
        }

        // Sort all valid candidates by vector distance, return top-k
        std::sort(all_candidates.begin(), all_candidates.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.second < b.second;
                  });
        if (static_cast<int>(all_candidates.size()) > k) {
            all_candidates.resize(k);
        }
        return all_candidates;
    }

    // Baseline search: standard HNSW + post-filter
    // For comparison purposes
    template <typename HnswIndex, typename DistanceFunc>
    std::vector<SearchResult>
    SearchBaseline(const HnswIndex& index, const float* query, int k,
                   const FilterDistanceCalculator& filter_calc, const DistanceFunc& dist_func,
                   const Config& config, Metrics* metrics = nullptr) {
        std::vector<SearchResult> results;
        results.reserve(k);

        std::vector<SearchState> frontier;
        frontier.reserve(config.ef_search + 1);
        std::unordered_set<int64_t> visited;

        Metrics local_metrics;
        Metrics* m = metrics ? metrics : &local_metrics;

        int64_t entry_point = GetEntryPoint(index);
        if (entry_point < 0) {
            return results;
        }

        // Initialize - only use vector distance for ranking
        {
            DistanceType entry_dist = dist_func(query, entry_point);
            m->distance_computations++;

            frontier.push_back({entry_point, entry_dist, 0, entry_dist});
            std::push_heap(frontier.begin(), frontier.end());
        }

        // Track all candidates (not just matched ones)
        std::vector<SearchResult> all_candidates;
        all_candidates.reserve(config.ef_search * 2);

        while (!frontier.empty() && static_cast<int>(visited.size()) < config.max_visits) {
            std::pop_heap(frontier.begin(), frontier.end());
            SearchState current = frontier.back();
            frontier.pop_back();

            if (visited.count(current.node_id)) {
                continue;
            }
            visited.insert(current.node_id);
            m->nodes_visited++;

            // Check filter and add to candidates
            bool matches = filter_calc.Match(current.node_id);
            m->filter_checks++;

            if (matches) {
                all_candidates.push_back({current.node_id, current.vector_dist});
                m->valid_visits++;
            }

            // Expand neighbors (using only vector distance)
            auto neighbors = GetNeighbors(index, current.node_id);
            for (int64_t neighbor : neighbors) {
                if (visited.count(neighbor)) {
                    continue;
                }

                DistanceType dist = dist_func(query, neighbor);
                m->distance_computations++;

                if (static_cast<int>(frontier.size()) < config.ef_search ||
                    dist < GetWorstDistance(frontier)) {
                    frontier.push_back({neighbor, dist, 0, dist});
                    std::push_heap(frontier.begin(), frontier.end());
                }
            }
        }

        // Sort and filter results
        std::sort(all_candidates.begin(), all_candidates.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.second < b.second;
                  });

        // Return top-k
        int count = std::min(k, static_cast<int>(all_candidates.size()));
        results.assign(all_candidates.begin(), all_candidates.begin() + count);

        return results;
    }

 private:
    // Helper to get entry point from index
    template <typename HnswIndex>
    int64_t
    GetEntryPoint(const HnswIndex& index) const {
        if (index.ntotal == 0) {
            return -1;
        }
        return index.hnsw.entry_point;  // faiss HNSW structure
    }

    // Helper to get neighbors from index
    template <typename HnswIndex>
    std::vector<int64_t>
    GetNeighbors(const HnswIndex& index, int64_t node_id) const {
        std::vector<int64_t> neighbors;

        const auto& hnsw = index.hnsw;
        int level = 0;  // Use level 0 for search

        size_t begin, end;
        hnsw.neighbor_range(node_id, level, &begin, &end);

        for (size_t i = begin; i < end; i++) {
            int32_t neighbor = hnsw.neighbors[i];
            if (neighbor != -1) {  // -1 indicates empty slot
                neighbors.push_back(neighbor);
            }
        }

        return neighbors;
    }

    // Get worst (highest) combined distance in frontier.
    // O(n) scan over the vector — acceptable since frontier is bounded to beam_width*2.
    DistanceType
    GetWorstCombinedDistance(const std::vector<SearchState>& frontier) const {
        if (frontier.empty()) {
            return std::numeric_limits<DistanceType>::max();
        }
        DistanceType worst = std::numeric_limits<DistanceType>::lowest();
        for (const auto& s : frontier) {
            if (s.combined_dist > worst) {
                worst = s.combined_dist;
            }
        }
        return worst;
    }

    // Get worst (highest) vector distance in frontier.
    DistanceType
    GetWorstDistance(const std::vector<SearchState>& frontier) const {
        if (frontier.empty()) {
            return std::numeric_limits<DistanceType>::max();
        }
        DistanceType worst = std::numeric_limits<DistanceType>::lowest();
        for (const auto& s : frontier) {
            if (s.vector_dist > worst) {
                worst = s.vector_dist;
            }
        }
        return worst;
    }
};

}  // namespace knowhere

#endif  // KNOWHERE_INDEX_HNSW_JAG_HNSW_SEARCHER_H_
