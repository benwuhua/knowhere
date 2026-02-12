# JAG-HNSW Minimal Validation Design

## Overview

This document describes a minimal validation implementation of JAG (Join-And-AGgregate) search strategy on Knowhere's HNSW index for Label filtering scenarios.

## Background

### Problem

In filtered vector search, traditional approaches use post-filtering: search by vector similarity first, then filter by scalar attributes. This becomes inefficient when filter ratio is high (most results get filtered out).

### JAG Approach

JAG uses **filter distance** to guide graph search, prioritizing nodes that are both vector-similar AND likely to pass the filter. This reduces wasted visits to filtered-out nodes.

```
Native HNSW:
  Ranking = Vector Distance
  Filter = Check bitset after visiting

JAG-HNSW:
  Ranking = Vector Distance + Weight * Filter Distance
  Effect = Explore "filter-close" nodes first
```

## Goals

1. Validate JAG search strategy effectiveness on HNSW
2. Compare with baseline (HNSW + post-filter) on:
   - Recall@K
   - QPS
   - Nodes visited
   - Valid visit ratio
3. Minimal code changes - no graph structure modification

## Architecture

### Module Structure

```
knowhere/
├── src/common/
│   └── filter_distance.h         # [NEW] Filter distance calculation
│
├── src/index/hnsw/
│   ├── faiss_hnsw.cc             # Existing HNSW
│   └── jag_hnsw_searcher.h       # [NEW] JAG search strategy
│
└── tests/ut/
    └── test_jag_hnsw.cc          # [NEW] Validation tests
```

### Data Flow

```
                 +------------------+
                 |   SIFT1M Data    |
                 +--------+---------+
                          |
                          v
                 +--------+---------+
                 |  Generate Labels |
                 |  (10 categories) |
                 +--------+---------+
                          |
          +---------------+---------------+
          |                               |
          v                               v
+---------+----------+          +---------+----------+
|  Build HNSW Index  |          |  Query Constraints |
|  (existing code)   |          |  (target labels)   |
+---------+----------+          +---------+----------+
          |                               |
          +---------------+---------------+
                          |
                          v
          +---------------+---------------+
          |                               |
          v                               v
+---------+----------+          +---------+----------+
| Baseline Search    |          | JAG Search         |
| (post-filter)      |          | (filter-guided)    |
+---------+----------+          +---------+----------+
          |                               |
          +---------------+---------------+
                          |
                          v
                 +--------+---------+
                 | Compare Metrics  |
                 | - Recall         |
                 | - QPS            |
                 | - Visit Count    |
                 +------------------+
```

## Component Design

### 1. Filter Distance Module

**File**: `src/common/filter_distance.h`

```cpp
// Filter information storage
struct LabelFilterSet {
    std::vector<int32_t> labels;                           // label per point
    std::unordered_map<int32_t, std::vector<int32_t>> label_to_ids;

    int32_t GetLabel(int64_t id) const;
    const std::vector<int32_t>& GetIdsByLabel(int32_t label) const;
    float GetFilterRatio(int32_t label) const;
};

// Query constraint
struct LabelFilterConstraint {
    int32_t target_label;

    int Distance(int32_t point_label) const {
        return (point_label == target_label) ? 0 : 1;
    }
};

// Generic interface for future extension
class FilterDistanceCalculator {
public:
    virtual ~FilterDistanceCalculator() = default;
    virtual int Calculate(int64_t point_id) const = 0;
    virtual bool Match(int64_t point_id) const = 0;
};

// Label filter implementation
class LabelFilterDistance : public FilterDistanceCalculator {
    const LabelFilterSet& filter_set_;
    LabelFilterConstraint constraint_;

public:
    int Calculate(int64_t point_id) const override;
    bool Match(int64_t point_id) const override;
};
```

### 2. JAG Search Strategy

**File**: `src/index/hnsw/jag_hnsw_searcher.h`

```cpp
template <typename DistanceType>
class JAGHnswSearcher {
public:
    struct Config {
        int beam_width = 16;
        float filter_weight = 1.0f;
        int max_visits = 10000;
    };

    struct SearchState {
        int64_t node_id;
        DistanceType vector_dist;
        int filter_dist;
        DistanceType combined_dist;

        bool operator<(const SearchState& other) const;
    };

    std::vector<std::pair<int64_t, DistanceType>> Search(
        const faiss::HNSW& hnsw,
        const float* query,
        int k,
        const FilterDistanceCalculator& filter_calc,
        const Config& config);

private:
    DistanceType ComputeDistance(const float* query, int64_t node_id);
    std::vector<int64_t> GetNeighbors(const faiss::HNSW& hnsw, int64_t node_id);
};
```

**Search Algorithm**:

1. Initialize frontier with entry point
2. For each node in frontier (sorted by combined distance):
   - If filter matches, add to results
   - Expand neighbors, compute combined distance
   - Prune if combined distance exceeds threshold
3. Return top-k matched results

### 3. Test & Evaluation

**File**: `tests/ut/test_jag_hnsw.cc`

**Test Data Generation**:
- SIFT1M: 1M vectors, 128 dimensions
- Labels: 10 categories, randomly assigned
- Queries: Select queries with target labels
- Ground Truth: Brute-force with filter constraint

**Metrics**:

```cpp
struct JAGMetrics {
    float recall_at_k;
    double qps;
    int64_t avg_nodes_visited;
    int64_t avg_valid_nodes;
    float valid_visit_ratio;
};
```

**Test Scenarios**:
- Filter ratios: 10%, 30%, 50%, 70%, 90%
- K values: 10, 100
- Compare: Baseline vs JAG with different filter_weight

## Implementation Plan

### File Changes

```
NEW FILES:
├── src/common/filter_distance.h         (~150 lines)
├── src/index/hnsw/jag_hnsw_searcher.h   (~200 lines)
├── tests/ut/test_jag_hnsw.cc            (~400 lines)
└── python/generate_filter_data.py       (~50 lines)

MODIFIED FILES:
├── CMakeLists.txt                       (add new source)
└── tests/ut/CMakeLists.txt              (add test)
```

### Steps

| Step | Task | Description |
|------|------|-------------|
| 1 | Filter Distance | Implement `filter_distance.h` with Label support |
| 2 | JAG Searcher | Implement `jag_hnsw_searcher.h` with beam search |
| 3 | Test Framework | Implement `test_jag_hnsw.cc` with data generation |
| 4 | Integration | Build and run unit tests |
| 5 | Benchmark | Run comparison at different filter ratios |
| 6 | Analysis | Tune parameters, document results |

## Expected Results

### Hypothesis

JAG should show significant improvement in **valid visit ratio** when filter ratio is high (50%+). This translates to:
- Fewer nodes visited for same recall
- Better scalability for selective filters

### Output Format

```
=== Filter Ratio: 50% ===
Metric                  | Baseline    | JAG-HNSW    | Change
------------------------|-------------|-------------|--------
Recall@10               | 0.850       | 0.855       | +0.6%
QPS                     | 1250        | 1180        | -5.6%
Nodes Visited           | 450         | 280         | -37.8%
Valid Visit Ratio       | 52%         | 78%         | +50%
```

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Effect not significant | Tune filter_weight, try Threshold-JAG variant |
| Performance regression | Focus on visit efficiency, not raw QPS |
| Build issues | Minimal changes, new files only |

## Future Work

If validation is successful:

1. **Full JAG Implementation**: Graph building with filter-aware prune
2. **More Filter Types**: Range, Sparse, Subset
3. **DiskANN Integration**: Apply JAG to disk-based search
4. **Production Integration**: Integrate with Milvus scalar filtering

## References

- JAG Paper: Filtered Vector Search Benchmark
- JAG Code: `/Users/ryan/Code/Paper/JAG/`
- Knowhere HNSW: `src/index/hnsw/faiss_hnsw.cc`
- Knowhere Filtering: `include/knowhere/bitsetview.h`
