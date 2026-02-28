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

#ifndef KNOWHERE_COMMON_FILTER_DISTANCE_H_
#define KNOWHERE_COMMON_FILTER_DISTANCE_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace knowhere {

// Label filter set - stores label information for all points
struct LabelFilterSet {
    std::vector<int32_t> labels;                              // label for each point
    std::unordered_map<int32_t, std::vector<int32_t>> label_to_ids;  // label -> point IDs

    int32_t
    GetLabel(int64_t id) const {
        return labels[id];
    }

    const std::vector<int32_t>&
    GetIdsByLabel(int32_t label) const {
        static const std::vector<int32_t> empty;
        auto it = label_to_ids.find(label);
        return (it != label_to_ids.end()) ? it->second : empty;
    }

    float
    GetFilterRatio(int32_t label) const {
        auto it = label_to_ids.find(label);
        if (it == label_to_ids.end()) {
            return 0.0f;
        }
        return static_cast<float>(it->second.size()) / labels.size();
    }

    size_t
    Size() const {
        return labels.size();
    }

    size_t
    NumLabels() const {
        return label_to_ids.size();
    }
};

// Label filter constraint for queries
struct LabelFilterConstraint {
    int32_t target_label;

    // Calculate filter distance: 0 = match, 1 = no match
    int
    Distance(int32_t point_label) const {
        return (point_label == target_label) ? 0 : 1;
    }

    bool
    Match(int32_t point_label) const {
        return point_label == target_label;
    }
};

// Generic filter distance calculator interface
// Can be extended for Range, Sparse, Subset filters
class FilterDistanceCalculator {
 public:
    virtual ~FilterDistanceCalculator() = default;

    // Calculate filter distance for a point
    // 0 = exact match, larger = more distant
    virtual int
    Calculate(int64_t point_id) const = 0;

    // Check if point matches the filter exactly
    virtual bool
    Match(int64_t point_id) const = 0;

    // Get filter ratio (fraction of points that match)
    virtual float
    FilterRatio() const = 0;
};

// Label filter distance implementation
class LabelFilterDistance : public FilterDistanceCalculator {
 public:
    LabelFilterDistance(const LabelFilterSet& filter_set, const LabelFilterConstraint& constraint)
        : filter_set_(filter_set), constraint_(constraint) {
    }

    int
    Calculate(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return 1;  // Invalid ID treated as non-matching
        }
        return constraint_.Distance(filter_set_.GetLabel(point_id));
    }

    bool
    Match(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return false;
        }
        return constraint_.Match(filter_set_.GetLabel(point_id));
    }

    float
    FilterRatio() const override {
        return filter_set_.GetFilterRatio(constraint_.target_label);
    }

    int32_t
    TargetLabel() const {
        return constraint_.target_label;
    }

 private:
    const LabelFilterSet& filter_set_;
    LabelFilterConstraint constraint_;
};

// Helper function to generate random labels for testing
inline LabelFilterSet
GenerateRandomLabels(int64_t num_points, int num_labels, int seed = 42) {
    LabelFilterSet filter_set;
    filter_set.labels.resize(num_points);

    // Simple deterministic random number generator
    uint64_t state = seed;
    auto next_random = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int32_t>((state >> 33) & 0x7FFFFFFF);
    };

    for (int64_t i = 0; i < num_points; i++) {
        int32_t label = next_random() % num_labels;
        filter_set.labels[i] = label;
        filter_set.label_to_ids[label].push_back(static_cast<int32_t>(i));
    }

    return filter_set;
}

// ============================================================================
// Range Filter Support (JAG paper Section 3.2)
// ============================================================================

// Range filter set - stores numeric values for all points
// Used for filtering based on scalar attributes (e.g., price, timestamp, rating)
struct RangeFilterSet {
    std::vector<float> values;  // numeric value for each point
    float min_val = 0.0f;       // minimum value in the dataset
    float max_val = 0.0f;       // maximum value in the dataset

    float
    GetValue(int64_t id) const {
        return values[id];
    }

    size_t
    Size() const {
        return values.size();
    }

    float
    Range() const {
        return max_val - min_val;
    }

    // Compute statistics for normalization
    void
    ComputeStats() {
        if (values.empty()) {
            return;
        }
        min_val = values[0];
        max_val = values[0];
        for (float v : values) {
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
    }
};

// Range filter constraint for queries
// Defines the acceptable range [low, high) for filtering
struct RangeFilterConstraint {
    float range_low;   // inclusive lower bound
    float range_high;  // exclusive upper bound

    // Calculate filter distance:
    // - 0 if value is in range [low, high)
    // - Proportional distance to nearest boundary if outside
    // JAG paper: "distance to the range boundary"
    float
    Distance(float value) const {
        if (value >= range_low && value < range_high) {
            return 0.0f;  // Inside range
        }
        if (value < range_low) {
            return range_low - value;  // Distance below range
        }
        return value - range_high;  // Distance above range
    }

    bool
    Match(float value) const {
        return value >= range_low && value < range_high;
    }

    // Get the width of the acceptable range
    float
    Width() const {
        return range_high - range_low;
    }
};

// Range filter distance implementation for JAG
// Key difference from Label: distance is proportional, not binary
class RangeFilterDistance : public FilterDistanceCalculator {
 public:
    RangeFilterDistance(const RangeFilterSet& filter_set, const RangeFilterConstraint& constraint,
                        float normalization_scale = 0.0f)
        : filter_set_(filter_set), constraint_(constraint) {
        // Auto-compute normalization scale if not provided
        if (normalization_scale > 0.0f) {
            scale_ = normalization_scale;
        } else {
            // Use the data range as normalization scale
            // This normalizes distances to [0, 1] range roughly
            scale_ = filter_set_.Range();
            if (scale_ < 1e-6f) {
                scale_ = 1.0f;  // Avoid division by zero
            }
        }

        // Pre-compute filter ratio
        int64_t matches = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(filter_set_.Size()); i++) {
            if (constraint_.Match(filter_set_.GetValue(i))) {
                matches++;
            }
        }
        filter_ratio_ = static_cast<float>(matches) / filter_set_.Size();
    }

    // Returns integer distance (scaled for JAG compatibility)
    // 0 = in range, larger = further from range
    int
    Calculate(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return 1;  // Invalid ID
        }
        float value = filter_set_.GetValue(point_id);

        // Use Match() to determine if in range (not Distance())
        // This handles boundary cases correctly
        if (constraint_.Match(value)) {
            return 0;  // In range
        }

        // Outside range - calculate proportional distance
        float dist = constraint_.Distance(value);

        // Normalize and convert to integer scale
        // Scale to integer range [1, 1000] for reasonable weight interaction
        int int_dist = static_cast<int>(dist / scale_ * 100.0f) + 1;
        return int_dist;
    }

    bool
    Match(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return false;
        }
        return constraint_.Match(filter_set_.GetValue(point_id));
    }

    float
    FilterRatio() const override {
        return filter_ratio_;
    }

    float
    GetRangeLow() const {
        return constraint_.range_low;
    }

    float
    GetRangeHigh() const {
        return constraint_.range_high;
    }

    float
    GetScale() const {
        return scale_;
    }

 private:
    const RangeFilterSet& filter_set_;
    RangeFilterConstraint constraint_;
    float scale_;
    float filter_ratio_;
};

// Helper function to generate random range values for testing
inline RangeFilterSet
GenerateRandomRangeValues(int64_t num_points, float min_val = 0.0f, float max_val = 100.0f, int seed = 42) {
    RangeFilterSet filter_set;
    filter_set.values.resize(num_points);
    filter_set.min_val = min_val;
    filter_set.max_val = max_val;

    // Simple deterministic random number generator
    uint64_t state = seed;
    auto next_random = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>((state >> 33) & 0x7FFFFFFF) / 0x7FFFFFFF;
    };

    float range = max_val - min_val;
    for (int64_t i = 0; i < num_points; i++) {
        filter_set.values[i] = min_val + next_random() * range;
    }

    return filter_set;
}

// ============================================================================
// Sparse Filter Support (JAG paper Section 3.3 - IDF weighted sparse labels)
// ============================================================================

// Sparse filter set - stores multiple labels per point (e.g., tags, keywords)
// Each point can have zero or more labels
struct SparseFilterSet {
    // labels_per_point[i] = list of labels for point i
    std::vector<std::vector<int32_t>> labels_per_point;

    // label_to_points[label] = list of points that have this label
    std::unordered_map<int32_t, std::vector<int32_t>> label_to_points;

    // IDF weights: idf[label] = log(N / df[label]) where N = total points, df = document frequency
    std::unordered_map<int32_t, float> idf_weights;

    int64_t num_points = 0;

    const std::vector<int32_t>&
    GetLabels(int64_t id) const {
        static const std::vector<int32_t> empty;
        if (id < 0 || id >= static_cast<int64_t>(labels_per_point.size())) {
            return empty;
        }
        return labels_per_point[id];
    }

    size_t
    Size() const {
        return labels_per_point.size();
    }

    bool
    HasLabel(int64_t id, int32_t label) const {
        const auto& labels = GetLabels(id);
        for (int32_t l : labels) {
            if (l == label) return true;
        }
        return false;
    }

    // Compute IDF weights based on document frequency
    void
    ComputeIDFWeights() {
        num_points = static_cast<int64_t>(labels_per_point.size());
        if (num_points == 0) return;

        for (const auto& [label, points] : label_to_points) {
            float df = static_cast<float>(points.size());
            // Standard IDF formula: log(N / df)
            idf_weights[label] = std::log(static_cast<float>(num_points) / df);
        }
    }

    float
    GetIDF(int32_t label) const {
        auto it = idf_weights.find(label);
        return (it != idf_weights.end()) ? it->second : 0.0f;
    }

    float
    GetFilterRatio(int32_t label) const {
        auto it = label_to_points.find(label);
        if (it == label_to_points.end() || num_points == 0) {
            return 0.0f;
        }
        return static_cast<float>(it->second.size()) / num_points;
    }
};

// Sparse filter constraint for queries
// Defines required labels with optional minimum match count
struct SparseFilterConstraint {
    std::vector<int32_t> required_labels;  // Labels that should be present
    int min_matches;                       // Minimum number of required labels to match (default: all)

    SparseFilterConstraint() : min_matches(-1) {}  // -1 means match all

    SparseFilterConstraint(const std::vector<int32_t>& labels, int min_match = -1)
        : required_labels(labels), min_matches(min_match) {
        if (min_matches < 0) {
            min_matches = static_cast<int>(labels.size());  // Default: match all
        }
    }

    // Calculate filter distance using IDF weighting
    // Distance = sum of IDF weights for missing required labels
    // Lower distance = more labels matched, especially rare ones
    float
    Distance(const std::vector<int32_t>& point_labels, const SparseFilterSet& filter_set) const {
        if (required_labels.empty()) {
            return 0.0f;  // No requirements
        }

        // Create a set of point labels for fast lookup
        std::unordered_set<int32_t> point_label_set(point_labels.begin(), point_labels.end());

        float missing_weight = 0.0f;
        int matched = 0;

        for (int32_t req_label : required_labels) {
            if (point_label_set.count(req_label) > 0) {
                matched++;
            } else {
                // Add IDF weight of missing label
                missing_weight += filter_set.GetIDF(req_label);
            }
        }

        // If minimum matches not met, add penalty
        if (matched < min_matches) {
            missing_weight += static_cast<float>(min_matches - matched) * 2.0f;  // Penalty per missing match
        }

        return missing_weight;
    }

    bool
    Match(const std::vector<int32_t>& point_labels) const {
        if (required_labels.empty()) {
            return true;  // No requirements
        }

        // Create a set of point labels for fast lookup
        std::unordered_set<int32_t> point_label_set(point_labels.begin(), point_labels.end());

        int matched = 0;
        for (int32_t req_label : required_labels) {
            if (point_label_set.count(req_label) > 0) {
                matched++;
            }
        }

        return matched >= min_matches;
    }
};

// Sparse filter distance implementation for JAG
// Uses IDF weighting - rare labels have higher weight
class SparseFilterDistance : public FilterDistanceCalculator {
 public:
    SparseFilterDistance(const SparseFilterSet& filter_set, const SparseFilterConstraint& constraint,
                         float normalization_scale = 0.0f)
        : filter_set_(filter_set), constraint_(constraint) {
        // Compute normalization scale
        if (normalization_scale > 0.0f) {
            scale_ = normalization_scale;
        } else {
            // Use average IDF as normalization scale
            float total_idf = 0.0f;
            int count = 0;
            for (const auto& [label, idf] : filter_set_.idf_weights) {
                total_idf += idf;
                count++;
            }
            scale_ = (count > 0) ? total_idf / count : 1.0f;
            if (scale_ < 1e-6f) {
                scale_ = 1.0f;
            }
        }

        // Pre-compute filter ratio
        int64_t matches = 0;
        for (int64_t i = 0; i < static_cast<int64_t>(filter_set_.Size()); i++) {
            if (constraint_.Match(filter_set_.GetLabels(i))) {
                matches++;
            }
        }
        filter_ratio_ = static_cast<float>(matches) / filter_set_.Size();
    }

    // Returns integer distance (scaled for JAG compatibility)
    // 0 = all required labels matched, larger = more mismatched
    int
    Calculate(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return 1;  // Invalid ID
        }

        const auto& labels = filter_set_.GetLabels(point_id);

        // Check if matched
        if (constraint_.Match(labels)) {
            return 0;  // Matched
        }

        // Calculate IDF-weighted distance
        float dist = constraint_.Distance(labels, filter_set_);

        // Normalize and convert to integer scale
        int int_dist = static_cast<int>(dist / scale_ * 10.0f) + 1;
        return int_dist;
    }

    bool
    Match(int64_t point_id) const override {
        if (point_id < 0 || point_id >= static_cast<int64_t>(filter_set_.Size())) {
            return false;
        }
        return constraint_.Match(filter_set_.GetLabels(point_id));
    }

    float
    FilterRatio() const override {
        return filter_ratio_;
    }

    const std::vector<int32_t>&
    RequiredLabels() const {
        return constraint_.required_labels;
    }

    float
    GetScale() const {
        return scale_;
    }

 private:
    const SparseFilterSet& filter_set_;
    SparseFilterConstraint constraint_;
    float scale_;
    float filter_ratio_;
};

// Helper function to generate random sparse labels for testing
inline SparseFilterSet
GenerateRandomSparseLabels(int64_t num_points, int num_unique_labels, int avg_labels_per_point, int seed = 42) {
    SparseFilterSet filter_set;
    filter_set.labels_per_point.resize(num_points);

    // Simple deterministic random number generator
    uint64_t state = seed;
    auto next_random = [&state]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<int32_t>((state >> 33) & 0x7FFFFFFF);
    };

    // Use Zipf-like distribution for labels (some labels more common)
    // This creates realistic IDF distribution
    for (int64_t i = 0; i < num_points; i++) {
        // Determine number of labels for this point (Poisson-like)
        int num_labels = avg_labels_per_point;
        int variation = next_random() % 3 - 1;  // -1, 0, or 1
        num_labels = std::max(1, num_labels + variation);

        for (int j = 0; j < num_labels; j++) {
            // Zipf-like: lower label IDs are more common
            int32_t label = static_cast<int32_t>(next_random() % next_random() % num_unique_labels);
            label = std::abs(label) % num_unique_labels;

            // Add label if not already present
            bool already_exists = false;
            for (int32_t l : filter_set.labels_per_point[i]) {
                if (l == label) {
                    already_exists = true;
                    break;
                }
            }
            if (!already_exists) {
                filter_set.labels_per_point[i].push_back(label);
                filter_set.label_to_points[label].push_back(static_cast<int32_t>(i));
            }
        }
    }

    filter_set.ComputeIDFWeights();
    return filter_set;
}

}  // namespace knowhere

#endif  // KNOWHERE_COMMON_FILTER_DISTANCE_H_
