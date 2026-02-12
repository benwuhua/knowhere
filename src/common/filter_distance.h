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

}  // namespace knowhere

#endif  // KNOWHERE_COMMON_FILTER_DISTANCE_H_
