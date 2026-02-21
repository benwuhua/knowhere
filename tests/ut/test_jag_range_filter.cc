// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <catch2/catch_test_macros.hpp>

#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "common/filter_distance.h"
#include "knowhere/log.h"

namespace {

// Generate random range values
std::vector<float>
GenerateRandomValues(int64_t n, float min_val, float max_val, int seed = 42) {
    std::vector<float> values(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(min_val, max_val);
    for (int64_t i = 0; i < n; i++) {
        values[i] = dist(rng);
    }
    return values;
}

}  // namespace

TEST_CASE("JAG Range Filter Basic", "[jag][range]") {
    // Test parameters
    const int64_t n = 5000;

    // Generate range values for filtering (simulating e.g., price, timestamp)
    auto values = GenerateRandomValues(n, 0.0f, 100.0f, 99999);
    knowhere::RangeFilterSet range_set;
    range_set.values = std::move(values);
    range_set.min_val = 0.0f;
    range_set.max_val = 100.0f;

    SECTION("Range Filter Distance Calculation") {
        // Test with different range constraints
        std::vector<std::pair<float, float>> ranges = {
            {0.0f, 10.0f},    // ~10% filter ratio
            {0.0f, 30.0f},    // ~30% filter ratio
            {30.0f, 50.0f},   // ~20% filter ratio (middle range)
            {70.0f, 100.0f},  // ~30% filter ratio (high values)
        };

        for (const auto& [low, high] : ranges) {
            knowhere::RangeFilterConstraint constraint{low, high};
            knowhere::RangeFilterDistance range_filter(range_set, constraint);

            INFO("Range [" << low << ", " << high << ")");
            INFO("Filter ratio: " << range_filter.FilterRatio());

            // Verify filter ratio is reasonable
            CHECK(range_filter.FilterRatio() > 0.0f);
            CHECK(range_filter.FilterRatio() < 1.0f);

            // Verify distance calculation
            for (int64_t i = 0; i < n; i++) {
                float value = range_set.GetValue(i);
                bool in_range = (value >= low && value < high);
                int dist = range_filter.Calculate(i);

                if (in_range) {
                    CHECK(dist == 0);
                } else {
                    CHECK(dist > 0);
                }
                CHECK(range_filter.Match(i) == in_range);
            }
        }
    }

    SECTION("Range Filter vs Label Filter Comparison") {
        // Create equivalent label filter (binary) and range filter (proportional)
        // For comparison, we'll see how the distance differs

        float range_low = 40.0f;
        float range_high = 60.0f;

        knowhere::RangeFilterConstraint constraint{range_low, range_high};
        knowhere::RangeFilterDistance range_filter(range_set, constraint);

        // Count points in range
        int in_range_count = 0;
        for (int64_t i = 0; i < n; i++) {
            if (range_filter.Match(i)) {
                in_range_count++;
            }
        }

        float expected_ratio = static_cast<float>(in_range_count) / n;
        CHECK(std::abs(range_filter.FilterRatio() - expected_ratio) < 0.01f);

        // Compare distance values for points outside range
        int outside_count = 0;
        float total_range_dist = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            if (!range_filter.Match(i)) {
                float value = range_set.GetValue(i);
                float raw_dist = constraint.Distance(value);
                total_range_dist += raw_dist;
                outside_count++;

                // Range filter should give proportional distance
                int scaled_dist = range_filter.Calculate(i);
                CHECK(scaled_dist > 0);
            }
        }

        float avg_range_dist = total_range_dist / outside_count;
        LOG_KNOWHERE_INFO_ << "Average range distance for out-of-range points: " << avg_range_dist;
        LOG_KNOWHERE_INFO_ << "Filter ratio: " << range_filter.FilterRatio();
    }
}

TEST_CASE("JAG Range Filter Performance", "[jag][range][benchmark]") {
    // Test parameters
    const int64_t n = 10000;

    // Generate range values (simulating e.g., product prices 0-1000)
    auto values = GenerateRandomValues(n, 0.0f, 1000.0f, 99999);
    knowhere::RangeFilterSet range_set;
    range_set.values = std::move(values);
    range_set.min_val = 0.0f;
    range_set.max_val = 1000.0f;

    // Test different range constraints
    struct TestCase {
        float low;
        float high;
        std::string name;
    };

    std::vector<TestCase> test_cases = {
        {0.0f, 100.0f, "10%_low"},      // ~10% of data
        {0.0f, 300.0f, "30%_low"},      // ~30% of data
        {350.0f, 650.0f, "30%_mid"},    // ~30% middle
        {0.0f, 500.0f, "50%_low"},      // ~50% of data
        {700.0f, 1000.0f, "30%_high"},  // ~30% high values
    };

    std::cout << "\n========================================\n";
    std::cout << "JAG Range Filter Benchmark\n";
    std::cout << "Dataset: " << n << " points, range [0, 1000)\n";
    std::cout << "========================================\n\n";

    std::cout << "Range        | Filter% | Avg_Range_Dist | Notes\n";
    std::cout << "--------------------------------------------------------------\n";

    for (const auto& tc : test_cases) {
        knowhere::RangeFilterConstraint constraint{tc.low, tc.high};
        knowhere::RangeFilterDistance range_filter(range_set, constraint);

        // Calculate statistics
        float filter_ratio = range_filter.FilterRatio();
        float avg_range_dist = 0.0f;
        int outside_count = 0;

        for (int64_t i = 0; i < n; i++) {
            if (!range_filter.Match(i)) {
                float value = range_set.GetValue(i);
                avg_range_dist += constraint.Distance(value);
                outside_count++;
            }
        }
        if (outside_count > 0) {
            avg_range_dist /= outside_count;
        }

        std::cout << "[" << tc.low << ", " << tc.high << ") | "
                  << std::fixed << std::setprecision(1) << (filter_ratio * 100) << "%   | "
                  << std::setprecision(2) << avg_range_dist << "           | "
                  << tc.name << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "- Range filter distance is proportional to distance from range\n";
    std::cout << "- Points closer to range boundary have smaller filter distance\n";
    std::cout << "- This allows JAG to prioritize near-boundary points\n";
    std::cout << "========================================\n";
}

TEST_CASE("JAG Range Filter Integration", "[jag][range][integration]") {
    // This test demonstrates how Range filter would integrate with JAG search
    // Note: Full integration requires changes to the search pipeline

    const int64_t n = 1000;

    // Generate range values
    auto values = GenerateRandomValues(n, 0.0f, 100.0f, 99999);
    knowhere::RangeFilterSet range_set;
    range_set.values = std::move(values);
    range_set.min_val = 0.0f;
    range_set.max_val = 100.0f;

    // Create range filter
    knowhere::RangeFilterConstraint constraint{30.0f, 70.0f};  // ~40% filter ratio
    knowhere::RangeFilterDistance range_filter(range_set, constraint);

    // The key insight for JAG with Range filter:
    // - Label filter: distance is 0 or 1 (binary)
    // - Range filter: distance is proportional (0 for in-range, >0 for out-of-range)
    // - This means weight tuning needs to account for the scale of range distances

    LOG_KNOWHERE_INFO_ << "Range filter integration test";
    LOG_KNOWHERE_INFO_ << "Range: [30, 70)";
    LOG_KNOWHERE_INFO_ << "Filter ratio: " << range_filter.FilterRatio();
    LOG_KNOWHERE_INFO_ << "Scale: " << range_filter.GetScale();

    // Demonstrate distance gradient
    std::vector<std::pair<int, int>> distance_histogram;
    for (int dist_val = 0; dist_val <= 10; dist_val++) {
        int count = 0;
        for (int64_t i = 0; i < n; i++) {
            if (range_filter.Calculate(i) == dist_val) {
                count++;
            }
        }
        if (count > 0) {
            distance_histogram.push_back({dist_val, count});
        }
    }

    LOG_KNOWHERE_INFO_ << "Distance histogram:";
    for (const auto& [dist, count] : distance_histogram) {
        LOG_KNOWHERE_INFO_ << "  Distance " << dist << ": " << count << " points";
    }

    CHECK(range_filter.FilterRatio() > 0.3f);
    CHECK(range_filter.FilterRatio() < 0.5f);
}

TEST_CASE("JAG Range Filter Edge Cases", "[jag][range]") {
    // Test edge cases for range filter
    const int64_t n = 100;

    // Generate values with some known boundary cases
    std::vector<float> values(n);
    for (int64_t i = 0; i < n; i++) {
        values[i] = static_cast<float>(i);  // 0, 1, 2, ..., 99
    }

    knowhere::RangeFilterSet range_set;
    range_set.values = std::move(values);
    range_set.min_val = 0.0f;
    range_set.max_val = 99.0f;
    range_set.ComputeStats();

    SECTION("Boundary Values") {
        // Test exact boundary values
        knowhere::RangeFilterConstraint constraint{25.0f, 75.0f};
        knowhere::RangeFilterDistance range_filter(range_set, constraint);

        // Check boundary behavior
        CHECK(range_filter.Match(25) == true);   // Exactly at low boundary - should match
        CHECK(range_filter.Match(74) == true);   // Just below high boundary - should match
        CHECK(range_filter.Match(75) == false);  // Exactly at high boundary - should NOT match (exclusive)
        CHECK(range_filter.Match(24) == false);  // Just below low boundary - should NOT match

        // Check distance values
        CHECK(range_filter.Calculate(50) == 0);   // In range
        CHECK(range_filter.Calculate(25) == 0);   // At low boundary
        CHECK(range_filter.Calculate(74) == 0);   // Just below high boundary
        CHECK(range_filter.Calculate(75) > 0);    // At high boundary (exclusive)
        CHECK(range_filter.Calculate(24) > 0);    // Just below low boundary
    }

    SECTION("Full Range") {
        // Test when all points are in range
        knowhere::RangeFilterConstraint constraint{0.0f, 100.0f};
        knowhere::RangeFilterDistance range_filter(range_set, constraint);

        CHECK(range_filter.FilterRatio() == 1.0f);
        for (int64_t i = 0; i < n; i++) {
            CHECK(range_filter.Match(i) == true);
            CHECK(range_filter.Calculate(i) == 0);
        }
    }

    SECTION("Empty Range") {
        // Test when no points are in range
        knowhere::RangeFilterConstraint constraint{100.0f, 200.0f};
        knowhere::RangeFilterDistance range_filter(range_set, constraint);

        CHECK(range_filter.FilterRatio() == 0.0f);
        for (int64_t i = 0; i < n; i++) {
            CHECK(range_filter.Match(i) == false);
            CHECK(range_filter.Calculate(i) > 0);
        }
    }
}
