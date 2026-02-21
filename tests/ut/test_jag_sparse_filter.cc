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

TEST_CASE("JAG Sparse Filter Basic", "[jag][sparse]") {
    // Create a simple sparse filter set manually
    knowhere::SparseFilterSet filter_set;

    // Point 0: labels {0, 1}
    filter_set.labels_per_point.push_back({0, 1});
    // Point 1: labels {1, 2}
    filter_set.labels_per_point.push_back({1, 2});
    // Point 2: labels {0, 2, 3}
    filter_set.labels_per_point.push_back({0, 2, 3});
    // Point 3: labels {}
    filter_set.labels_per_point.push_back({});
    // Point 4: labels {0}
    filter_set.labels_per_point.push_back({0});

    // Build inverse index
    filter_set.label_to_points[0] = {0, 2, 4};
    filter_set.label_to_points[1] = {0, 1};
    filter_set.label_to_points[2] = {1, 2};
    filter_set.label_to_points[3] = {2};

    filter_set.ComputeIDFWeights();

    SECTION("IDF Weight Calculation") {
        // IDF(label) = log(N / df)
        // N = 5 points
        // df(0) = 3, df(1) = 2, df(2) = 2, df(3) = 1
        CHECK(filter_set.num_points == 5);

        // Label 0 appears in 3 points -> IDF = log(5/3) ≈ 0.51
        float idf0 = filter_set.GetIDF(0);
        CHECK(idf0 > 0.0f);

        // Label 3 appears in 1 point -> IDF = log(5/1) ≈ 1.61 (highest)
        float idf3 = filter_set.GetIDF(3);
        CHECK(idf3 > idf0);

        LOG_KNOWHERE_INFO_ << "IDF weights: 0=" << idf0 << ", 3=" << idf3;
    }

    SECTION("Single Label Query") {
        // Query: must have label 0
        knowhere::SparseFilterConstraint constraint({0});
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        // Points 0, 2, 4 have label 0
        CHECK(sparse_filter.Match(0) == true);
        CHECK(sparse_filter.Match(1) == false);
        CHECK(sparse_filter.Match(2) == true);
        CHECK(sparse_filter.Match(3) == false);
        CHECK(sparse_filter.Match(4) == true);

        // Distance should be 0 for matching points
        CHECK(sparse_filter.Calculate(0) == 0);
        CHECK(sparse_filter.Calculate(2) == 0);
        CHECK(sparse_filter.Calculate(4) == 0);

        // Non-matching points should have positive distance
        CHECK(sparse_filter.Calculate(1) > 0);
        CHECK(sparse_filter.Calculate(3) > 0);

        // Filter ratio should be 3/5 = 0.6
        CHECK(std::abs(sparse_filter.FilterRatio() - 0.6f) < 0.01f);
    }

    SECTION("Multiple Labels Query (AND)") {
        // Query: must have labels {0, 1} (both)
        knowhere::SparseFilterConstraint constraint({0, 1});
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        // Only point 0 has both labels 0 and 1
        CHECK(sparse_filter.Match(0) == true);
        CHECK(sparse_filter.Match(1) == false);  // has 1 but not 0
        CHECK(sparse_filter.Match(2) == false);  // has 0 but not 1
        CHECK(sparse_filter.Match(3) == false);
        CHECK(sparse_filter.Match(4) == false);  // has 0 but not 1

        // Filter ratio should be 1/5 = 0.2
        CHECK(std::abs(sparse_filter.FilterRatio() - 0.2f) < 0.01f);
    }

    SECTION("Partial Match Query") {
        // Query: at least 1 of labels {0, 3}
        knowhere::SparseFilterConstraint constraint({0, 3}, 1);  // min_matches = 1
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        // Points with label 0: 0, 2, 4
        // Points with label 3: 2
        // Combined: 0, 2, 4
        CHECK(sparse_filter.Match(0) == true);   // has 0
        CHECK(sparse_filter.Match(1) == false);  // has neither
        CHECK(sparse_filter.Match(2) == true);   // has both 0 and 3
        CHECK(sparse_filter.Match(3) == false);  // has neither
        CHECK(sparse_filter.Match(4) == true);   // has 0
    }

    SECTION("IDF-Weighted Distance") {
        // Query: must have label 3 (rare, high IDF)
        knowhere::SparseFilterConstraint constraint({3});
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        // Point 2 has label 3 -> distance = 0
        CHECK(sparse_filter.Calculate(2) == 0);

        // Other points don't have label 3 -> distance = IDF(3)
        int dist1 = sparse_filter.Calculate(1);
        int dist3 = sparse_filter.Calculate(3);

        CHECK(dist1 > 0);
        CHECK(dist3 > 0);
        // Both should have similar distance since both are missing the same label
        CHECK(dist1 == dist3);
    }
}

TEST_CASE("JAG Sparse Filter Random", "[jag][sparse]") {
    const int64_t n = 1000;
    const int num_labels = 100;
    const int avg_labels_per_point = 5;

    auto filter_set = knowhere::GenerateRandomSparseLabels(n, num_labels, avg_labels_per_point, 12345);

    SECTION("Filter Set Statistics") {
        CHECK(static_cast<int64_t>(filter_set.Size()) == n);

        // Check that IDF weights are computed
        int labels_with_idf = 0;
        for (const auto& [label, idf] : filter_set.idf_weights) {
            if (idf > 0.0f) {
                labels_with_idf++;
            }
        }
        CHECK(labels_with_idf > 0);

        LOG_KNOWHERE_INFO_ << "Number of unique labels: " << filter_set.label_to_points.size();
        LOG_KNOWHERE_INFO_ << "Number of IDF weights: " << labels_with_idf;
    }

    SECTION("Single Label Filtering") {
        // Test filtering with label 0
        knowhere::SparseFilterConstraint constraint({0});
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        float filter_ratio = sparse_filter.FilterRatio();
        LOG_KNOWHERE_INFO_ << "Filter ratio for label 0: " << filter_ratio;

        // Count matches manually
        int matches = 0;
        for (int64_t i = 0; i < n; i++) {
            if (sparse_filter.Match(i)) {
                matches++;
                CHECK(sparse_filter.Calculate(i) == 0);
            } else {
                CHECK(sparse_filter.Calculate(i) > 0);
            }
        }

        float expected_ratio = static_cast<float>(matches) / n;
        CHECK(std::abs(filter_ratio - expected_ratio) < 0.01f);
    }

    SECTION("Multiple Labels Filtering") {
        // Test filtering with multiple labels
        std::vector<int32_t> query_labels = {0, 1, 2};
        knowhere::SparseFilterConstraint constraint(query_labels);
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        // Count matches
        int matches = 0;
        for (int64_t i = 0; i < n; i++) {
            if (sparse_filter.Match(i)) {
                matches++;
                CHECK(sparse_filter.Calculate(i) == 0);
            }
        }

        LOG_KNOWHERE_INFO_ << "Filter ratio for labels {0,1,2}: " << sparse_filter.FilterRatio();
        LOG_KNOWHERE_INFO_ << "Matches: " << matches << " / " << n;
    }
}

TEST_CASE("JAG Sparse Filter Benchmark", "[jag][sparse][benchmark]") {
    const int64_t n = 10000;
    const int num_labels = 1000;
    const int avg_labels_per_point = 10;

    auto filter_set = knowhere::GenerateRandomSparseLabels(n, num_labels, avg_labels_per_point, 99999);

    std::cout << "\n========================================\n";
    std::cout << "JAG Sparse Filter Benchmark\n";
    std::cout << "Dataset: " << n << " points, " << num_labels << " unique labels\n";
    std::cout << "Average labels per point: " << avg_labels_per_point << "\n";
    std::cout << "========================================\n\n";

    // Test different query configurations
    struct TestCase {
        std::vector<int32_t> labels;
        int min_matches;
        std::string name;
    };

    std::vector<TestCase> test_cases = {
        {{0}, -1, "single_common"},
        {{999}, -1, "single_rare"},
        {{0, 1, 2}, -1, "multi_all"},
        {{0, 1, 2, 3, 4}, 2, "partial_5_labels"},
        {{0, 1, 2, 3, 4}, -1, "all_5_labels"},
    };

    std::cout << "Query              | Filter% | Avg_Match_Labels | Notes\n";
    std::cout << "--------------------------------------------------------------\n";

    for (const auto& tc : test_cases) {
        knowhere::SparseFilterConstraint constraint(tc.labels, tc.min_matches);
        knowhere::SparseFilterDistance sparse_filter(filter_set, constraint);

        float filter_ratio = sparse_filter.FilterRatio();

        // Calculate average number of matched labels per point
        float avg_matched = 0.0f;
        int matched_count = 0;
        for (int64_t i = 0; i < n; i++) {
            const auto& labels = filter_set.GetLabels(i);
            int matched = 0;
            for (int32_t req_label : tc.labels) {
                for (int32_t l : labels) {
                    if (l == req_label) {
                        matched++;
                        break;
                    }
                }
            }
            avg_matched += matched;
            if (sparse_filter.Match(i)) {
                matched_count++;
            }
        }
        avg_matched /= n;

        std::string query_str;
        if (tc.min_matches == -1) {
            query_str = "ALL of " + std::to_string(tc.labels.size());
        } else {
            query_str = std::to_string(tc.min_matches) + " of " + std::to_string(tc.labels.size());
        }

        std::cout << std::setw(18) << std::left << query_str << " | "
                  << std::fixed << std::setprecision(1) << (filter_ratio * 100) << "%   | "
                  << std::setprecision(2) << avg_matched << "              | "
                  << tc.name << "\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Key Insights:\n";
    std::cout << "- Rare labels (high IDF) contribute more to distance\n";
    std::cout << "- Partial match queries have higher filter ratios\n";
    std::cout << "- JAG can prioritize points matching rare labels\n";
    std::cout << "========================================\n";
}

TEST_CASE("JAG Sparse Filter vs Label Filter", "[jag][sparse][comparison]") {
    const int64_t n = 500;
    const int num_labels = 10;

    // Create both Label and Sparse filter sets
    auto label_set = knowhere::GenerateRandomLabels(n, num_labels, 42);

    knowhere::SparseFilterSet sparse_set;
    sparse_set.labels_per_point.resize(n);
    for (int64_t i = 0; i < n; i++) {
        sparse_set.labels_per_point[i].push_back(label_set.labels[i]);
    }
    // Build inverse index
    for (int64_t i = 0; i < n; i++) {
        int32_t label = label_set.labels[i];
        sparse_set.label_to_points[label].push_back(static_cast<int32_t>(i));
    }
    sparse_set.ComputeIDFWeights();

    SECTION("Equivalent Filtering") {
        // Both should give same results for single-label queries
        int32_t target_label = 5;

        knowhere::LabelFilterConstraint label_constraint{target_label};
        knowhere::LabelFilterDistance label_filter(label_set, label_constraint);

        knowhere::SparseFilterConstraint sparse_constraint({target_label});
        knowhere::SparseFilterDistance sparse_filter(sparse_set, sparse_constraint);

        // Check filter ratios match
        CHECK(std::abs(label_filter.FilterRatio() - sparse_filter.FilterRatio()) < 0.01f);

        // Check individual matches
        for (int64_t i = 0; i < n; i++) {
            CHECK(label_filter.Match(i) == sparse_filter.Match(i));
        }
    }

    SECTION("Sparse Advantage: Multiple Labels") {
        // Sparse filter can handle multiple labels, Label filter cannot
        knowhere::SparseFilterConstraint constraint({0, 1, 2}, 1);  // Any of 3 labels
        knowhere::SparseFilterDistance sparse_filter(sparse_set, constraint);

        // Count matches
        int matches = 0;
        for (int64_t i = 0; i < n; i++) {
            if (sparse_filter.Match(i)) {
                matches++;
            }
        }

        float ratio = static_cast<float>(matches) / n;
        LOG_KNOWHERE_INFO_ << "Sparse filter ratio (any of 3 labels): " << ratio;

        // Should be higher than any single label ratio
        CHECK(ratio >= label_set.GetFilterRatio(0));
    }
}
