// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

#pragma once

#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace knowhere::pipnn_diskann {

class VamanaSerializer {
 public:
    static void
    write(const std::vector<std::vector<uint32_t>>& graph, uint32_t entry_point, const std::string& path) {
        uint64_t index_size = kHeaderBytes;
        uint32_t max_degree = 0;

        for (const auto& neighbors : graph) {
            const auto degree = static_cast<uint32_t>(neighbors.size());
            if (degree > max_degree) {
                max_degree = degree;
            }
            index_size += sizeof(uint32_t) + static_cast<uint64_t>(degree) * sizeof(uint32_t);
        }

        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("failed to open Vamana index output: " + path);
        }

        constexpr uint64_t num_frozen_pts = 0;
        out.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
        out.write(reinterpret_cast<const char*>(&max_degree), sizeof(max_degree));
        out.write(reinterpret_cast<const char*>(&entry_point), sizeof(entry_point));
        out.write(reinterpret_cast<const char*>(&num_frozen_pts), sizeof(num_frozen_pts));

        for (const auto& neighbors : graph) {
            const auto degree = static_cast<uint32_t>(neighbors.size());
            out.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
            if (degree > 0) {
                const auto bytes = static_cast<std::streamsize>(static_cast<uint64_t>(degree) * sizeof(uint32_t));
                out.write(reinterpret_cast<const char*>(neighbors.data()), bytes);
            }
        }

        if (!out.good()) {
            throw std::runtime_error("failed to write complete Vamana index: " + path);
        }
    }

    static uint32_t
    find_medoid(const float* data, uint32_t n, uint32_t d) {
        if (data == nullptr || n == 0 || d == 0) {
            return 0;
        }

        std::vector<float> centroid(d, 0.0f);
        const float inv_n = 1.0f / static_cast<float>(n);
        for (uint32_t i = 0; i < n; ++i) {
            const float* point = data + static_cast<size_t>(i) * d;
            for (uint32_t j = 0; j < d; ++j) {
                centroid[j] += point[j] * inv_n;
            }
        }

        uint32_t best_id = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (uint32_t i = 0; i < n; ++i) {
            const float* point = data + static_cast<size_t>(i) * d;
            float dist = 0.0f;
            for (uint32_t j = 0; j < d; ++j) {
                const float diff = point[j] - centroid[j];
                dist += diff * diff;
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_id = i;
            }
        }

        return best_id;
    }

 private:
    static constexpr uint64_t kHeaderBytes =
        sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
};

}  // namespace knowhere::pipnn_diskann
