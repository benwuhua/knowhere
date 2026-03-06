// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifndef KNOWHERE_INDEX_DISKANN_IMPL_PIPNN_BUILD_PROFILE_H
#define KNOWHERE_INDEX_DISKANN_IMPL_PIPNN_BUILD_PROFILE_H

#include <cstdint>
#include <string_view>

namespace knowhere::pipnn_diskann {

enum class BuildStage {
    kGraphConstruction,
    kPQAndDiskLayout,
    kTotal,
};

inline constexpr std::string_view
BuildStageName(BuildStage stage) {
    switch (stage) {
        case BuildStage::kGraphConstruction:
            return "PiPNN graph construction";
        case BuildStage::kPQAndDiskLayout:
            return "DiskANN PQ/disk layout";
        case BuildStage::kTotal:
            return "PiPNN-DiskANN total build";
    }
    return "unknown";
}

inline constexpr double
NsToMs(int64_t ns) {
    return static_cast<double>(ns) / 1e6;
}

struct BuildProfile {
    int64_t graph_construction_ns = 0;
    int64_t pq_and_disk_layout_ns = 0;

    int64_t
    total_ns() const {
        return graph_construction_ns + pq_and_disk_layout_ns;
    }

    double
    stage_ms(BuildStage stage) const {
        switch (stage) {
            case BuildStage::kGraphConstruction:
                return NsToMs(graph_construction_ns);
            case BuildStage::kPQAndDiskLayout:
                return NsToMs(pq_and_disk_layout_ns);
            case BuildStage::kTotal:
                return NsToMs(total_ns());
        }
        return 0.0;
    }
};

}  // namespace knowhere::pipnn_diskann

#endif  // KNOWHERE_INDEX_DISKANN_IMPL_PIPNN_BUILD_PROFILE_H
