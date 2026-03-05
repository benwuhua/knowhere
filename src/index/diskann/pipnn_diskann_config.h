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

#ifndef PIPNN_DISKANN_CONFIG_H
#define PIPNN_DISKANN_CONFIG_H

#include "diskann_config.h"

namespace knowhere {

class PiPNNDiskANNConfig : public DiskANNConfig {
 public:
    // RBC partition parameters
    CFG_INT pipnn_leaf_max_size;
    CFG_INT pipnn_fanout_l1;
    CFG_INT pipnn_fanout_l2;

    // HashPrune parameters
    CFG_INT pipnn_k_nn;
    CFG_INT pipnn_hash_bits;

    // RobustPrune parameter
    CFG_BOOL pipnn_final_prune;

    KNOHWERE_DECLARE_CONFIG(PiPNNDiskANNConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_leaf_max_size)
            .description("RBC partition leaf max size")
            .set_default(1000)
            .set_range(100, 10000)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_fanout_l1)
            .description("RBC partition fanout at level 1")
            .set_default(10)
            .set_range(1, 50)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_fanout_l2)
            .description("RBC partition fanout at level 2")
            .set_default(3)
            .set_range(1, 20)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_k_nn)
            .description("HashPrune k nearest neighbors")
            .set_default(3)
            .set_range(1, 10)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_hash_bits)
            .description("HashPrune hash bits")
            .set_default(12)
            .set_range(6, 16)
            .for_train();

        KNOWHERE_CONFIG_DECLARE_FIELD(pipnn_final_prune)
            .description("enable final RobustPrune")
            .set_default(true)
            .for_train();
    }
};

}  // namespace knowhere

#endif /* PIPNN_DISKANN_CONFIG_H */
