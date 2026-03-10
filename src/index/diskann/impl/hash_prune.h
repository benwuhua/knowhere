// Copyright (C) 2019-2023 Zilliz. All rights reserved.
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace knowhere::pipnn_diskann {

// HashPrune: history-independent online pruning via residualized LSH.
// Each slot covers one angular cone from point p. Maintains the closest
// candidate per cone, regardless of insertion order (Theorem 3.1, PiPNN paper).
//
// Memory per slot: 8 bytes (4B id + 2B hash + 2B bfloat16 dist)
// Total: 8 * max_degree bytes per point
class HashPrune {
 public:
    enum class InsertDecision {
        kCollisionReplace,
        kCollisionReject,
        kAppend,
        kReservoirReplace,
        kReservoirReject,
    };

    HashPrune(uint32_t dim, uint32_t hash_bits, uint32_t max_degree)
        : dim_(dim), m_(hash_bits), max_degree_(max_degree) {
        reservoir_.resize(max_degree_);
        size_ = 0;
        farthest_idx_ = 0;
        farthest_dist_ = 0.0f;

        // Pre-generate m random hyperplanes (m x dim matrix)
        hyperplanes_.resize(m_ * dim_);
        std::mt19937 rng(/*seed=*/12345 + dim_ * 31 + m_);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& h : hyperplanes_) h = normal(rng);
    }

    // compute_sketch: project vec onto m hyperplanes → m floats
    // Caller allocates sketch[m_]
    void compute_sketch(const float* vec, float* sketch) const {
        for (uint32_t i = 0; i < m_; i++) {
            float dot = 0.0f;
            const float* hp = &hyperplanes_[i * dim_];
            for (uint32_t j = 0; j < dim_; j++) dot += hp[j] * vec[j];
            sketch[i] = dot;
        }
    }

    // residual_hash: m-bit hash of (c - p) using sketch difference
    // sketch_p = compute_sketch(p), sketch_c = compute_sketch(c)

    // Static overload — callable without a HashPrune instance
    static uint16_t residual_hash(const float* sketch_p, const float* sketch_c, uint32_t m) {
        uint16_t h = 0;
        for (uint32_t i = 0; i < m && i < 16; i++) {
            if (sketch_c[i] >= sketch_p[i]) h |= (1u << i);
        }
        return h;
    }

    // Instance method — delegates to static for backward compat
    uint16_t residual_hash(const float* sketch_p, const float* sketch_c) const {
        return residual_hash(sketch_p, sketch_c, m_);
    }

    // insert: stream candidate (candidate_id, dist from p) into reservoir.
    // sketch_p = pre-computed sketch of point p
    // sketch_c = pre-computed sketch of candidate c
    // Returns true if candidate was accepted.
    bool insert(uint32_t candidate_id, const float* sketch_p, const float* sketch_c, float dist) {
        const auto decision = insert_with_decision(candidate_id, sketch_p, sketch_c, dist);
        return decision != InsertDecision::kCollisionReject && decision != InsertDecision::kReservoirReject;
    }

    InsertDecision insert_with_decision(uint32_t candidate_id, const float* sketch_p, const float* sketch_c, float dist) {
        uint16_t h = residual_hash(sketch_p, sketch_c);
        uint16_t dist_bf = float_to_bfloat16(dist);

        // Case 1: find collision slot (same hash)
        for (uint32_t i = 0; i < size_; i++) {
            if (reservoir_[i].hash == h) {
                if (dist < bfloat16_to_float(reservoir_[i].dist_bf)) {
                    reservoir_[i].id = candidate_id;
                    reservoir_[i].dist_bf = dist_bf;
                    // Recompute farthest if we replaced it
                    if (i == farthest_idx_) recompute_farthest();
                    return InsertDecision::kCollisionReplace;
                }
                return InsertDecision::kCollisionReject;
            }
        }

        // Case 2: no collision, reservoir not full
        if (size_ < max_degree_) {
            reservoir_[size_] = {candidate_id, h, dist_bf};
            if (dist > farthest_dist_) {
                farthest_dist_ = dist;
                farthest_idx_ = size_;
            }
            size_++;
            return InsertDecision::kAppend;
        }

        // Case 3: no collision, reservoir full — replace farthest if closer
        if (dist < farthest_dist_) {
            reservoir_[farthest_idx_] = {candidate_id, h, dist_bf};
            recompute_farthest();
            return InsertDecision::kReservoirReplace;
        }
        return InsertDecision::kReservoirReject;
    }

    // Return final neighbor IDs
    std::vector<uint32_t> neighbors() const {
        std::vector<uint32_t> result(size_);
        for (uint32_t i = 0; i < size_; i++) result[i] = reservoir_[i].id;
        return result;
    }

    uint32_t size() const { return size_; }

 private:
    struct Slot {
        uint32_t id = 0;
        uint16_t hash = 0;
        uint16_t dist_bf = 0;  // bfloat16 distance to p
    };

    uint32_t dim_;
    uint32_t m_;           // number of hash bits
    uint32_t max_degree_;  // reservoir capacity

    std::vector<Slot> reservoir_;
    uint32_t size_;
    uint32_t farthest_idx_;
    float farthest_dist_;

    std::vector<float> hyperplanes_;  // m × dim

    void recompute_farthest() {
        farthest_dist_ = 0.0f;
        farthest_idx_ = 0;
        for (uint32_t i = 0; i < size_; i++) {
            float d = bfloat16_to_float(reservoir_[i].dist_bf);
            if (d > farthest_dist_) {
                farthest_dist_ = d;
                farthest_idx_ = i;
            }
        }
    }

    static uint16_t float_to_bfloat16(float f) {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        return static_cast<uint16_t>(bits >> 16);
    }

    static float bfloat16_to_float(uint16_t bf) {
        uint32_t bits = static_cast<uint32_t>(bf) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

// HashReservoir: per-point global candidate reservoir.
// Same cone-pruning logic as HashPrune but no hyperplane storage.
// Caller provides pre-computed residual hash via HashPrune::residual_hash().
// Memory: max_degree * 8 bytes + 16 bytes overhead per point.
class HashReservoir {
 public:
    HashReservoir() = default;
    HashReservoir(uint32_t hash_bits, uint32_t max_degree)
        : m_(hash_bits), max_degree_(max_degree), size_(0), farthest_idx_(0), farthest_dist_(0.0f) {
        reservoir_.resize(max_degree_);
    }

    // Insert candidate with pre-computed residual hash and L2 distance.
    // Returns true if candidate was accepted into the reservoir.
    bool insert(uint32_t candidate_id, uint16_t hash, float dist) {
        const uint16_t dist_bf = float_to_bfloat16(dist);

        for (uint32_t i = 0; i < size_; i++) {
            if (reservoir_[i].hash == hash) {
                if (dist < bfloat16_to_float(reservoir_[i].dist_bf)) {
                    reservoir_[i] = {candidate_id, hash, dist_bf};
                    if (i == farthest_idx_) recompute_farthest();
                    return true;
                }
                return false;
            }
        }
        if (size_ < max_degree_) {
            reservoir_[size_] = {candidate_id, hash, dist_bf};
            if (dist > farthest_dist_) { farthest_dist_ = dist; farthest_idx_ = size_; }
            size_++;
            return true;
        }
        if (dist < farthest_dist_) {
            reservoir_[farthest_idx_] = {candidate_id, hash, dist_bf};
            recompute_farthest();
            return true;
        }
        return false;
    }

    std::vector<uint32_t> neighbors() const {
        std::vector<uint32_t> result(size_);
        for (uint32_t i = 0; i < size_; i++) result[i] = reservoir_[i].id;
        return result;
    }

    uint32_t size() const { return size_; }

 private:
    struct Slot { uint32_t id = 0; uint16_t hash = 0; uint16_t dist_bf = 0; };

    uint32_t m_ = 0;
    uint32_t max_degree_ = 0;
    uint32_t size_ = 0;
    uint32_t farthest_idx_ = 0;
    float farthest_dist_ = 0.0f;
    std::vector<Slot> reservoir_;

    void recompute_farthest() {
        farthest_dist_ = 0.0f;
        farthest_idx_ = 0;
        for (uint32_t i = 0; i < size_; i++) {
            const float d = bfloat16_to_float(reservoir_[i].dist_bf);
            if (d > farthest_dist_) { farthest_dist_ = d; farthest_idx_ = i; }
        }
    }

    static uint16_t float_to_bfloat16(float f) {
        uint32_t bits; std::memcpy(&bits, &f, sizeof(bits)); return static_cast<uint16_t>(bits >> 16);
    }
    static float bfloat16_to_float(uint16_t bf) {
        uint32_t bits = static_cast<uint32_t>(bf) << 16; float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }
};

}  // namespace knowhere::pipnn_diskann
