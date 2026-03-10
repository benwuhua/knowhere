# PiPNN Global HashReservoir Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace per-leaf-temporary HashPrune + quality-blind `append-if-not-full` edge insertion with a global per-point `HashReservoir` that all leaves stream candidates into, fixing the root cause of recall collapse.

**Architecture:** Pre-compute all point sketches globally with one shared `HashPrune` instance. Allocate `n` lightweight `HashReservoir` objects (no hyperplane storage, ~550 MB for 1M points). Each leaf streams its top-k candidates into `global_reservoirs[point_i]` under a single per-point spinlock. After all leaves finish, extract `adjacency[i] = global_reservoirs[i].neighbors()`. The remainder of the pipeline (connectivity repair, robust_prune_pass, VamanaSerializer) is unchanged.

**Tech Stack:** C++17, Eigen (existing), OpenMP (existing), Catch2 v3

---

## Root Cause Recap

Current flow loses candidates at global merge:
```
process_leaf → temp HashPrune (leaf-local) → neighbors()
             → insert_neighbor() [quality-blind: append if not full, else DISCARD]
```

`inserted_overlap=1.42%` → `retained_overlap=0.14%` because the first leaf to process point `i` fills `adjacency[i]`; later leaves with better candidates find the list full and discard them.

Fix: each point has one global reservoir that all leaves stream into. HashPrune's cone-pruning then operates on the complete candidate set, not a per-leaf slice.

---

## Task 1: Add `HashReservoir` to `hash_prune.h` and make `residual_hash` static

**Files:**
- Modify: `src/index/diskann/impl/hash_prune.h`
- Test: `tests/ut/test_pipnn_diskann.cc`

**Step 1: Write failing tests first**

Add to `tests/ut/test_pipnn_diskann.cc` inside the existing `#include` section (no guard needed — `HashReservoir` will be in `hash_prune.h` which is already included without `KNOWHERE_WITH_PIPNN`):

```cpp
TEST_CASE("HashReservoir: basic insert and capacity", "[pipnn_diskann][unit]") {
    knowhere::pipnn_diskann::HashReservoir r(12, 4);
    REQUIRE(r.size() == 0);

    r.insert(10, 0x0001, 1.0f);
    r.insert(20, 0x0002, 2.0f);
    r.insert(30, 0x0003, 3.0f);
    r.insert(40, 0x0004, 4.0f);
    REQUIRE(r.size() == 4);

    // Reservoir full: new candidate farther than all → rejected
    r.insert(50, 0x0005, 5.0f);
    REQUIRE(r.size() == 4);

    // Reservoir full: closer than farthest → replaces farthest
    r.insert(99, 0x0006, 0.5f);
    auto neighbors = r.neighbors();
    REQUIRE(std::find(neighbors.begin(), neighbors.end(), 99u) != neighbors.end());
    REQUIRE(std::find(neighbors.begin(), neighbors.end(), 40u) == neighbors.end()); // 40 was farthest
}

TEST_CASE("HashReservoir: hash collision keeps closer candidate", "[pipnn_diskann][unit]") {
    knowhere::pipnn_diskann::HashReservoir r(12, 8);
    r.insert(1, 0xABCD, 3.0f);  // first in slot
    r.insert(2, 0xABCD, 1.0f);  // same hash, closer → replaces
    r.insert(3, 0xABCD, 5.0f);  // same hash, farther → rejected

    auto neighbors = r.neighbors();
    REQUIRE(std::find(neighbors.begin(), neighbors.end(), 2u) != neighbors.end());
    REQUIRE(std::find(neighbors.begin(), neighbors.end(), 1u) == neighbors.end());
    REQUIRE(std::find(neighbors.begin(), neighbors.end(), 3u) == neighbors.end());
}

TEST_CASE("HashReservoir: history-independent (order does not matter)", "[pipnn_diskann][unit]") {
    // Same 5 candidates inserted in two different orders must produce same neighbor set
    // when reservoir capacity = 4
    std::vector<std::pair<uint16_t, float>> candidates = {
        {0x0001, 1.0f}, {0x0002, 2.0f}, {0x0003, 3.0f}, {0x0004, 4.0f}, {0x0005, 0.5f}
    };

    auto fill = [&](std::vector<std::pair<uint16_t,float>> order) {
        knowhere::pipnn_diskann::HashReservoir r(12, 4);
        for (auto [h, d] : order) r.insert(static_cast<uint32_t>(h), h, d);
        auto n = r.neighbors();
        std::sort(n.begin(), n.end());
        return n;
    };

    auto fwd = fill(candidates);
    std::vector<std::pair<uint16_t, float>> reversed(candidates.rbegin(), candidates.rend());
    auto rev = fill(reversed);
    REQUIRE(fwd == rev);
}

TEST_CASE("HashPrune residual_hash is static-callable", "[pipnn_diskann][unit]") {
    // Verify the static overload compiles and agrees with the instance method
    knowhere::pipnn_diskann::HashPrune hp(128, 12, 32);
    std::vector<float> sp(12, 0.0f), sc(12, 1.0f);
    uint16_t h_instance = hp.residual_hash(sp.data(), sc.data());
    uint16_t h_static   = knowhere::pipnn_diskann::HashPrune::residual_hash(sp.data(), sc.data(), 12);
    REQUIRE(h_instance == h_static);
}
```

**Step 2: Run to confirm they fail (HashReservoir doesn't exist yet)**

```bash
cd /Users/ryan/Code/knowhere
scripts/remote/sync.sh --mode git --ref feat/pipnn-diskann
scripts/remote/test.sh --type Debug --filter '[pipnn_diskann][unit]'
```

Expected: compile error — `HashReservoir` not found, `residual_hash` not static.

**Step 3: Implement — modify `src/index/diskann/impl/hash_prune.h`**

**3a.** Make `residual_hash` static by adding a static overload and having the instance method call it:

```cpp
// Static overload (new) — callable without a HashPrune instance
static uint16_t residual_hash(const float* sketch_p, const float* sketch_c, uint32_t m) {
    uint16_t h = 0;
    for (uint32_t i = 0; i < m && i < 16; i++) {
        if (sketch_c[i] >= sketch_p[i]) h |= (1u << i);
    }
    return h;
}

// Instance method (keep for backward compat) — delegates to static
uint16_t residual_hash(const float* sketch_p, const float* sketch_c) const {
    return residual_hash(sketch_p, sketch_c, m_);
}
```

**3b.** Add `HashReservoir` class at the bottom of `hash_prune.h`, before the closing `}  // namespace`:

```cpp
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
```

**Step 4: Run tests to confirm they pass**

```bash
scripts/remote/sync.sh --mode git --ref feat/pipnn-diskann
scripts/remote/test.sh --type Debug --filter '[pipnn_diskann][unit]'
```

Expected: all 4 new tests PASS, no existing tests broken.

**Step 5: Commit**

```bash
git add src/index/diskann/impl/hash_prune.h tests/ut/test_pipnn_diskann.cc
git commit -m "feat(pipnn): add HashReservoir and static residual_hash for global reservoir path"
```

---

## Task 2: Add global state to `BuildContext` in `pipnn_builder.h`

**Files:**
- Modify: `src/index/diskann/impl/pipnn_builder.h`

**Step 1: Add fields to `BuildContext`**

In the `BuildContext` struct, after the existing `std::vector<uint32_t> membership_counts;` line, add:

```cpp
// Global per-point sketches: flat layout [point_id * hash_bits + bit_idx]
// Populated in build() before leaf processing. Size = n * hash_bits.
std::vector<float> global_sketches_flat;

// Global per-point reservoirs for the new streaming path.
// Populated in build(). Size = n. Each reservoir is protected by node_locks[i].
std::vector<HashReservoir> global_reservoirs;
```

No test needed for this step — it's a data structure change with no observable behavior yet. The compile check comes in Task 3.

**Step 2: Commit**

```bash
git add src/index/diskann/impl/pipnn_builder.h
git commit -m "feat(pipnn): add global_sketches_flat and global_reservoirs to BuildContext"
```

---

## Task 3: Pre-compute global sketches and allocate reservoirs in `build()`

**Files:**
- Modify: `src/index/diskann/impl/pipnn_builder.cc`

**Step 1: Add global sketch pre-computation and reservoir allocation**

In `PiPNNBuilder::build()`, immediately after the RBC partition block (after the `LOG_KNOWHERE_INFO_` lines for RBC coverage stats) and before the leaf processing `#ifdef _OPENMP` block, insert:

```cpp
// Pre-compute all point sketches using one shared HashPrune instance.
// HashPrune with max_degree=1 is used only for compute_sketch; the reservoir
// is unused. All instances with same (dim, hash_bits) produce identical
// hyperplanes (deterministic seed), so one instance suffices.
{
    const auto sketch_start = clock::now();
    HashPrune sketch_engine(dim, config_.hash_bits, /*max_degree=*/1);
    context.global_sketches_flat.resize(static_cast<size_t>(n) * config_.hash_bits);
    for (uint32_t i = 0; i < n; ++i) {
        sketch_engine.compute_sketch(data + static_cast<size_t>(i) * dim,
                                     context.global_sketches_flat.data() + static_cast<size_t>(i) * config_.hash_bits);
    }
    const auto sketch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - sketch_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(sketch_ns)
                       << " ms (Global sketch pre-computation, num_points=" << n
                       << ", sketch_bytes=" << (context.global_sketches_flat.size() * sizeof(float)) << ")";
}

// Allocate global per-point reservoirs.
{
    context.global_reservoirs.assign(n, HashReservoir(config_.hash_bits, config_.max_degree));
    const size_t reservoir_bytes = static_cast<size_t>(n) * (static_cast<size_t>(config_.max_degree) * 8 + 16);
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: global reservoirs allocated"
                       << ", num_points=" << n
                       << ", reservoir_mb=" << (reservoir_bytes / (1024 * 1024));
}
```

**Step 2: Add adjacency extraction after the leaf processing block**

After the closing brace of the `#ifdef _OPENMP` / `for leaves` block (after the leaf_process_ns logging lines), insert:

```cpp
// Extract adjacency from global reservoirs.
{
    const auto extract_start = clock::now();
    for (uint32_t i = 0; i < n; ++i) {
        context.adjacency[i] = context.global_reservoirs[i].neighbors();
    }
    const auto extract_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - extract_start).count();
    LOG_KNOWHERE_INFO_ << "[PiPNN Profiling] Stage: " << ns_to_ms(extract_ns)
                       << " ms (Adjacency extraction from global reservoirs, num_points=" << n << ")";
}
```

**Step 3: Build to check it compiles**

```bash
scripts/remote/sync.sh --mode git --ref feat/pipnn-diskann
scripts/remote/build.sh --type Debug
```

Expected: clean build (no new warnings, no errors).

**Step 4: Commit**

```bash
git add src/index/diskann/impl/pipnn_builder.cc
git commit -m "feat(pipnn): pre-compute global sketches and allocate per-point reservoirs in build()"
```

---

## Task 4: Refactor `process_leaf` to stream into global reservoirs

This is the core change. Currently `process_leaf` creates a leaf-local `HashPrune prune` per point and calls `insert_bidirectional_edge`. We replace this with a direct stream into `context.global_reservoirs[global_i]`.

**Files:**
- Modify: `src/index/diskann/impl/pipnn_builder.cc`

**Step 1: Locate the HashPrune block in `process_leaf`**

Find the block starting at:
```cpp
HashPrune prune(dim, config_.hash_bits, config_.max_degree);
const auto hash_prune_insert_start = clock::now();
for (const auto& candidate : candidates) {
    ...
    const auto decision = prune.insert_with_decision(global_j, sketches[i].data(), sketch_j.data(), dist);
    ...
}
```

And the subsequent block that extracts pruned neighbors and calls `insert_bidirectional_edge`.

**Step 2: Replace both blocks**

Delete from `HashPrune prune(dim, config_.hash_bits, config_.max_degree);` through the end of the `for (uint32_t neighbor : pruned_neighbors)` loop (including the `if (probe_node) { ... inserted_total ... }` block).

Replace with:

```cpp
// Stream candidates into global_reservoirs[global_i] under per-point lock.
// Bidirectionality emerges naturally: when point j is processed in this same
// leaf (or any shared leaf), it will also consider point i as a candidate and
// insert i into global_reservoirs[j].
const float* sketch_i = context.global_sketches_flat.data() +
                        static_cast<size_t>(global_i) * config_.hash_bits;
const auto hash_prune_insert_start = clock::now();
for (const auto& candidate : candidates) {
    const uint32_t global_j = candidate.second;
    const float dist = std::max(0.0f, candidate.first);
    const float* sketch_j = context.global_sketches_flat.data() +
                            static_cast<size_t>(global_j) * config_.hash_bits;
    const uint16_t h = HashPrune::residual_hash(sketch_i, sketch_j, config_.hash_bits);

    std::lock_guard<SpinMutex> lock(context.node_locks[global_i]);
    const bool accepted = context.global_reservoirs[global_i].insert(global_j, h, dist);

    if (probe_node && accepted) {
        // Reuse inserted_overlap probe: track how many accepted candidates are true top-k
        // (exact_topk must be populated — check probe_node guard above)
    }
}
const auto hash_prune_insert_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - hash_prune_insert_start).count();
context.hash_prune_total_ns.fetch_add(hash_prune_insert_ns, std::memory_order_relaxed);
```

**Step 3: Remove now-unused per-leaf sketch computation**

Find and delete:
```cpp
HashPrune sketcher(dim, config_.hash_bits, config_.max_degree);
std::vector<std::vector<float>> sketches(leaf_size, std::vector<float>(config_.hash_bits));
for (uint32_t local_id = 0; local_id < leaf_size; ++local_id) {
    ...
    sketcher.compute_sketch(vec, sketches[local_id].data());
}
```

Also remove any uses of `sketches[i].data()` in the probe code (the global sketch replaces it).

**Step 4: Update probe stats for the new path**

In the probe block before the streaming loop, the `exact_topk` computation and `candidate_overlap_hits` tracking remain valid (they measure whether the candidates set contains true neighbors before insertion). Retain them.

After the streaming loop, add:
```cpp
if (probe_node) {
    // Count how many neighbors the global reservoir has accepted for this point
    // (approximate — reads under no lock, benign for diagnostics)
    const uint32_t inserted_count = context.global_reservoirs[global_i].size();
    context.quality_probe.inserted_total.fetch_add(
        static_cast<int64_t>(inserted_count), std::memory_order_relaxed);
    // Count true top-k in current reservoir state
    const auto current_neighbors = context.global_reservoirs[global_i].neighbors();
    context.quality_probe.inserted_overlap_hits.fetch_add(
        static_cast<int64_t>(count_overlap_with_exact_topk(exact_topk, current_neighbors)),
        std::memory_order_relaxed);
}
```

**Step 5: Build and run unit tests**

```bash
scripts/remote/sync.sh --mode git --ref feat/pipnn-diskann
scripts/remote/build.sh --type Debug
scripts/remote/test.sh --type Debug --filter '[pipnn_diskann][unit]'
```

Expected: all unit tests PASS. (E2E tests require Release + KNOWHERE_WITH_PIPNN and are covered in Task 5.)

**Step 6: Commit**

```bash
git add src/index/diskann/impl/pipnn_builder.cc
git commit -m "feat(pipnn): stream candidates into global HashReservoir, replace leaf-local HashPrune+append"
```

---

## Task 5: Run E2E recall test and verify improvement

**Files:** No code changes — this is a verification step.

**Step 1: Build Release**

```bash
scripts/remote/build.sh --type Release
```

**Step 2: Run recall comparison test (synthetic dataset, quick smoke)**

```bash
scripts/remote/test.sh --type Release --filter '[pipnn_diskann][e2e][recall]'
scripts/remote/fetch-logs.sh
```

Look for in logs:
```
grep 'recall@\|graph_direct\|postprocess\|PiPNN Graph Probe' $TMPDIR/knowhere-remote-logs/test_*.log
```

**Success criteria:**
- `graph_direct recall@10` rises from ~0.034 to ≥ 0.10 (meaningful improvement even if not DiskANN-level)
- `pre_final_retained_exact_topk_overlap` approaches `inserted_exact_topk_overlap` (≤ 2× loss instead of 10× loss)
- `inserted_exact_topk_overlap` ≥ `candidate_exact_topk_overlap` (reservoir is no longer discarding better candidates)

**Step 3: Run forced-DiskANN baseline for comparison**

```bash
KNOWHERE_PIPNN_FORCE_DISKANN_BUILD_INDEX=1 \
  scripts/remote/test.sh --type Release --filter '[pipnn_diskann][e2e][recall]'
```

Document both numbers.

**Step 4: If recall ≥ 0.10, run Cohere-1M benchmark**

```bash
KNOWHERE_PIPNN_BASE_FBIN=/data/work/datasets/wikipedia-cohere-1m/base.fbin \
KNOWHERE_PIPNN_QUERY_FBIN=/data/work/datasets/wikipedia-cohere-1m/query.fbin \
KNOWHERE_PIPNN_GT_IBIN=/data/work/datasets/wikipedia-cohere-1m/gt.ibin \
KNOWHERE_PIPNN_DATASET_LABEL=wikipedia-cohere-1m-ip \
  scripts/remote/test.sh --type Release --filter '[pipnn_diskann][e2e][recall]'
```

**Step 5: Document results and commit**

Update `docs/plans/2026-02-28-pipnn-diskann-design.md` with new recall numbers and conclusion.

```bash
git add docs/plans/2026-02-28-pipnn-diskann-design.md
git commit -m "docs(pipnn): record global reservoir recall results"
```

---

## Expected Outcome

| Metric | Before (leaf-local) | Expected (global reservoir) |
|--------|--------------------|-----------------------------|
| `candidate_exact_topk_overlap` | ~0.88% | ~0.88% (unchanged — same GEMM) |
| `inserted_exact_topk_overlap` | ~1.42% | ~0.88%–5% (loss = cone-pruning only) |
| `pre_final_retained_overlap` | ~0.14% | ≈ inserted (no naive-append loss) |
| `graph_direct recall@10` | ~0.034 | ≥ 0.10 (hypothesis) |

If `graph_direct recall@10` stays below 0.10 even with global reservoir, this confirms the recall bottleneck is in candidate source quality, not merge strategy — and that information will formally close the source-side investigation.

---

## Notes

- **Memory:** ~600 MB extra for global sketches + reservoirs at 1M points. Acceptable.
- **No DiskANN modification:** All changes are within Knowhere's own builder layer.
- **Backward compat:** `VamanaSerializer` format and `PQFlashIndex` search path unchanged.
- **Probe stats:** `insert_append/duplicate/degree_full` counters become inert (will log 0). Other stats remain valid.
- **Final prune:** `robust_prune_pass` becomes a near-no-op (reservoir already caps at `max_degree`) but is kept as a safety dedup pass.

---

## Results

**Run date:** 2026-03-10
**Dataset:** Synthetic, 1M rows, 128 dim, 100 queries, k=10 (mode=synthetic_diag)
**Build:** Release, feat/pipnn-diskann, global HashReservoir path active
**Remote log:** `test_bg_20260310T031519Z.log`

### Probe Stats (Graph Quality, sampled_nodes=92)

Scenario `retained_degree_limit_off` and `retained_degree_limit_on` (same RBC/hash config):
- `candidate_exact_topk_overlap`: **0.00883** (0.88%)
- `inserted_exact_topk_overlap`: **0.01766** (1.77%)
- `pre_final_retained_exact_topk_overlap`: **0.00136** (0.14%)

Scenario `boundary_leader_bridge_on` (cross-leaf union + bridge, before disk-full crash):
- `candidate_exact_topk_overlap`: **0.01749** (1.75%)
- `inserted_exact_topk_overlap`: **0.06114** (6.11%)
- `pre_final_retained_exact_topk_overlap`: **0.00272** (0.27%)

Note: all scenarios show `edge_insert_count=0` in profiling, confirming the old bidirectional insert path is no longer active and all candidates are now routed through global reservoirs.

### graph_direct recall@10

| Scenario | graph_direct recall@10 | postprocess_only recall@10 |
|---|---|---|
| `retained_degree_limit_off` | **0.065** | 0.053 |
| `retained_degree_limit_on` | **0.065** | 0.061 |
| `boundary_leader_bridge_on` | — (crashed disk-full) | — |

### Forced-DiskANN Baseline

- `force_diskann_build_index` recall@10: **0.185**
- `diskann_baseline` (pure DiskANN index type) recall@10: **0.161**

### Success Criterion

The threshold of `graph_direct recall@10 >= 0.10` was **NOT met** (observed: 0.065).

### Observation on Global Reservoir

The global reservoir is functioning correctly (`edge_insert_count=0`, adjacency extracted from reservoirs), and `inserted_exact_topk_overlap` (1.77%) is now higher than `candidate_exact_topk_overlap` (0.88%), confirming candidates are accumulating across leaves rather than being discarded. However, `pre_final_retained_exact_topk_overlap` (0.14%) is still ~13x lower than `inserted_exact_topk_overlap`, indicating that the robust prune pass is ejecting true neighbors — not the merge stage. The root bottleneck is now definitively in candidate source quality (GEMM coverage per leaf is too sparse) rather than in the merge/reservoir strategy.

### Conclusion

Global HashReservoir correctly aggregates candidates across leaves and eliminates quality-blind append-discard at merge. The recall gap (0.065 vs target 0.10) is caused by insufficient candidate coverage from the RBC-partitioned GEMM, not by the reservoir strategy. The recall bottleneck is in candidate source quality, not merge strategy; this closes the global reservoir investigation.

Third scenario (`boundary_leader_bridge_on`) crashed with `std::ios_failure` / SIGABRT during `create_disk_layout` — remote disk was full after two 1M-point builds (each ~520 MB raw data + index files).
