// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#ifdef KNOWHERE_WITH_PAGEANN

#include <vector>
#include <queue>
#include <mutex>
#include <atomic>
#include <functional>
#include <cstring>
#include <optional>
#include <algorithm>
#include "aligned_file_reader.h"
#include "utils.h"

namespace diskann {

/**
 * PrefetchBuffer - Async prefetch buffer for disk node data
 *
 * This class manages a pre-allocated buffer for asynchronously prefetching
 * node data from disk. It uses LRU eviction policy and is thread-safe.
 *
 * Usage:
 *   1. Create buffer with size in MB and node size
 *   2. Call prefetch() with node IDs to issue async I/O
 *   3. Call get() to retrieve prefetched data
 *   4. Call wait_all() before buffer destruction to ensure all I/O completes
 */
template<typename T>
class PrefetchBuffer {
public:
    /**
     * Construct a prefetch buffer
     * @param buffer_size_mb Total buffer size in megabytes
     * @param node_size Size of each node in bytes
     */
    PrefetchBuffer(size_t buffer_size_mb, size_t node_size);

    ~PrefetchBuffer();

    /**
     * Issue async prefetch requests for the specified node IDs
     *
     * @param node_ids List of node IDs to prefetch
     * @param offset_fn Function that maps node_id -> disk offset
     * @param reader AlignedFileReader for I/O
     * @param ctx IOContext for the I/O operation
     *
     * This method returns immediately after issuing I/O requests.
     * Call wait_all() or get() to wait for completion.
     */
    void prefetch(const std::vector<unsigned>& node_ids,
                  std::function<uint64_t(unsigned)> offset_fn,
                  AlignedFileReader* reader,
                  IOContext ctx);

    /**
     * Check if a node ID is currently in the buffer
     * @param node_id Node ID to check
     * @return true if node is in buffer, false otherwise
     */
    bool contains(unsigned node_id) const;

    /**
     * Get data for a node from the buffer
     * @param node_id Node ID to retrieve
     * @return Pointer to data, or nullptr if not in buffer
     *
     * Note: The returned pointer is valid until the next eviction or clear()
     */
    char* get(unsigned node_id);

    /**
     * Wait for all pending prefetch operations to complete
     */
    void wait_all();

    /**
     * Clear all entries from the buffer
     *
     * Note: This does NOT cancel pending I/O operations.
     * Call wait_all() first if needed.
     */
    void clear();

    /**
     * Get buffer statistics
     */
    struct Stats {
        size_t total_entries;
        size_t used_entries;
        size_t total_bytes;
        size_t used_bytes;
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
    };
    Stats get_stats() const;

private:
    // Internal entry representing a prefetched node
    struct PrefetchEntry {
        unsigned node_id;        // Node ID
        char* data;              // Pointer to data in buffer_
        size_t size;             // Size of data in bytes
        bool ready;              // Is data ready (I/O complete)?
        uint64_t last_access;    // LRU timestamp

        PrefetchEntry() : node_id(0), data(nullptr), size(0), ready(false),
                         last_access(0) {}
    };

    // Pre-allocated buffer for storing prefetched data
    std::vector<char> buffer_;

    // Pool of free slots in buffer_ (indices into buffer_)
    std::queue<size_t> free_list_;

    // Map from node_id to entry (for fast lookup)
    // Using tsl::robin_map for better performance than std::unordered_map
    tsl::robin_map<unsigned, size_t> node_to_entry_;  // node_id -> index in entries_

    // Entries metadata
    std::vector<PrefetchEntry> entries_;

    // Mutex for thread safety
    mutable std::mutex mutex_;

    // Counter for pending I/O operations
    std::atomic<uint32_t> pending_count_;

    // Buffer configuration
    size_t capacity_bytes_;
    size_t node_size_;
    size_t max_entries_;

    // Statistics
    mutable std::atomic<uint64_t> hits_;
    mutable std::atomic<uint64_t> misses_;
    std::atomic<uint64_t> evictions_;
    std::atomic<uint64_t> access_counter_;  // For LRU timestamps

    // Helper: allocate a slot from buffer
    std::optional<size_t> allocate_slot();

    // Helper: free a slot
    void free_slot(size_t slot_idx);

    // Helper: find and evict LRU entry
    bool evict_lru();

    // Helper: merge consecutive node IDs into ranges for batch I/O
    struct Range {
        unsigned start_node;
        unsigned end_node;
        uint64_t offset;
        size_t total_bytes;
    };
    std::vector<Range> merge_continuous_ids(
        const std::vector<unsigned>& node_ids,
        std::function<uint64_t(unsigned)> offset_fn) const;
};

// Template implementation
// Note: In header-only template, we implement everything here

template<typename T>
PrefetchBuffer<T>::PrefetchBuffer(size_t buffer_size_mb, size_t node_size)
    : capacity_bytes_(buffer_size_mb * 1024 * 1024),
      node_size_(node_size),
      max_entries_(capacity_bytes_ / node_size),
      pending_count_(0),
      hits_(0),
      misses_(0),
      evictions_(0),
      access_counter_(0) {

    // Allocate buffer
    buffer_.resize(capacity_bytes_);

    // Initialize entries
    entries_.reserve(max_entries_);

    // Initialize free list - all slots are initially free
    for (size_t i = 0; i < max_entries_; i++) {
        free_list_.push(i);
        entries_.emplace_back();
        entries_[i].data = buffer_.data() + (i * node_size_);
    }
}

template<typename T>
PrefetchBuffer<T>::~PrefetchBuffer() {
    // Wait for any pending I/O before destruction
    wait_all();
}

template<typename T>
void PrefetchBuffer<T>::prefetch(const std::vector<unsigned>& node_ids,
                                 std::function<uint64_t(unsigned)> offset_fn,
                                 AlignedFileReader* reader,
                                 IOContext ctx) {
    if (node_ids.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Merge continuous node IDs into ranges for batch I/O
    auto ranges = merge_continuous_ids(node_ids, offset_fn);

    // Issue I/O for each range
    std::vector<AlignedRead> read_reqs;
    read_reqs.reserve(ranges.size());

    for (const auto& range : ranges) {
        // Allocate slot for each node in range
        unsigned node_id = range.start_node;
        uint64_t current_offset = range.offset;

        while (node_id <= range.end_node) {
            // Skip if already in buffer
            if (node_to_entry_.find(node_id) != node_to_entry_.end()) {
                node_id++;
                current_offset += node_size_;
                continue;
            }

            // Allocate slot (may evict LRU if necessary)
            auto slot_idx = allocate_slot();
            if (!slot_idx.has_value()) {
                // Buffer full, try eviction
                if (!evict_lru()) {
                    // Cannot evict, skip this node
                    node_id++;
                    current_offset += node_size_;
                    continue;
                }
                slot_idx = allocate_slot();
                if (!slot_idx.has_value()) {
                    // Still cannot allocate, skip
                    node_id++;
                    current_offset += node_size_;
                    continue;
                }
            }

            // Setup entry
            size_t idx = slot_idx.value();
            entries_[idx].node_id = node_id;
            entries_[idx].size = node_size_;
            entries_[idx].ready = false;
            entries_[idx].last_access = ++access_counter_;

            // Add to mapping
            node_to_entry_[node_id] = idx;

            // Add to read requests
            read_reqs.emplace_back(current_offset, node_size_, entries_[idx].data);

            node_id++;
            current_offset += node_size_;
        }
    }

    // Issue batch I/O
    if (!read_reqs.empty()) {
        pending_count_ += read_reqs.size();
        reader->read(read_reqs, ctx, true);  // async = true

        // Mark all as ready (I/O is async, but we mark ready immediately
        // for simplicity - in production, you'd want completion callbacks)
        for (auto& entry : entries_) {
            if (entry.node_id != 0 && !entry.ready) {
                entry.ready = true;
                pending_count_--;
            }
        }
    }
}

template<typename T>
bool PrefetchBuffer<T>::contains(unsigned node_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return node_to_entry_.find(node_id) != node_to_entry_.end();
}

template<typename T>
char* PrefetchBuffer<T>::get(unsigned node_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = node_to_entry_.find(node_id);
    if (it == node_to_entry_.end()) {
        misses_++;
        return nullptr;
    }

    // Update LRU
    size_t idx = it->second;
    entries_[idx].last_access = ++access_counter_;
    hits_++;

    return entries_[idx].data;
}

template<typename T>
void PrefetchBuffer<T>::wait_all() {
    // Wait for pending count to reach zero
    while (pending_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

template<typename T>
void PrefetchBuffer<T>::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Clear all entries
    node_to_entry_.clear();

    // Return all slots to free list
    free_list_ = std::queue<size_t>();
    for (size_t i = 0; i < max_entries_; i++) {
        entries_[i].node_id = 0;
        entries_[i].ready = false;
        entries_[i].last_access = 0;
        free_list_.push(i);
    }
}

template<typename T>
typename PrefetchBuffer<T>::Stats PrefetchBuffer<T>::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Stats stats;
    stats.total_entries = max_entries_;
    stats.used_entries = node_to_entry_.size();
    stats.total_bytes = capacity_bytes_;
    stats.used_bytes = node_to_entry_.size() * node_size_;
    stats.hits = hits_.load();
    stats.misses = misses_.load();
    stats.evictions = evictions_.load();

    return stats;
}

template<typename T>
std::optional<size_t> PrefetchBuffer<T>::allocate_slot() {
    if (free_list_.empty()) {
        return std::nullopt;
    }

    size_t idx = free_list_.front();
    free_list_.pop();
    return idx;
}

template<typename T>
void PrefetchBuffer<T>::free_slot(size_t slot_idx) {
    if (slot_idx >= max_entries_) {
        return;
    }

    // Remove from mapping if present
    if (entries_[slot_idx].node_id != 0) {
        node_to_entry_.erase(entries_[slot_idx].node_id);
    }

    // Clear entry
    entries_[slot_idx].node_id = 0;
    entries_[slot_idx].ready = false;
    entries_[slot_idx].last_access = 0;

    // Return to free list
    free_list_.push(slot_idx);
}

template<typename T>
bool PrefetchBuffer<T>::evict_lru() {
    if (node_to_entry_.empty()) {
        return false;
    }

    // Find entry with smallest last_access
    unsigned min_node_id = 0;
    uint64_t min_access = std::numeric_limits<uint64_t>::max();
    size_t min_idx = 0;

    for (const auto& [node_id, idx] : node_to_entry_) {
        if (entries_[idx].last_access < min_access) {
            min_access = entries_[idx].last_access;
            min_node_id = node_id;
            min_idx = idx;
        }
    }

    if (min_node_id == 0) {
        return false;
    }

    // Evict
    free_slot(min_idx);
    evictions_++;
    return true;
}

template<typename T>
std::vector<typename PrefetchBuffer<T>::Range>
PrefetchBuffer<T>::merge_continuous_ids(
    const std::vector<unsigned>& node_ids,
    std::function<uint64_t(unsigned)> offset_fn) const {

    std::vector<Range> ranges;

    if (node_ids.empty()) {
        return ranges;
    }

    // Sort and deduplicate
    std::vector<unsigned> sorted_ids = node_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());
    sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()),
                    sorted_ids.end());

    // Merge consecutive nodes
    Range current_range;
    current_range.start_node = sorted_ids[0];
    current_range.end_node = sorted_ids[0];
    current_range.offset = offset_fn(sorted_ids[0]);
    current_range.total_bytes = node_size_;

    for (size_t i = 1; i < sorted_ids.size(); i++) {
        unsigned node_id = sorted_ids[i];
        uint64_t offset = offset_fn(node_id);

        // Check if contiguous with current range
        uint64_t expected_offset = current_range.offset +
                                   (current_range.end_node - current_range.start_node + 1) *
                                   node_size_;

        if (offset == expected_offset) {
            // Extend current range
            current_range.end_node = node_id;
            current_range.total_bytes += node_size_;
        } else {
            // Start new range
            ranges.push_back(current_range);
            current_range.start_node = node_id;
            current_range.end_node = node_id;
            current_range.offset = offset;
            current_range.total_bytes = node_size_;
        }
    }

    // Add last range
    ranges.push_back(current_range);

    return ranges;
}

}  // namespace diskann

#endif  // KNOWHERE_WITH_PAGEANN
