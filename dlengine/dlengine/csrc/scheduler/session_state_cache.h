#pragma once

#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dlengine {

// One parked session's retained resources. After a request from a session
// (identified by ``affinity_key``) finishes, instead of freeing its KV blocks
// and GatedDeltaNet recurrent-state slot we PARK them here keyed by the
// session. The next turn from the same session reuses them as a chunked-prefill
// continuation (num_cached_tokens == length), so neither the shared full-
// attention KV nor the linear-attention recurrent state has to be recomputed.
//
// Ownership: a parked entry *exclusively* owns its KV blocks (ref_count held at
// 1) and its GDN state slot. Adoption transfers ownership to the new sequence
// and removes the entry from the cache; eviction frees the resources. This
// keeps reference counting simple and double-free-free (see GroupManager).
struct ParkedSession {
    uint64_t         affinity_key    = 0;
    int              state_slot      = -1;  // GDN recurrent/conv state slot
    int              master_group_id = 0;
    int              length          = 0;  // parked context length in tokens
    std::vector<int> token_ids;            // prefix tokens (size == length)
    // Per-group retained block tables (physical block ids). Index = group_id.
    // Session caching is only enabled for group_size == 1, so in practice this
    // holds a single group, but the layout is kept general.
    std::vector<std::vector<int>> group_block_tables;
};

// Fixed-capacity LRU store of parked sessions. Pure bookkeeping: it never
// touches GPU memory or the block/state managers. The owning GroupManager is
// responsible for actually freeing the resources of any entry this returns
// (replaced, evicted, or explicitly taken-but-rejected).
class SessionStateCache {
public:
    explicit SessionStateCache(int capacity = 0): capacity_(capacity < 0 ? 0 : capacity) {}

    bool enabled() const
    {
        return capacity_ > 0;
    }
    int capacity() const
    {
        return capacity_;
    }
    void set_capacity(int capacity)
    {
        capacity_ = capacity < 0 ? 0 : capacity;
    }
    int size() const
    {
        return static_cast<int>(lru_.size());
    }
    bool empty() const
    {
        return lru_.empty();
    }

    // Read-only lookup without mutating recency. Returns nullptr on miss.
    const ParkedSession* peek(uint64_t key) const
    {
        auto it = index_.find(key);
        return it == index_.end() ? nullptr : &(*it->second);
    }

    // Remove and return the entry for ``key`` (ownership transferred to the
    // caller). Returns nullopt on miss.
    std::optional<ParkedSession> take(uint64_t key)
    {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return std::nullopt;
        }
        ParkedSession out = std::move(*it->second);
        lru_.erase(it->second);
        index_.erase(it);
        return out;
    }

    // Insert (or replace) an entry, making it most-recently-used, then evict
    // least-recently-used entries down to capacity. Any displaced entry
    // (replaced same-key entry or evicted LRU) is returned so the caller can
    // free its retained resources.
    std::vector<ParkedSession> put(ParkedSession entry)
    {
        std::vector<ParkedSession> displaced;
        if (capacity_ <= 0) {
            // Caching disabled: caller must free the entry it tried to park.
            displaced.push_back(std::move(entry));
            return displaced;
        }

        uint64_t key = entry.affinity_key;
        auto     it  = index_.find(key);
        if (it != index_.end()) {
            displaced.push_back(std::move(*it->second));
            lru_.erase(it->second);
            index_.erase(it);
        }

        lru_.push_back(std::move(entry));
        index_[key] = std::prev(lru_.end());

        while (static_cast<int>(lru_.size()) > capacity_) {
            displaced.push_back(std::move(lru_.front()));
            index_.erase(lru_.front().affinity_key);
            lru_.pop_front();
        }
        return displaced;
    }

    // Evict and return the least-recently-used entry (for KV-pressure relief).
    // Returns nullopt when empty.
    std::optional<ParkedSession> evict_lru()
    {
        if (lru_.empty()) {
            return std::nullopt;
        }
        ParkedSession out = std::move(lru_.front());
        index_.erase(out.affinity_key);
        lru_.pop_front();
        return out;
    }

private:
    int                                                              capacity_;
    std::list<ParkedSession>                                         lru_;  // front = LRU, back = MRU
    std::unordered_map<uint64_t, std::list<ParkedSession>::iterator> index_;
};

}  // namespace dlengine
