#pragma once

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

namespace dlengine {

class Sequence;

class GDNStateManager {
public:
    GDNStateManager(const std::string& engine_id, int group_id, int num_slots);

    // Rebuild the free list for a new total slot count, dropping all current
    // allocations. Must only be called at init (before any allocate()); used to
    // enlarge the pool for session-scoped state caching.
    void reset(int num_slots);

    // State slot allocation and deallocation
    bool can_allocate() const;
    void allocate(Sequence& seq);
    void deallocate(Sequence& seq, BlockContextSlot slot);

    // Return a slot id to the free list without a backing Sequence. Used to
    // free the GDN state slot retained by a parked session (see
    // SessionStateCache) on eviction. No-op for slot_id < 0.
    void free_slot(int slot_id)
    {
        if (slot_id >= 0) {
            deallocate_slot(slot_id);
        }
    }

    // Accessors
    std::vector<int> free_slots() const
    {
        return std::vector<int>(free_slots_.begin(), free_slots_.end());
    }
    int num_free_slots() const
    {
        return static_cast<int>(free_slots_.size());
    }

private:
    int  allocate_slot();
    void deallocate_slot(int slot_id);

    std::string engine_id_;
    int         group_id_;
    int         num_slots_;

    std::list<int>                        free_slots_;
    std::vector<std::list<int>::iterator> slot_id_to_free_list_it_;
    std::unordered_set<int>               used_slots_;
};

}  // namespace dlengine
