#pragma once

#include <list>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "nanodeploy/csrc/sequence/sequence.h"

namespace nanodeploy {

class Sequence;

class GDNStateManager {
public:
    GDNStateManager(const std::string& engine_id, int group_id, int num_slots);

    // State slot allocation and deallocation
    bool can_allocate() const;
    void allocate(Sequence& seq);
    void deallocate(Sequence& seq, BlockContextSlot slot);

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

}  // namespace nanodeploy
