#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "nanodeploy/csrc/sequence/sequence.h"

#include "gdn_state_manager.h"

namespace nanodeploy {

GDNStateManager::GDNStateManager(const std::string& engine_id, int group_id, int num_slots):
    engine_id_(engine_id), group_id_(group_id), num_slots_(num_slots)
{
    slot_id_to_free_list_it_.resize(num_slots);
    for (int i = 0; i < num_slots; ++i) {
        free_slots_.push_back(i);
        slot_id_to_free_list_it_[i] = std::prev(free_slots_.end());
    }
}

int GDNStateManager::allocate_slot()
{
    if (free_slots_.empty()) {
        throw std::runtime_error("No free state slots available");
    }

    int slot_id = free_slots_.front();
    free_slots_.pop_front();
    slot_id_to_free_list_it_[slot_id] = free_slots_.end();

    used_slots_.insert(slot_id);
    return slot_id;
}

void GDNStateManager::deallocate_slot(int slot_id)
{
    if (used_slots_.find(slot_id) == used_slots_.end()) {
        throw std::runtime_error("Cannot deallocate an unused state slot");
    }

    used_slots_.erase(slot_id);
    free_slots_.push_back(slot_id);
    slot_id_to_free_list_it_[slot_id] = std::prev(free_slots_.end());
}

bool GDNStateManager::can_allocate() const
{
    return !free_slots_.empty();
}

void GDNStateManager::allocate(Sequence& seq)
{
    int existing_slot = seq.state_slot(BlockContextSlot::ACTIVE);
    if (existing_slot != -1) {
        throw std::runtime_error("Sequence already has an allocated state slot.");
    }

    int slot_id = allocate_slot();
    seq.set_state_slot(BlockContextSlot::ACTIVE, slot_id);
}

void GDNStateManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    int slot_id = seq.state_slot(slot);
    if (slot_id == -1) {
        // No slot allocated
        return;
    }

    deallocate_slot(slot_id);
    seq.set_state_slot(slot, -1);
}

}  // namespace nanodeploy
