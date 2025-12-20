#pragma once
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include "sp_state_manager.h"
#include "sequence.h"

namespace nanodeploy {

// Return type: vector<pair<sequence pointer, target DP index>>
// Corresponds to Python return type: List[Tuple[Sequence, int]]
using MigrationList = std::vector<std::pair<std::shared_ptr<Sequence>, int>>;

MigrationList postprocess_sequences(
    std::vector<std::shared_ptr<SPStateManager>> worker_states,
    // dp_seqs: [dp_rank][sp_rank][batch_idx] -> Sequence*
    const std::vector<std::vector<std::vector<std::shared_ptr<Sequence>>>>& dp_seqs,
    // dp_token_ids: [dp_rank][sp_rank][batch_idx] -> List[int]
    const std::vector<std::vector<std::vector<std::vector<int>>>>& dp_token_ids,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    bool update_metrics
);

} // namespace nanodeploy