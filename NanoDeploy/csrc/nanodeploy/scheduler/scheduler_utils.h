#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"
#include "thread_pool.h"

namespace nanoinfra {

// Return type: vector<pair<sequence pointer, target DP index>>
// Corresponds to Python return type: List[Tuple[Sequence, int]]
using MigrationList = std::vector<std::pair<std::shared_ptr<Sequence>, int>>;

MigrationList postprocess_sequences(std::vector<std::shared_ptr<SPStateManager>> worker_states,
                                    // dp_sp_seqs: [dp_rank * sp_rank + sp_rank][batch_idx] -> Sequence*
                                    const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_sp_seqs,
                                    // dp_sp_token_ids: [dp_rank * sp_rank + sp_rank][batch_idx] -> List[int]
                                    const std::vector<std::vector<std::vector<int>>>& dp_sp_token_ids,
                                    int                                               eos_id,
                                    bool                                              is_prefill,
                                    bool                                              update_metrics,
                                    ThreadPool*                                       thread_pool = nullptr);

}  // namespace nanoinfra
