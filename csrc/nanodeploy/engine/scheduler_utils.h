#pragma once
#include <vector>
#include <string>
#include <utility>
#include "sp_state_manager.h"
#include "sequence.h"

namespace nanodeploy {

// 定义返回值类型: vector<pair<序列指针, 目标DP索引>>
// 对应 Python 返回值: List[Tuple[Sequence, int]]
using MigrationList = std::vector<std::pair<Sequence*, int>>;

MigrationList postprocess_sequences(
    std::vector<SPStateManager*> worker_states,
    // dp_seqs: [dp_rank][sp_rank][batch_idx] -> Sequence*
    const std::vector<std::vector<std::vector<Sequence*>>>& dp_seqs,
    // dp_token_ids: [dp_rank][sp_rank][batch_idx] -> List[int]
    const std::vector<std::vector<std::vector<std::vector<int>>>>& dp_token_ids,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    bool update_metrics
);

} // namespace nanodeploy