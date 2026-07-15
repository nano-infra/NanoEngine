#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"

namespace nanodeploy {

namespace {

std::string trim_copy(const std::string& input)
{
    const auto start = input.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) {
        return "";
    }
    const auto end = input.find_last_not_of(" \t\n\r");
    return input.substr(start, end - start + 1);
}

std::vector<SPBucketInterval> parse_bucket_policy(const std::string& text, int max_sp)
{
    std::vector<SPBucketInterval> intervals;
    const std::string             trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return intervals;
    }

    std::stringstream ss(trimmed);
    std::string       item;
    int               prev_high = std::numeric_limits<int>::min();
    while (std::getline(ss, item, ';')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        const auto colon = item.find(':');
        const auto dash  = item.find('-', colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || dash == std::string::npos) {
            throw std::runtime_error("Invalid dynamic_sp_bucket_policy item: " + item);
        }
        SPBucketInterval interval;
        interval.sp_size      = std::stoi(trim_copy(item.substr(0, colon)));
        interval.seq_len_low  = std::stoi(trim_copy(item.substr(colon + 1, dash - colon - 1)));
        interval.seq_len_high = std::stoi(trim_copy(item.substr(dash + 1)));
        if (interval.sp_size < 1 || interval.sp_size > max_sp) {
            throw std::runtime_error("Bucket sp_size out of range: " + item);
        }
        if (interval.seq_len_low < 0 || interval.seq_len_high < interval.seq_len_low) {
            throw std::runtime_error("Invalid bucket seq range: " + item);
        }
        if (!intervals.empty() && interval.seq_len_low <= prev_high) {
            throw std::runtime_error("dynamic_sp_bucket_policy ranges must be strictly increasing");
        }
        prev_high = interval.seq_len_high;
        intervals.push_back(interval);
    }
    return intervals;
}

const char* dynamic_sp_size_strategy_name(DynamicSPSizeStrategy strategy)
{
    switch (strategy) {
        case DynamicSPSizeStrategy::Legacy:
            return "legacy";
        case DynamicSPSizeStrategy::LongShortSP8:
            return "long_short_sp8";
        case DynamicSPSizeStrategy::Bucket:
            return "bucket";
    }
    return "unknown";
}

}  // namespace

SPStateManager::SPStateManager(const std::string& engine_id,
                               int                attention_sp,
                               int                num_kvcache_blocks,
                               int                kvcache_block_size,
                               int                max_num_seqs,
                               int                max_num_batched_tokens,
                               int                max_num_recv_seqs,
                               double             reserved_blocks_per_req,
                               int                segment_size,
                               bool               enable_dynamic_sp_size,
                               const std::string& dynamic_sp_size_strategy,
                               int                dynamic_sp_long_request_threshold,
                               int                dynamic_sp_long_request_size,
                               bool               enable_dynamic_sp_bucket_policy,
                               const std::string& dynamic_sp_bucket_policy,
                               double             attention_cost_a,
                               double             attention_cost_b,
                               double             q_cost_a,
                               double             q_cost_b,
                               double             res_cost_a,
                               double             res_cost_b,
                               double             lse_cost_a,
                               double             lse_cost_b,
                               int                q_bytes_per_edge,
                               int                res_bytes_per_edge,
                               int                lse_bytes_per_edge,
                               bool               enable_non_uniform_split,
                               const std::string& sp_master_selector,
                               bool               sp_debug,
                               int                fixed_sp_size):
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    reserved_blocks_per_req_(reserved_blocks_per_req),
    kvcache_block_size_(kvcache_block_size),
    segment_size_(segment_size),
    dynamic_sp_size_strategy_(DynamicSPSizeStrategy::Legacy),
    long_request_sp_threshold_(dynamic_sp_long_request_threshold),
    long_request_sp_size_(dynamic_sp_long_request_size > 0 ? dynamic_sp_long_request_size : attention_sp),
    enable_dynamic_sp_bucket_policy_(enable_dynamic_sp_bucket_policy),
    num_recv_seqs_per_sp_(attention_sp, 0),
    enable_dynamic_sp_size_(enable_dynamic_sp_size),
    cost_model_{
        {attention_cost_a, attention_cost_b},
        {q_cost_a, q_cost_b},
        {res_cost_a, res_cost_b},
        {lse_cost_a, lse_cost_b},
    },
    traffic_model_{q_bytes_per_edge, res_bytes_per_edge, lse_bytes_per_edge},
    enable_non_uniform_split_(enable_non_uniform_split),
    sp_debug_(sp_debug),
    fixed_sp_size_(fixed_sp_size)
{
    Sequence::block_size = kvcache_block_size_;
    // Initialize Strategy
    if (sp_master_selector == "LeastBatch") {
        master_selector_ = SPMasterSelector::LeastBatch;
    }
    else if (sp_master_selector == "LeastCache") {
        master_selector_ = SPMasterSelector::LeastCache;
    }
    else {
        master_selector_ = SPMasterSelector::RoundRobin;
    }

    if (dynamic_sp_size_strategy == "legacy") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Legacy;
    }
    else if (dynamic_sp_size_strategy == "long_short_sp8") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::LongShortSP8;
    }
    else if (dynamic_sp_size_strategy == "bucket") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Bucket;
    }
    else {
        throw std::runtime_error("Unsupported dynamic_sp_size_strategy: " + dynamic_sp_size_strategy);
    }
    dynamic_sp_bucket_policy_ = parse_bucket_policy(dynamic_sp_bucket_policy, attention_sp_);

    if (attention_sp_ <= 0) {
        throw std::runtime_error("attention_sp must be positive to prevent division by zero");
    }
    if (kvcache_block_size_ <= 0) {
        throw std::runtime_error("kvcache_block_size must be positive to prevent division by zero");
    }
    if (fixed_sp_size_ < 0 || fixed_sp_size_ > attention_sp_) {
        throw std::runtime_error("fixed_sp_size must be in [0, attention_sp]");
    }
    if (fixed_sp_size_ > 0
        && (enable_dynamic_sp_size_ || dynamic_sp_size_strategy_ != DynamicSPSizeStrategy::Legacy
            || enable_dynamic_sp_bucket_policy_ || sp_debug_)) {
        throw std::runtime_error("fixed_sp_size cannot be combined with dynamic SP size strategies");
    }

    // Initialize Running Load Counter
    master_seq_counts_.assign(attention_sp_, 0);

    for (int i = 0; i < attention_sp; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }

    initialize_dummy_seqs();

    std::cerr << "[SPStateManager] Initialized with attention_sp=" << attention_sp_
              << ", kvcache_block_size=" << kvcache_block_size_
              << ", reserved_blocks_per_req=" << reserved_blocks_per_req_ << ", segment_size=" << segment_size_
              << ", fixed_sp_size=" << fixed_sp_size_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_name(dynamic_sp_size_strategy_)
              << ", dynamic_sp_long_request_threshold=" << long_request_sp_threshold_
              << ", dynamic_sp_long_request_size=" << long_request_sp_size_
              << ", enable_dynamic_sp_bucket_policy=" << enable_dynamic_sp_bucket_policy_ << std::endl;
}

int SPStateManager::effective_target_sp_size(int requested_sp_size, int num_tokens) const
{
    int target_sp_size = std::max(1, std::min(requested_sp_size, attention_sp_));
    if (fixed_sp_size_ > 0) {
        target_sp_size = std::min(target_sp_size, std::max(1, num_tokens));
    }
    return target_sp_size;
}

std::optional<int> SPStateManager::select_bucket_sp_size(int seq_len) const
{
    if (!enable_dynamic_sp_bucket_policy_) {
        return std::nullopt;
    }
    for (const auto& interval : dynamic_sp_bucket_policy_) {
        if (interval.seq_len_low <= seq_len && seq_len <= interval.seq_len_high) {
            return interval.sp_size;
        }
    }
    return std::nullopt;
}

void SPStateManager::initialize_dummy_seqs()
{
    // Use a fixed seed for reproducibility or random device
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 7999);

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        std::vector<int> token_ids = {dis(gen)};

        auto dummy_seq = std::make_shared<Sequence>(token_ids,
                                                    1.0,   // temperature
                                                    256,   // max_tokens
                                                    false  // ignore_eos
        );
        dummy_seq->active(engine_id_, attention_sp_, 1);
        dummy_seq->block_ctx().master_sp_idx_ = sp_idx;

        dummy_seq->append_token(dis(gen), BlockContextSlot::ACTIVE, sp_idx);

        block_manager[sp_idx]->allocate(*dummy_seq);
        dummy_seqs.push_back(dummy_seq);
    }
}

int SPStateManager::select_master_rank()
{
    if (master_selector_ == SPMasterSelector::RoundRobin) {
        int idx        = sp_rr_counter_;
        sp_rr_counter_ = (sp_rr_counter_ + 1) % attention_sp_;
        return idx;
    }
    else if (master_selector_ == SPMasterSelector::LeastBatch) {
        int best_idx = 0;
        int min_load = std::numeric_limits<int>::max();

        for (int i = 0; i < attention_sp_; ++i) {

            int current_load = master_seq_counts_[i];

            if (current_load < min_load) {
                min_load = current_load;
                best_idx = i;
            }
        }
        return best_idx;
    }
    else if (master_selector_ == SPMasterSelector::LeastCache) {
        int best_idx = 0;
        int max_free = -1;

        for (int i = 0; i < attention_sp_; ++i) {

            int free_blocks = block_manager[i]->num_free_blocks();

            if (free_blocks > max_free) {
                max_free = free_blocks;
                best_idx = i;
            }
        }
        return best_idx;
    }
    return 0;  // Fallback
}

bool SPStateManager::can_append(Sequence& seq, int num_tokens)
{
    int master_sp_idx = seq.block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
    if (block_manager.find(master_sp_idx) == block_manager.end()) {
        return false;
    }
    return block_manager[master_sp_idx]->can_append(seq, num_tokens);
}

bool SPStateManager::may_append(Sequence& seq, int num_tokens)
{
    int master_sp_idx = seq.block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
    if (block_manager.find(master_sp_idx) != block_manager.end()) {
        return block_manager[master_sp_idx]->may_append(seq, num_tokens);
    }
    return false;
}

bool SPStateManager::can_append_on_sp(Sequence& seq, int sp_idx, int num_tokens) const
{
    auto it = block_manager.find(sp_idx);
    return it != block_manager.end() && it->second->can_append(seq, num_tokens);
}

bool SPStateManager::may_append_on_sp(Sequence& seq, int sp_idx, int num_tokens)
{
    auto it = block_manager.find(sp_idx);
    return it != block_manager.end() && it->second->may_append(seq, num_tokens);
}

std::vector<int> SPStateManager::group_used_kv_tokens(const std::vector<std::shared_ptr<Sequence>>& seqs) const
{
    std::vector<int> used(attention_sp_, 0);
    for (const auto& seq : seqs) {
        if (!seq || seq->status == SequenceStatus::FINISHED) {
            continue;
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            used[sp_idx] += seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx);
        }
    }
    return used;
}

std::vector<int> SPStateManager::group_used_kv_blocks(const std::vector<std::shared_ptr<Sequence>>& seqs) const
{
    std::vector<int> used(attention_sp_, 0);
    for (const auto& seq : seqs) {
        if (!seq || seq->status == SequenceStatus::FINISHED) {
            continue;
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int tokens = seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx);
            used[sp_idx] += (tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
        }
    }
    return used;
}

int SPStateManager::estimate_pending_append_capacity(
    int                                           rank,
    const std::vector<std::shared_ptr<Sequence>>& requests,
    const std::vector<std::shared_ptr<Sequence>>& group_sequences) const
{
    auto manager = block_manager.find(rank);
    if (rank < 0 || rank >= attention_sp_ || manager == block_manager.end()) {
        return 0;
    }

    int free_blocks = manager->second->num_free_blocks();
    // Reservations for the still-pending token (and the next sampled token)
    // can be discarded before an iteration plan is committed. Add them back
    // to the read-only simulation, then charge the selected target afresh.
    for (const auto& seq : group_sequences) {
        if (!seq || seq->status == SequenceStatus::FINISHED) {
            continue;
        }
        int committed        = seq->committed_context_len(BlockContextSlot::ACTIVE, rank);
        int committed_blocks = (committed + kvcache_block_size_ - 1) / kvcache_block_size_;
        int table_blocks     = static_cast<int>(seq->block_table(BlockContextSlot::ACTIVE, rank).size());
        free_blocks += std::max(0, table_blocks - committed_blocks);
    }

    int accepted = 0;
    for (const auto& seq : requests) {
        if (!seq || seq->status != SequenceStatus::RUNNING
            || !seq->block_ctx(BlockContextSlot::ACTIVE).pending_token_present_) {
            break;
        }
        if (accepted >= std::min(max_num_seqs_, max_num_batched_tokens_)) {
            break;
        }

        int committed     = seq->committed_context_len(BlockContextSlot::ACTIVE, rank);
        int before_blocks = (committed + kvcache_block_size_ - 1) / kvcache_block_size_;
        // One slot is for the current pending input. NanoDeploy also reserves
        // the slot for the sampled token that becomes pending at postprocess.
        int after_blocks  = (committed + 2 + kvcache_block_size_ - 1) / kvcache_block_size_;
        int needed_blocks = after_blocks - before_blocks;
        if (free_blocks < needed_blocks) {
            break;
        }
        free_blocks -= needed_blocks;

        int reserve_headroom = static_cast<int>(std::ceil((accepted + 1) * reserved_blocks_per_req_));
        if (free_blocks < reserve_headroom) {
            break;
        }
        accepted++;
    }
    return accepted;
}

SPStateManager::LSDecodeMasterPlan
SPStateManager::plan_iteration_masters_source_greedy(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                     const std::vector<int>&                       allocation,
                                                     const std::vector<int>&                       extra_ranks,
                                                     int                                           batch_per_master,
                                                     bool enable_memory_scale_up) const
{
    LSDecodeMasterPlan result;
    result.group_used_kv_tokens = group_used_kv_tokens(requests);
    result.group_used_kv_blocks = group_used_kv_blocks(requests);
    if (requests.empty()) {
        result.success    = true;
        result.allocation = allocation;
        return result;
    }
    if (batch_per_master <= 0) {
        result.failure_reason = "batch_per_master must be positive";
        return result;
    }

    auto normalize_ranks = [&](const std::vector<int>& ranks) {
        std::vector<int> normalized;
        for (int rank : ranks) {
            if (rank >= 0 && rank < attention_sp_
                && std::find(normalized.begin(), normalized.end(), rank) == normalized.end()) {
                normalized.push_back(rank);
            }
        }
        return normalized;
    };

    std::vector<int> current_allocation = normalize_ranks(allocation);
    std::vector<int> extras             = normalize_ranks(extra_ranks);
    extras.erase(std::remove_if(extras.begin(),
                                extras.end(),
                                [&](int rank) {
                                    return std::find(current_allocation.begin(), current_allocation.end(), rank)
                                           != current_allocation.end();
                                }),
                 extras.end());
    std::stable_sort(extras.begin(), extras.end(), [&](int lhs, int rhs) {
        int lhs_capacity = estimate_pending_append_capacity(lhs, requests, requests);
        int rhs_capacity = estimate_pending_append_capacity(rhs, requests, requests);
        if (lhs_capacity != rhs_capacity) {
            return lhs_capacity > rhs_capacity;
        }
        return lhs < rhs;
    });

    bool      compute_scaled = false;
    bool      memory_scaled  = false;
    const int max_restarts   = attention_sp_ + 1;
    for (int restart = 0; restart < max_restarts; ++restart) {
        auto             candidates = current_allocation;
        std::vector<int> recv_counts(attention_sp_, 0);
        for (const auto& seq : requests) {
            int master = seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (sp_idx != master && seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0) {
                    recv_counts[sp_idx]++;
                }
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
            if (result.group_used_kv_tokens[lhs] != result.group_used_kv_tokens[rhs]) {
                return result.group_used_kv_tokens[lhs] > result.group_used_kv_tokens[rhs];
            }
            if (result.group_used_kv_blocks[lhs] != result.group_used_kv_blocks[rhs]) {
                return result.group_used_kv_blocks[lhs] > result.group_used_kv_blocks[rhs];
            }
            if (recv_counts[lhs] != recv_counts[rhs]) {
                return recv_counts[lhs] < recv_counts[rhs];
            }
            return lhs < rhs;
        });

        std::vector<int> masters;
        std::vector<int> chunks;
        std::vector<int> assignments(requests.size(), -1);
        std::vector<int> planned_remote_recv(attention_sp_, 0);
        size_t           remaining_begin = 0;
        size_t           candidate_begin = 0;
        bool             need_restart    = false;
        bool             failed          = false;
        std::string      failure;

        auto receiver_prefix_capacity = [&](int rank, size_t begin) {
            std::vector<int> simulated = planned_remote_recv;
            int              accepted  = 0;
            for (size_t request_idx = begin; request_idx < requests.size(); ++request_idx) {
                bool feasible = true;
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    if (owner != rank
                        && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0
                        && simulated[owner] + 1 > max_num_recv_seqs_) {
                        feasible = false;
                        break;
                    }
                }
                if (!feasible) {
                    break;
                }
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    if (owner != rank
                        && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0) {
                        simulated[owner]++;
                    }
                }
                accepted++;
            }
            return accepted;
        };

        while (remaining_begin < requests.size()) {
            std::vector<size_t> append_capable;
            for (size_t idx = candidate_begin; idx < candidates.size(); ++idx) {
                std::vector<std::shared_ptr<Sequence>> suffix(
                    requests.begin() + static_cast<std::ptrdiff_t>(remaining_begin), requests.end());
                int rank = candidates[idx];
                if (estimate_pending_append_capacity(rank, suffix, requests) > 0
                    && receiver_prefix_capacity(rank, remaining_begin) > 0) {
                    append_capable.push_back(idx);
                }
            }

            if (append_capable.empty()) {
                if (enable_memory_scale_up && !extras.empty()) {
                    int rank = extras.front();
                    extras.erase(extras.begin());
                    current_allocation.push_back(rank);
                    result.new_allocation_ranks.push_back(rank);
                    memory_scaled = true;
                    need_restart  = true;
                }
                else {
                    failed  = true;
                    failure = "append capacity cannot cover remaining requests";
                }
                break;
            }

            int n_left    = static_cast<int>(append_capable.size());
            int remaining = static_cast<int>(requests.size() - remaining_begin);
            if (remaining / n_left > batch_per_master && !extras.empty()) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                compute_scaled = true;
                need_restart   = true;
                break;
            }

            size_t                                 rank_pos = append_capable.front();
            int                                    rank     = candidates[rank_pos];
            std::vector<std::shared_ptr<Sequence>> suffix(
                requests.begin() + static_cast<std::ptrdiff_t>(remaining_begin), requests.end());
            int capacity     = estimate_pending_append_capacity(rank, suffix, requests);
            capacity         = std::min(capacity, receiver_prefix_capacity(rank, remaining_begin));
            int target_chunk = std::max(remaining / n_left, batch_per_master);
            int chunk        = std::min({remaining, target_chunk, capacity});
            candidate_begin  = rank_pos + 1;
            if (chunk <= 0) {
                continue;
            }
            masters.push_back(rank);
            chunks.push_back(chunk);
            for (int i = 0; i < chunk; ++i) {
                size_t request_idx       = remaining_begin + i;
                assignments[request_idx] = rank;
                for (int owner = 0; owner < attention_sp_; ++owner) {
                    if (owner != rank
                        && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0) {
                        planned_remote_recv[owner]++;
                    }
                }
            }
            remaining_begin += chunk;
        }

        if (need_restart) {
            continue;
        }
        if (failed) {
            result.failure_reason = failure;
            return result;
        }

        result.success               = true;
        result.allocation            = current_allocation;
        result.master_ranks          = std::move(masters);
        result.master_batch_sizes    = std::move(chunks);
        result.sequence_master_ranks = std::move(assignments);
        if (compute_scaled && memory_scaled) {
            result.scale_reason = "compute+memory";
        }
        else if (compute_scaled) {
            result.scale_reason = "compute";
        }
        else if (memory_scaled) {
            result.scale_reason = "memory";
        }
        return result;
    }

    result.failure_reason = "master planning exceeded restart bound";
    return result;
}

bool SPStateManager::validate_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                    const LSDecodeMasterPlan&                     plan,
                                                    std::string*                                  error) const
{
    auto fail = [&](const std::string& message) {
        if (error) {
            *error = message;
        }
        return false;
    };
    if (!plan.success) {
        return fail(plan.failure_reason.empty() ? "planner returned failure" : plan.failure_reason);
    }
    if (plan.sequence_master_ranks.size() != requests.size()) {
        return fail("sequence assignment count mismatch");
    }

    std::unordered_set<const Sequence*> unique_requests;
    for (const auto& request : requests) {
        if (!request) {
            return fail("iteration request is null");
        }
        if (!unique_requests.insert(request.get()).second) {
            return fail("iteration request list contains a duplicate sequence");
        }
        const auto& ctx = request->block_ctx(BlockContextSlot::ACTIVE);
        if (request->status != SequenceStatus::RUNNING || !ctx.pending_token_present_
            || ctx.pending_token_target_sp_ < 0 || ctx.pending_token_target_sp_ >= attention_sp_
            || ctx.num_dispatched_tokens[ctx.pending_token_target_sp_] <= 0) {
            return fail("iteration request has an invalid pending frontier");
        }
    }

    std::vector<std::vector<std::shared_ptr<Sequence>>> assigned(attention_sp_);
    std::vector<int>                                    remote_recv(attention_sp_, 0);
    for (size_t idx = 0; idx < requests.size(); ++idx) {
        int master = plan.sequence_master_ranks[idx];
        if (master < 0 || master >= attention_sp_
            || std::find(plan.allocation.begin(), plan.allocation.end(), master) == plan.allocation.end()) {
            return fail("iteration master is outside group allocation");
        }
        assigned[master].push_back(requests[idx]);
        if (static_cast<int>(assigned[master].size()) > std::min(max_num_seqs_, max_num_batched_tokens_)) {
            return fail("master batch exceeds Decode metadata capacity");
        }
        for (int owner = 0; owner < attention_sp_; ++owner) {
            if (owner != master && requests[idx]->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0) {
                remote_recv[owner]++;
            }
        }
    }
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (!assigned[rank].empty()
            && estimate_pending_append_capacity(rank, assigned[rank], requests)
                   < static_cast<int>(assigned[rank].size())) {
            return fail("master append capacity changed during validation");
        }
        if (remote_recv[rank] > max_num_recv_seqs_) {
            return fail("remote attention capacity exceeds max_num_recv_seqs");
        }
    }
    return true;
}

void SPStateManager::set_decode_master(Sequence& seq, int master_sp_idx)
{
    if (master_sp_idx < 0 || master_sp_idx >= attention_sp_) {
        throw std::runtime_error("decode master_sp_idx out of range");
    }
    auto& ctx        = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   old_master = ctx.master_sp_idx_;
    if (old_master == master_sp_idx) {
        return;
    }
    if (old_master >= 0 && old_master < attention_sp_ && master_seq_counts_[old_master] > 0) {
        master_seq_counts_[old_master]--;
    }
    master_seq_counts_[master_sp_idx]++;
    ctx.master_sp_idx_ = master_sp_idx;
    cached_running_state_.reset();
}

bool SPStateManager::reassign_pending_append(Sequence& seq, int target_sp_idx)
{
    if (target_sp_idx < 0 || target_sp_idx >= attention_sp_) {
        return false;
    }
    auto& ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    if (!ctx.pending_token_present_ || ctx.pending_token_target_sp_ < 0
        || ctx.pending_token_target_sp_ >= attention_sp_) {
        return false;
    }

    std::vector<std::shared_ptr<Sequence>> request_view;
    request_view.emplace_back(&seq, [](Sequence*) {});
    // Complete the read-only capacity simulation before releasing the old
    // frontier. With the single-threaded state manager, the following trim and
    // reservation operations are then deterministic and cannot fail for an
    // expected capacity condition.
    if (estimate_pending_append_capacity(target_sp_idx, request_view, request_view) < 1) {
        return false;
    }

    int old_target    = ctx.pending_token_target_sp_;
    int old_committed = seq.committed_context_len(BlockContextSlot::ACTIVE, old_target);
    block_manager[old_target]->trim_blocks_to_token_count(seq, BlockContextSlot::ACTIVE, old_committed);
    if (ctx.num_dispatched_tokens[old_target] <= 0) {
        return false;
    }
    ctx.num_dispatched_tokens[old_target]--;
    ctx.pending_token_present_   = false;
    ctx.pending_token_target_sp_ = -1;

    int target_committed = ctx.num_dispatched_tokens[target_sp_idx];
    block_manager[target_sp_idx]->trim_blocks_to_token_count(seq, BlockContextSlot::ACTIVE, target_committed);
    if (!can_append_on_sp(seq, target_sp_idx, 1) || !may_append_on_sp(seq, target_sp_idx, 1)) {
        throw std::runtime_error("pending append preflight diverged while installing input token");
    }
    ctx.num_dispatched_tokens[target_sp_idx]++;
    ctx.pending_token_present_   = true;
    ctx.pending_token_target_sp_ = target_sp_idx;
    set_decode_master(seq, target_sp_idx);

    // Reserve the sampled token that postprocess will append as the pending
    // input for the next iteration. This preserves the existing NanoDeploy
    // block-table contract while historical ownership remains untouched.
    if (!can_append_on_sp(seq, target_sp_idx, 1) || !may_append_on_sp(seq, target_sp_idx, 1)) {
        throw std::runtime_error("pending append preflight diverged while reserving sampled token");
    }
    return true;
}

bool SPStateManager::commit_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                                  const LSDecodeMasterPlan&                     plan)
{
    std::string error;
    if (!validate_iteration_master_plan(requests, plan, &error)) {
        return false;
    }

    // Phase 1 releases every stale pending/output reservation before any new
    // destination consumes capacity. This makes swaps between full ranks
    // deterministic and prevents request-order-dependent partial commits.
    for (const auto& seq : requests) {
        auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
        if (!ctx.pending_token_present_ || ctx.pending_token_target_sp_ < 0
            || ctx.pending_token_target_sp_ >= attention_sp_) {
            return false;
        }
        int old_target = ctx.pending_token_target_sp_;
        int committed  = seq->committed_context_len(BlockContextSlot::ACTIVE, old_target);
        block_manager[old_target]->trim_blocks_to_token_count(*seq, BlockContextSlot::ACTIVE, committed);
        if (ctx.num_dispatched_tokens[old_target] <= 0) {
            return false;
        }
        ctx.num_dispatched_tokens[old_target]--;
        ctx.pending_token_present_   = false;
        ctx.pending_token_target_sp_ = -1;
    }

    // Phase 2 installs all new pending destinations and reserves the next
    // sampled token. Historical counts and tables are never changed.
    for (size_t idx = 0; idx < requests.size(); ++idx) {
        auto& seq       = *requests[idx];
        auto& ctx       = seq.block_ctx(BlockContextSlot::ACTIVE);
        int   target    = plan.sequence_master_ranks[idx];
        int   committed = ctx.num_dispatched_tokens[target];
        block_manager[target]->trim_blocks_to_token_count(seq, BlockContextSlot::ACTIVE, committed);
        if (!can_append_on_sp(seq, target, 1) || !may_append_on_sp(seq, target, 1)) {
            throw std::runtime_error("validated LS plan diverged while installing input token");
        }
        ctx.num_dispatched_tokens[target]++;
        ctx.pending_token_present_   = true;
        ctx.pending_token_target_sp_ = target;
        set_decode_master(seq, target);
        if (!can_append_on_sp(seq, target, 1) || !may_append_on_sp(seq, target, 1)) {
            throw std::runtime_error("validated LS plan diverged while reserving sampled token");
        }
    }
    rebuild_decode_role_counters();
    return true;
}

std::shared_ptr<SPStateManager::LSKVConsolidationPlan>
SPStateManager::plan_kv_consolidation(uint64_t                                      transaction_id,
                                      uint64_t                                      group_id,
                                      int                                           dp_idx,
                                      const std::vector<std::shared_ptr<Sequence>>& sequences,
                                      int                                           source_rank,
                                      const std::vector<int>&                       retained_ranks)
{
    auto plan            = std::make_shared<LSKVConsolidationPlan>();
    plan->transaction_id = transaction_id;
    plan->group_id       = group_id;
    plan->dp_idx         = dp_idx;
    plan->source_rank    = source_rank;
    plan->retained_ranks = retained_ranks;

    auto reject = [&](const std::string& reason) {
        plan->success        = false;
        plan->failure_reason = reason;
        plan->state          = LSKVConsolidationPlan::State::REJECTED;
        return plan;
    };

    if (dp_idx != dp_idx_) {
        return reject("KV consolidation DP does not match the state manager");
    }
    if (source_rank < 0 || source_rank >= attention_sp_) {
        return reject("KV consolidation source rank is out of range");
    }
    if (retained_ranks.empty()) {
        return reject("KV consolidation cannot release the last allocated rank");
    }

    std::unordered_set<int> unique_retained;
    for (int rank : retained_ranks) {
        if (rank < 0 || rank >= attention_sp_ || rank == source_rank) {
            return reject("KV consolidation retained rank is invalid");
        }
        if (!unique_retained.insert(rank).second) {
            return reject("KV consolidation retained ranks contain a duplicate");
        }
    }

    struct Assignment {
        int dst_rank          = -1;
        int num_tokens        = 0;
        int dst_logical_start = 0;
    };
    struct Draft {
        LSKVConsolidationPlan::SequenceStage stage;
        std::vector<Assignment>              assignments;
        std::vector<int>                     additional_blocks;
        int                                  source_tokens = 0;
    };

    std::vector<int> free_blocks(attention_sp_, 0);
    for (int rank : retained_ranks) {
        free_blocks[rank] = block_manager.at(rank)->num_free_blocks();
    }

    std::vector<Draft>                  drafts;
    std::unordered_set<const Sequence*> unique_sequences;
    bool                                found_source_kv = false;
    for (const auto& sequence : sequences) {
        if (!sequence || !unique_sequences.insert(sequence.get()).second) {
            return reject("KV consolidation sequence list is null or contains a duplicate");
        }
        if (sequence->status != SequenceStatus::RUNNING) {
            return reject("KV consolidation only supports RUNNING sequences");
        }
        plan->group_sequence_ids.push_back(sequence->seq_id);

        const auto& ctx = sequence->block_ctx(BlockContextSlot::ACTIVE);
        if (ctx.dp_idx_ != dp_idx || static_cast<int>(ctx.sp_block_table.size()) != attention_sp_
            || static_cast<int>(ctx.num_dispatched_tokens.size()) != attention_sp_) {
            return reject("KV consolidation sequence placement is inconsistent with the target DP");
        }
        plan->sequence_snapshots.push_back({sequence, ctx, sequence->status});
        if (ctx.master_sp_idx_ == source_rank
            || (ctx.pending_token_present_ && ctx.pending_token_target_sp_ == source_rank)) {
            return reject("KV consolidation source rank is an active or pending Decode master");
        }

        int source_tokens = sequence->committed_context_len(BlockContextSlot::ACTIVE, source_rank);
        if (source_tokens == 0) {
            if (!ctx.sp_block_table[source_rank].empty()) {
                return reject("KV consolidation source owns blocks without committed KV");
            }
            continue;
        }
        found_source_kv = true;
        if (static_cast<int>(ctx.sp_block_table[source_rank].size())
            < (source_tokens + kvcache_block_size_ - 1) / kvcache_block_size_) {
            return reject("KV consolidation source block table is shorter than committed KV");
        }

        Draft draft;
        draft.stage.sequence       = sequence;
        draft.stage.old_context    = ctx;
        draft.stage.staged_context = ctx;
        draft.stage.source_blocks.assign(ctx.sp_block_table[source_rank].begin(),
                                         ctx.sp_block_table[source_rank].end());
        draft.additional_blocks.assign(attention_sp_, 0);
        draft.source_tokens = source_tokens;

        std::vector<int> candidates = retained_ranks;
        std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
            bool lhs_master = lhs == ctx.master_sp_idx_;
            bool rhs_master = rhs == ctx.master_sp_idx_;
            if (lhs_master != rhs_master) {
                return lhs_master;
            }
            int lhs_tokens = sequence->committed_context_len(BlockContextSlot::ACTIVE, lhs);
            int rhs_tokens = sequence->committed_context_len(BlockContextSlot::ACTIVE, rhs);
            if (lhs_tokens != rhs_tokens) {
                return lhs_tokens > rhs_tokens;
            }
            return lhs < rhs;
        });

        auto movable_capacity = [&](int rank) {
            int committed = sequence->committed_context_len(BlockContextSlot::ACTIVE, rank);
            int frontier  = ctx.pending_token_present_ && ctx.pending_token_target_sp_ == rank ? 2 : 0;
            int blocks    = static_cast<int>(ctx.sp_block_table[rank].size()) + free_blocks[rank];
            return std::max(0, blocks * kvcache_block_size_ - committed - frontier);
        };

        int whole_destination = -1;
        for (int rank : candidates) {
            if (movable_capacity(rank) >= source_tokens) {
                whole_destination = rank;
                break;
            }
        }

        int remaining = source_tokens;
        for (int rank : candidates) {
            if (whole_destination >= 0 && rank != whole_destination) {
                continue;
            }
            int capacity = movable_capacity(rank);
            int moved    = std::min(remaining, capacity);
            if (moved <= 0) {
                continue;
            }

            int committed       = sequence->committed_context_len(BlockContextSlot::ACTIVE, rank);
            int frontier        = ctx.pending_token_present_ && ctx.pending_token_target_sp_ == rank ? 2 : 0;
            int required_blocks = (committed + moved + frontier + kvcache_block_size_ - 1) / kvcache_block_size_;
            int existing_blocks = static_cast<int>(ctx.sp_block_table[rank].size());
            int additional      = std::max(0, required_blocks - existing_blocks);
            if (additional > free_blocks[rank]) {
                return reject("KV consolidation capacity simulation diverged");
            }

            draft.assignments.push_back({rank, moved, committed});
            draft.additional_blocks[rank] = additional;
            draft.stage.staged_context.num_dispatched_tokens[rank] += moved;
            free_blocks[rank] -= additional;
            remaining -= moved;
            if (remaining == 0) {
                break;
            }
        }
        if (remaining != 0) {
            return reject("insufficient destination capacity for KV consolidation");
        }

        draft.stage.staged_context.num_dispatched_tokens[source_rank] = 0;
        draft.stage.staged_context.sp_block_table[source_rank].clear();
        drafts.push_back(std::move(draft));
    }

    if (!found_source_kv) {
        return reject("KV consolidation source rank has no committed KV");
    }

    std::unordered_map<const Sequence*, const BlockContext*> staged_contexts;
    staged_contexts.reserve(drafts.size());
    for (const auto& draft : drafts) {
        staged_contexts.emplace(draft.stage.sequence.get(), &draft.stage.staged_context);
    }
    std::vector<int> staged_remote_recv(attention_sp_, 0);
    for (const auto& sequence : running) {
        if (!sequence || sequence->status != SequenceStatus::RUNNING) {
            continue;
        }
        const auto  staged = staged_contexts.find(sequence.get());
        const auto& ctx =
            staged == staged_contexts.end() ? sequence->block_ctx(BlockContextSlot::ACTIVE) : *staged->second;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            int committed = ctx.num_dispatched_tokens[rank];
            if (ctx.pending_token_present_ && ctx.pending_token_target_sp_ == rank) {
                committed--;
            }
            if (rank != ctx.master_sp_idx_ && committed > 0) {
                staged_remote_recv[rank]++;
            }
        }
    }
    if (std::any_of(staged_remote_recv.begin(), staged_remote_recv.end(), [&](int count) {
            return count > max_num_recv_seqs_;
        })) {
        return reject("KV consolidation would exceed remote attention capacity");
    }

    size_t reserve_request_count = 0;
    for (const auto& draft : drafts) {
        reserve_request_count += static_cast<size_t>(std::count_if(
            draft.additional_blocks.begin(), draft.additional_blocks.end(), [](int count) { return count > 0; }));
    }
    std::vector<std::pair<int, std::vector<int>>> reserved_for_cleanup;
    reserved_for_cleanup.reserve(reserve_request_count);
    plan->sequence_stages.reserve(drafts.size());

    try {
        for (auto& draft : drafts) {
            for (int rank : retained_ranks) {
                int count = draft.additional_blocks[rank];
                if (count == 0) {
                    continue;
                }
                auto reserved = block_manager.at(rank)->reserve_blocks(count);
                reserved_for_cleanup.emplace_back(rank, std::move(reserved));
                const auto& tracked = reserved_for_cleanup.back().second;
                auto&       table   = draft.stage.staged_context.sp_block_table[rank];
                table.insert(table.end(), tracked.begin(), tracked.end());
                draft.stage.reserved_blocks.emplace_back(rank, tracked);
            }

            auto& locations = draft.stage.staged_context.block_location;
            locations.clear();
            size_t location_count = 0;
            for (const auto& table : draft.stage.staged_context.sp_block_table) {
                location_count += table.size();
            }
            locations.reserve(location_count);
            for (int rank = 0; rank < attention_sp_; ++rank) {
                for (int block_id : draft.stage.staged_context.sp_block_table[rank]) {
                    locations.emplace_back(rank, block_id);
                }
            }

            int source_cursor = 0;
            for (const auto& assignment : draft.assignments) {
                int destination_cursor = assignment.dst_logical_start;
                int assignment_left    = assignment.num_tokens;
                while (assignment_left > 0) {
                    int source_block_index = source_cursor / kvcache_block_size_;
                    int source_offset      = source_cursor % kvcache_block_size_;
                    int dest_block_index   = destination_cursor / kvcache_block_size_;
                    int dest_offset        = destination_cursor % kvcache_block_size_;
                    int length             = std::min(
                        {assignment_left, kvcache_block_size_ - source_offset, kvcache_block_size_ - dest_offset});
                    plan->moves.push_back(
                        {draft.stage.sequence->seq_id,
                         dp_idx,
                         source_rank,
                         assignment.dst_rank,
                         draft.stage.old_context.sp_block_table[source_rank][source_block_index],
                         source_offset,
                         draft.stage.staged_context.sp_block_table[assignment.dst_rank][dest_block_index],
                         dest_offset,
                         length});
                    source_cursor += length;
                    destination_cursor += length;
                    assignment_left -= length;
                }
            }
            if (source_cursor != draft.source_tokens) {
                throw std::runtime_error("KV consolidation move generation lost source tokens");
            }
            plan->num_tokens += draft.source_tokens;
            plan->sequence_stages.push_back(std::move(draft.stage));
        }
        reserved_for_cleanup.clear();
    }
    catch (...) {
        for (const auto& [rank, block_ids] : reserved_for_cleanup) {
            block_manager.at(rank)->release_blocks(block_ids);
        }
        throw;
    }

    plan->success = true;
    plan->state   = LSKVConsolidationPlan::State::RESERVED;
    return plan;
}

bool SPStateManager::commit_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan)
{
    if (!plan || !plan->success || plan->state != LSKVConsolidationPlan::State::RESERVED) {
        return false;
    }

    auto contexts_equal = [](const BlockContext& lhs, const BlockContext& rhs) {
        return lhs.engine_id_ == rhs.engine_id_ && lhs.dp_idx_ == rhs.dp_idx_
               && lhs.master_sp_idx_ == rhs.master_sp_idx_ && lhs.attention_sp_ == rhs.attention_sp_
               && lhs.attention_dp_ == rhs.attention_dp_ && lhs.pending_token_present_ == rhs.pending_token_present_
               && lhs.pending_token_target_sp_ == rhs.pending_token_target_sp_
               && lhs.block_location == rhs.block_location && lhs.sp_block_table == rhs.sp_block_table
               && lhs.num_dispatched_tokens == rhs.num_dispatched_tokens;
    };
    for (const auto& snapshot : plan->sequence_snapshots) {
        if (!snapshot.sequence || snapshot.sequence->status != snapshot.status
            || !contexts_equal(snapshot.sequence->block_ctx(BlockContextSlot::ACTIVE), snapshot.context)) {
            return false;
        }
    }

    for (auto& stage : plan->sequence_stages) {
        std::swap(stage.sequence->block_ctx(BlockContextSlot::ACTIVE), stage.staged_context);
    }
    // From this point onward the transaction is committed. A source-block
    // reclamation failure is engine-fatal and must never trigger ABORT, since
    // the reserved destination blocks are now reachable from ACTIVE metadata.
    plan->state = LSKVConsolidationPlan::State::COMMITTED;
    for (const auto& stage : plan->sequence_stages) {
        block_manager.at(plan->source_rank)->release_blocks(stage.source_blocks);
    }
    rebuild_decode_role_counters();
    return true;
}

void SPStateManager::abort_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan)
{
    if (!plan || plan->state != LSKVConsolidationPlan::State::RESERVED) {
        return;
    }
    for (auto& stage : plan->sequence_stages) {
        for (const auto& [rank, block_ids] : stage.reserved_blocks) {
            block_manager.at(rank)->release_blocks(block_ids);
        }
        stage.reserved_blocks.clear();
    }
    plan->state = LSKVConsolidationPlan::State::ABORTED;
}

int SPStateManager::get_active_master_count(const std::vector<std::shared_ptr<Sequence>>& seqs) const
{
    std::set<int> masters;
    for (const auto& seq : seqs) {
        if (seq && seq->status == SequenceStatus::RUNNING) {
            masters.insert(seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_);
        }
    }
    return static_cast<int>(masters.size());
}

int SPStateManager::get_kv_participant_count(const std::vector<std::shared_ptr<Sequence>>& seqs) const
{
    auto used = group_used_kv_tokens(seqs);
    return static_cast<int>(std::count_if(used.begin(), used.end(), [](int value) { return value > 0; }));
}

void SPStateManager::rebuild_decode_role_counters()
{
    std::fill(master_seq_counts_.begin(), master_seq_counts_.end(), 0);
    std::fill(num_recv_seqs_per_sp_.begin(), num_recv_seqs_per_sp_.end(), 0);
    num_running_seqs_   = 0;
    num_running_tokens_ = 0;
    for (const auto& seq : running) {
        if (!seq || seq->status != SequenceStatus::RUNNING) {
            continue;
        }
        const auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
        if (ctx.master_sp_idx_ >= 0 && ctx.master_sp_idx_ < attention_sp_) {
            master_seq_counts_[ctx.master_sp_idx_]++;
        }
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (sp_idx != ctx.master_sp_idx_ && seq->committed_context_len(BlockContextSlot::ACTIVE, sp_idx) > 0) {
                num_recv_seqs_per_sp_[sp_idx]++;
            }
        }
        num_running_seqs_++;
        num_running_tokens_ += seq->num_tokens;
    }
    cached_running_state_.reset();
}

void SPStateManager::add_communication(PlanningState&          state,
                                       int                     master_sp_idx,
                                       const std::vector<int>& dispatched_tokens) const
{
    std::vector<int> participants;
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (dispatched_tokens[sp_idx] > 0) {
            participants.push_back(sp_idx);
        }
    }
    if (participants.size() <= 1) {
        return;
    }

    for (int sp_idx : participants) {
        if (sp_idx == master_sp_idx) {
            continue;
        }
        state.send_q[master_sp_idx] += traffic_model_.q_bytes_per_edge;
        state.recv_q[sp_idx] += traffic_model_.q_bytes_per_edge;
        state.send_res[sp_idx] += traffic_model_.res_bytes_per_edge;
        state.recv_res[master_sp_idx] += traffic_model_.res_bytes_per_edge;
        state.send_lse[sp_idx] += traffic_model_.lse_bytes_per_edge;
        state.recv_lse[master_sp_idx] += traffic_model_.lse_bytes_per_edge;
    }
}

SPStateManager::PlanningState SPStateManager::build_running_state_snapshot() const
{
    PlanningState state;
    state.tokens.assign(attention_sp_, 0);
    state.master_counts = master_seq_counts_;
    state.recv_counts   = num_recv_seqs_per_sp_;
    state.free_blocks.assign(attention_sp_, 0);
    state.batch_tokens.assign(attention_sp_, 0);
    state.send_q.assign(attention_sp_, 0);
    state.recv_q.assign(attention_sp_, 0);
    state.send_res.assign(attention_sp_, 0);
    state.recv_res.assign(attention_sp_, 0);
    state.send_lse.assign(attention_sp_, 0);
    state.recv_lse.assign(attention_sp_, 0);
    state.rr_cursor = sp_rr_counter_;

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        state.free_blocks[sp_idx] = block_manager.at(sp_idx)->num_free_blocks();
    }

    for (const auto& seq : running) {
        const auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            state.tokens[sp_idx] += ctx.num_dispatched_tokens[sp_idx];
        }
        add_communication(state, ctx.master_sp_idx_, ctx.num_dispatched_tokens);
    }

    state.group_max = *std::max_element(state.tokens.begin(), state.tokens.end());
    return state;
}

void SPStateManager::begin_decode_planning() const
{
    if (!cached_running_state_.has_value()) {
        cached_running_state_ = build_running_state_snapshot();
    }
}

void SPStateManager::end_decode_planning() const
{
    cached_running_state_.reset();
}

std::optional<SPStateManager::DecodeBatchPlan>
SPStateManager::plan_decode_batch(const std::vector<std::shared_ptr<Sequence>>& pending_seqs) const
{
    auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

    auto allowed_sp_sizes = [&]() {
        std::vector<int> sizes;
        if (fixed_sp_size_ > 0) {
            sizes.push_back(fixed_sp_size_);
            return sizes;
        }
        sizes.push_back(1);
        for (int s = 2; s <= attention_sp_; s *= 2) {
            sizes.push_back(s);
        }
        if (sizes.back() != attention_sp_) {
            sizes.push_back(attention_sp_);
        }
        return sizes;
    }();

    auto count_active_ranks = [&](const PlannedPlacement& placement) {
        int active_ranks = 0;
        for (int count : placement.num_dispatched_tokens) {
            if (count > 0) {
                active_ranks++;
            }
        }
        return active_ranks;
    };

    auto count_extra_participants = [&](const std::vector<PlannedPlacement>& placements) {
        int extra = 0;
        for (const auto& placement : placements) {
            extra += std::max(0, count_active_ranks(placement) - 1);
        }
        return extra;
    };

    auto total_overflow = [&](const PlanningState& state, int running_group_max) {
        int overflow = 0;
        for (int load : state.tokens) {
            overflow += std::max(0, load - running_group_max);
        }
        return overflow;
    };

    auto max_send_or_recv = [&](const std::vector<int>& send, const std::vector<int>& recv) {
        int max_bytes = 0;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            max_bytes = std::max(max_bytes, std::max(send[sp_idx], recv[sp_idx]));
        }
        return static_cast<double>(max_bytes);
    };

    auto estimate_latency = [&](const PlanningState& state) {
        LatencyBreakdown breakdown;
        breakdown.max_tokens    = static_cast<double>(*std::max_element(state.tokens.begin(), state.tokens.end()));
        breakdown.max_q_bytes   = max_send_or_recv(state.send_q, state.recv_q);
        breakdown.max_res_bytes = max_send_or_recv(state.send_res, state.recv_res);
        breakdown.max_lse_bytes = max_send_or_recv(state.send_lse, state.recv_lse);
        breakdown.attention     = cost_model_.attention.predict(breakdown.max_tokens);
        breakdown.q             = breakdown.max_q_bytes > 0.0 ? cost_model_.q.predict(breakdown.max_q_bytes) : 0.0;
        breakdown.res   = breakdown.max_res_bytes > 0.0 ? cost_model_.res.predict(breakdown.max_res_bytes) : 0.0;
        breakdown.lse   = breakdown.max_lse_bytes > 0.0 ? cost_model_.lse.predict(breakdown.max_lse_bytes) : 0.0;
        breakdown.total = breakdown.attention + breakdown.q + breakdown.res + breakdown.lse;
        return breakdown;
    };

    auto better_candidate = [&](const LatencyBreakdown& lhs_latency,
                                int                     lhs_overflow,
                                int                     lhs_extra,
                                const LatencyBreakdown& rhs_latency,
                                int                     rhs_overflow,
                                int                     rhs_extra) {
        constexpr double kEps = 1e-9;
        if (std::fabs(lhs_latency.total - rhs_latency.total) > kEps) {
            return lhs_latency.total < rhs_latency.total;
        }
        if (std::fabs(lhs_latency.max_tokens - rhs_latency.max_tokens) > kEps) {
            return lhs_latency.max_tokens < rhs_latency.max_tokens;
        }
        if (lhs_overflow != rhs_overflow) {
            return lhs_overflow < rhs_overflow;
        }
        return lhs_extra < rhs_extra;
    };

    auto choose_master = [&](PlanningState& state, const Sequence& seq) -> int {
        int uncached_tokens = seq.num_tokens - seq.num_cached_tokens;

        if (master_selector_ == SPMasterSelector::RoundRobin) {
            for (int attempt = 0; attempt < attention_sp_; ++attempt) {
                int candidate = (state.rr_cursor + attempt) % attention_sp_;
                if (state.master_counts[candidate] + 1 <= max_num_seqs_
                    && state.batch_tokens[candidate] + uncached_tokens < max_num_batched_tokens_) {
                    state.rr_cursor = (candidate + 1) % attention_sp_;
                    return candidate;
                }
            }
            return -1;
        }

        int                            best_idx = -1;
        std::tuple<int, int, int, int> best_score;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (state.master_counts[sp_idx] + 1 > max_num_seqs_) {
                continue;
            }
            if (state.batch_tokens[sp_idx] + uncached_tokens >= max_num_batched_tokens_) {
                continue;
            }

            std::tuple<int, int, int, int> score;
            if (master_selector_ == SPMasterSelector::LeastCache) {
                score = std::make_tuple(
                    -state.free_blocks[sp_idx], state.master_counts[sp_idx], state.tokens[sp_idx], sp_idx);
            }
            else {
                score = std::make_tuple(
                    state.master_counts[sp_idx], state.tokens[sp_idx], -state.free_blocks[sp_idx], sp_idx);
            }

            if (best_idx == -1 || score < best_score) {
                best_idx   = sp_idx;
                best_score = score;
            }
        }
        return best_idx;
    };

    auto select_instances = [&](const PlanningState& state, int master_sp_idx, int target_sp) {
        std::vector<int> participants;
        participants.push_back(master_sp_idx);

        std::vector<int> others;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (sp_idx != master_sp_idx) {
                others.push_back(sp_idx);
            }
        }

        std::sort(others.begin(), others.end(), [&](int lhs, int rhs) {
            if (state.tokens[lhs] != state.tokens[rhs]) {
                return state.tokens[lhs] < state.tokens[rhs];
            }
            if (state.free_blocks[lhs] != state.free_blocks[rhs]) {
                return state.free_blocks[lhs] > state.free_blocks[rhs];
            }
            return lhs < rhs;
        });

        for (int sp_idx : others) {
            if ((int)participants.size() >= target_sp) {
                break;
            }
            participants.push_back(sp_idx);
        }
        return participants;
    };

    auto waterfill = [&](const Sequence& seq, const PlanningState& state, const std::vector<int>& participants) {
        std::vector<int> split(attention_sp_, 0);
        int              k = static_cast<int>(participants.size());
        if (k == 0 || seq.num_tokens < k) {
            return split;
        }

        std::vector<long long> loads;
        loads.reserve(k);
        for (int sp_idx : participants) {
            loads.push_back(state.tokens[sp_idx]);
        }

        long long low        = *std::min_element(loads.begin(), loads.end()) + 1;
        long long high       = *std::max_element(loads.begin(), loads.end()) + seq.num_tokens;
        long long best_level = low;

        auto required_tokens = [&](long long level) {
            long long required = 0;
            for (long long load : loads) {
                required += std::max(1LL, level - load);
            }
            return required;
        };

        while (low <= high) {
            long long mid = (low + high) / 2;
            if (required_tokens(mid) <= seq.num_tokens) {
                best_level = mid;
                low        = mid + 1;
            }
            else {
                high = mid - 1;
            }
        }

        std::vector<int> allocation(k, 1);
        int              used_tokens = 0;
        for (int i = 0; i < k; ++i) {
            allocation[i] = (int)std::max(1LL, best_level - loads[i]);
            used_tokens += allocation[i];
        }

        int              remaining = seq.num_tokens - used_tokens;
        std::vector<int> order(k);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
            long long lhs_final = loads[lhs] + allocation[lhs];
            long long rhs_final = loads[rhs] + allocation[rhs];
            if (lhs_final != rhs_final) {
                return lhs_final < rhs_final;
            }
            return participants[lhs] < participants[rhs];
        });

        int idx = 0;
        while (remaining > 0) {
            allocation[order[idx]]++;
            remaining--;
            idx = (idx + 1) % k;
        }

        for (int i = 0; i < k; ++i) {
            split[participants[i]] = allocation[i];
        }
        return split;
    };

    auto placement_feasible = [&](const Sequence& seq, const PlannedPlacement& placement, const PlanningState& state) {
        int master_sp_idx   = placement.master_sp_idx;
        int uncached_tokens = seq.num_tokens - seq.num_cached_tokens;

        if (master_sp_idx < 0 || master_sp_idx >= attention_sp_) {
            return false;
        }
        if (state.master_counts[master_sp_idx] + 1 > max_num_seqs_) {
            return false;
        }
        if (state.batch_tokens[master_sp_idx] + uncached_tokens >= max_num_batched_tokens_) {
            return false;
        }

        int total_dispatched = 0;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int dispatched = placement.num_dispatched_tokens[sp_idx];
            total_dispatched += dispatched;
            if (dispatched <= 0) {
                continue;
            }

            if (fixed_sp_size_ == 0 && sp_idx != master_sp_idx && state.recv_counts[sp_idx] + 1 > max_num_recv_seqs_) {
                return false;
            }

            int prefill_blocks_needed     = ceil_div(dispatched, kvcache_block_size_);
            int projected_master_count    = state.master_counts[sp_idx] + (sp_idx == master_sp_idx ? 1 : 0);
            int reservation_blocks_needed = (int)std::ceil(projected_master_count * reserved_blocks_per_req_);
            if (state.free_blocks[sp_idx] < prefill_blocks_needed + reservation_blocks_needed) {
                return false;
            }
        }

        return total_dispatched == seq.num_tokens;
    };

    auto apply_placement = [&](const Sequence& seq, const PlannedPlacement& placement, PlanningState& state) {
        int master_sp_idx = placement.master_sp_idx;
        state.master_counts[master_sp_idx]++;
        state.batch_tokens[master_sp_idx] += (seq.num_tokens - seq.num_cached_tokens);
        if (master_selector_ == SPMasterSelector::RoundRobin) {
            state.rr_cursor = (master_sp_idx + 1) % attention_sp_;
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int dispatched = placement.num_dispatched_tokens[sp_idx];
            if (dispatched <= 0) {
                continue;
            }

            state.tokens[sp_idx] += dispatched;
            state.free_blocks[sp_idx] -= ceil_div(dispatched, kvcache_block_size_);
            if (sp_idx != master_sp_idx) {
                state.recv_counts[sp_idx]++;
            }
        }
        add_communication(state, master_sp_idx, placement.num_dispatched_tokens);
    };

    auto build_placement = [&](const Sequence&      seq,
                               const PlanningState& state_template,
                               int                  target_sp,
                               std::optional<int>   fixed_master = std::nullopt) -> std::optional<PlannedPlacement> {
        if (fixed_sp_size_ > 0) {
            target_sp = effective_target_sp_size(target_sp, seq.num_tokens);
        }
        PlanningState state         = state_template;
        int           master_sp_idx = fixed_master.has_value() ? *fixed_master : choose_master(state, seq);
        if (master_sp_idx < 0) {
            return std::nullopt;
        }

        auto participants = select_instances(state_template, master_sp_idx, target_sp);
        if ((int)participants.size() != target_sp) {
            return std::nullopt;
        }

        PlannedPlacement placement;
        placement.master_sp_idx         = master_sp_idx;
        placement.num_dispatched_tokens = waterfill(seq, state_template, participants);
        if (!placement_feasible(seq, placement, state_template)) {
            return std::nullopt;
        }
        return placement;
    };

    PlanningState running_state =
        cached_running_state_.has_value() ? *cached_running_state_ : build_running_state_snapshot();
    PlanningState                 baseline_state = running_state;
    std::vector<PlannedPlacement> baseline_placements;
    baseline_placements.reserve(pending_seqs.size());

    for (const auto& seq : pending_seqs) {
        std::optional<PlannedPlacement> chosen;
        for (int target_sp : allowed_sp_sizes) {
            chosen = build_placement(*seq, baseline_state, target_sp);
            if (chosen.has_value()) {
                break;
            }
        }
        if (!chosen.has_value()) {
            return std::nullopt;
        }
        baseline_placements.push_back(*chosen);
        apply_placement(*seq, *chosen, baseline_state);
    }

    std::set<int> overflow_request_ids;
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        int overflow = baseline_state.tokens[sp_idx] - running_state.group_max;
        if (overflow <= 0) {
            continue;
        }

        std::vector<std::pair<int, int>> contributions;
        for (size_t idx = 0; idx < baseline_placements.size(); ++idx) {
            int dispatched = baseline_placements[idx].num_dispatched_tokens[sp_idx];
            if (dispatched > 0) {
                contributions.push_back({-dispatched, (int)idx});
            }
        }
        std::sort(contributions.begin(), contributions.end());

        int covered = 0;
        for (const auto& [neg_dispatched, idx] : contributions) {
            overflow_request_ids.insert(idx);
            covered += -neg_dispatched;
            if (covered >= overflow) {
                break;
            }
        }
    }

    std::vector<PlannedPlacement> best_placements         = baseline_placements;
    LatencyBreakdown              best_latency            = estimate_latency(baseline_state);
    int                           best_overflow           = total_overflow(baseline_state, running_state.group_max);
    int                           best_extra_participants = count_extra_participants(baseline_placements);

    for (int request_idx : overflow_request_ids) {
        int current_sp = count_active_ranks(baseline_placements[request_idx]);

        for (int target_sp : allowed_sp_sizes) {
            if (target_sp <= current_sp) {
                continue;
            }

            PlanningState                 candidate_state      = running_state;
            std::vector<PlannedPlacement> candidate_placements = baseline_placements;

            for (size_t idx = 0; idx < baseline_placements.size(); ++idx) {
                if ((int)idx == request_idx) {
                    continue;
                }
                apply_placement(*pending_seqs[idx], candidate_placements[idx], candidate_state);
            }

            auto repaired = build_placement(
                *pending_seqs[request_idx], candidate_state, target_sp, baseline_placements[request_idx].master_sp_idx);
            if (!repaired.has_value()) {
                continue;
            }

            candidate_placements[request_idx] = *repaired;
            apply_placement(*pending_seqs[request_idx], *repaired, candidate_state);

            LatencyBreakdown candidate_latency            = estimate_latency(candidate_state);
            int              candidate_overflow           = total_overflow(candidate_state, running_state.group_max);
            int              candidate_extra_participants = count_extra_participants(candidate_placements);

            if (better_candidate(candidate_latency,
                                 candidate_overflow,
                                 candidate_extra_participants,
                                 best_latency,
                                 best_overflow,
                                 best_extra_participants)) {
                best_latency            = candidate_latency;
                best_overflow           = candidate_overflow;
                best_extra_participants = candidate_extra_participants;
                best_placements         = std::move(candidate_placements);
            }
        }
    }

    DecodeBatchPlan plan;
    plan.placements         = std::move(best_placements);
    plan.latency            = best_latency;
    plan.max_tokens         = static_cast<int>(best_latency.max_tokens);
    plan.total_overflow     = best_overflow;
    plan.extra_participants = best_extra_participants;
    return plan;
}

void SPStateManager::apply_planned_placement(Sequence& seq, const PlannedPlacement& placement)
{
    auto& block_ctx                 = seq.block_ctx(BlockContextSlot::ACTIVE);
    block_ctx.master_sp_idx_        = placement.master_sp_idx;
    block_ctx.num_dispatched_tokens = placement.num_dispatched_tokens;
    block_ctx.block_location.clear();
    for (auto& table : block_ctx.sp_block_table) {
        table.clear();
    }

    if (master_selector_ == SPMasterSelector::RoundRobin) {
        sp_rr_counter_ = (placement.master_sp_idx + 1) % attention_sp_;
    }
}

bool SPStateManager::can_allocate(Sequence&                           seq,
                                  const std::unordered_map<int, int>& num_seqs,
                                  const std::unordered_map<int, int>& num_batched_tokens)
{
    if (attention_sp_ > 1) {
        // ==========================================
        //  Debug Strategy (sp_debug mode)
        // ==========================================
        if (sp_debug_) {
            auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);

            int num_tokens = seq.num_tokens;
            // Segment equals Block Size in debug mode
            int debug_segment_size = kvcache_block_size_;
            int num_segments       = (num_tokens + debug_segment_size - 1) / debug_segment_size;

            // SP Size is determined by number of segments
            int target_num_ranks = std::min(num_segments, attention_sp_);
            if (target_num_ranks == 0)
                target_num_ranks = 1;

            // Select Master Rank using Round Robin
            SPMasterSelector original_selector = master_selector_;
            master_selector_                   = SPMasterSelector::RoundRobin;
            int master_rank                    = select_master_rank();
            master_selector_                   = original_selector;  // Restore original selector

            // Check Master Rank capacity
            if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) {
                return false;
            }

            auto it_tokens              = num_batched_tokens.find(master_rank);
            int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
            if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
                return false;
            }

            // Prepare rank selection (excluding master for now)
            std::vector<std::pair<int, int>> rank_free_count;
            for (const auto& [rank, bm] : block_manager) {
                if (rank != master_rank) {
                    rank_free_count.push_back({rank, bm->num_free_blocks()});
                }
            }
            std::sort(rank_free_count.begin(),
                      rank_free_count.end(),
                      [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

            // Select participating ranks (non-master ranks first)
            std::vector<int> participating_ranks;
            int              non_master_ranks_needed = std::min((int)rank_free_count.size(), target_num_ranks - 1);
            for (int i = 0; i < non_master_ranks_needed; ++i) {
                participating_ranks.push_back(rank_free_count[i].first);
            }
            participating_ranks.push_back(master_rank);  // Master is always included

            // Initialize token dispatch
            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);
            block_ctx.master_sp_idx_ = master_rank;

            // Allocate tokens: non-master ranks get full segments, master gets the remainder (tail segment)
            int remaining_tokens                = num_tokens;
            int segments_assigned_to_non_master = std::min(non_master_ranks_needed, num_segments - 1);

            // Assign full segments to non-master ranks
            for (int i = 0; i < segments_assigned_to_non_master; ++i) {
                int sp_idx                              = participating_ranks[i];
                block_ctx.num_dispatched_tokens[sp_idx] = debug_segment_size;
                remaining_tokens -= debug_segment_size;
            }

            // Assign remaining tokens (tail segment) to master rank
            block_ctx.num_dispatched_tokens[master_rank] = remaining_tokens;

            // Reservation Check
            std::vector<int> master_req_counts(attention_sp_, 0);
            for (const auto& running_seq : running) {
                int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                if (m_idx >= 0 && m_idx < attention_sp_)
                    master_req_counts[m_idx]++;
            }
            for (const auto& [m_idx, count] : num_seqs) {
                if (m_idx >= 0 && m_idx < attention_sp_)
                    master_req_counts[m_idx] += count;
            }
            master_req_counts[master_rank]++;

            // Memory check
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (block_ctx.num_dispatched_tokens[sp_idx] > 0 || sp_idx == master_rank) {

                    if (fixed_sp_size_ == 0 && sp_idx != master_rank && block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                            return false;
                        }
                    }

                    int free_blocks           = block_manager[sp_idx]->num_free_blocks();
                    int prefill_tokens        = block_ctx.num_dispatched_tokens[sp_idx];
                    int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

                    double needed_float              = master_req_counts[sp_idx] * reserved_blocks_per_req_;
                    int    reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

                    if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) {
                        return false;
                    }

                    if (!block_manager[sp_idx]->can_allocate(seq)) {
                        return false;
                    }
                }
            }

            return true;
        }

        // ==========================================
        //  Optimized Strategy (Dynamic SP Size)
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);

        int num_tokens            = seq.num_tokens;
        int num_segments          = (num_tokens + segment_size_ - 1) / segment_size_;
        int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int initial_num_ranks     = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        if (num_segments_per_rank == 0)
            num_segments_per_rank = 1;
        if (initial_num_ranks == 0)
            initial_num_ranks = 1;

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_)
            return false;

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_)
            return false;

        // Rank selection preparation
        std::vector<std::pair<int, int>> rank_free_count;
        for (const auto& [rank, bm] : block_manager) {
            if (rank != master_rank) {
                rank_free_count.push_back({rank, bm->num_free_blocks()});
            }
        }
        std::sort(rank_free_count.begin(),
                  rank_free_count.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

        int  start_ranks                      = initial_num_ranks;
        int  end_ranks                        = enable_dynamic_sp_size_ ? attention_sp_ : initial_num_ranks;
        bool recompute_segments_for_forced_sp = false;
        if (fixed_sp_size_ > 0) {
            const int forced_num_ranks = effective_target_sp_size(fixed_sp_size_, seq.num_tokens);
            start_ranks                = forced_num_ranks;
            end_ranks                  = forced_num_ranks;
        }
        else if (dynamic_sp_size_strategy_ == DynamicSPSizeStrategy::Bucket) {
            const int forced_num_ranks =
                std::max(1, std::min(attention_sp_, select_bucket_sp_size(seq.num_tokens).value_or(initial_num_ranks)));
            start_ranks                      = forced_num_ranks;
            end_ranks                        = forced_num_ranks;
            recompute_segments_for_forced_sp = true;
        }
        else if (dynamic_sp_size_strategy_ == DynamicSPSizeStrategy::LongShortSP8) {
            const bool is_long_request       = seq.num_prompt_tokens > long_request_sp_threshold_;
            const int  forced_num_ranks      = is_long_request ? long_request_sp_size_ : 1;
            start_ranks                      = forced_num_ranks;
            end_ranks                        = forced_num_ranks;
            recompute_segments_for_forced_sp = true;
        }

        for (int target_num_ranks = start_ranks; target_num_ranks <= end_ranks; ++target_num_ranks) {
            int target_num_segments_per_rank = num_segments_per_rank;
            if (recompute_segments_for_forced_sp) {
                target_num_segments_per_rank = (num_segments + target_num_ranks - 1) / target_num_ranks;
                if (target_num_segments_per_rank == 0) {
                    target_num_segments_per_rank = 1;
                }
            }

            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);
            block_ctx.master_sp_idx_ = master_rank;

            // Select participating ranks
            std::vector<int> top_most_free_ranks;
            int              ranks_to_pick = std::min((int)rank_free_count.size(), target_num_ranks - 1);
            for (int i = 0; i < ranks_to_pick; ++i) {
                top_most_free_ranks.push_back(rank_free_count[i].first);
            }
            top_most_free_ranks.push_back(master_rank);

            // =================================================================
            // [New Feature] Non-Uniform Split (Water-filling / Valley-filling)
            // =================================================================
            if (enable_non_uniform_split_ && fixed_sp_size_ == 0) {
                // 1. Collect free blocks info for all participating ranks (including master)
                std::vector<std::pair<int, int>> sorted_ranks;  // {sp_idx, free_blocks}
                for (int sp_idx : top_most_free_ranks) {
                    sorted_ranks.push_back({sp_idx, block_manager[sp_idx]->num_free_blocks()});
                }

                // 2. Sort participating ranks by free blocks descending (richest first)
                std::sort(
                    sorted_ranks.begin(),
                    sorted_ranks.end(),
                    [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

                long long total_tokens_needed      = seq.num_tokens;
                long long final_target_free_tokens = 0;
                int       k                        = 0;  // Number of ranks contributing to "water-filling"

                // 3. Find the optimal "water level" (target free tokens)
                // We greedily check if the top k ranks can absorb the load such that
                // their remaining capacity is balanced.
                for (k = 1; k <= (int)sorted_ranks.size(); ++k) {
                    long long sum_free_tokens = 0;
                    for (int i = 0; i < k; ++i) {
                        sum_free_tokens += (long long)sorted_ranks[i].second * kvcache_block_size_;
                    }

                    // If we use top k ranks, what would be the equalized remaining capacity?
                    long long remaining_after_alloc = sum_free_tokens - total_tokens_needed;
                    long long target_free           = remaining_after_alloc / k;

                    // If we are at the last rank, or if the calculated target level is
                    // higher than the next rank's capacity (meaning next rank doesn't need to help),
                    // then we found our split point.
                    if (k == (int)sorted_ranks.size()) {
                        final_target_free_tokens = target_free;
                        break;
                    }
                    else {
                        long long next_rank_free = (long long)sorted_ranks[k].second * kvcache_block_size_;
                        if (target_free >= next_rank_free) {
                            final_target_free_tokens = target_free;
                            break;
                        }
                    }
                }

                // 4. Assign tokens based on the target level
                long long allocated_sum = 0;
                for (int i = 0; i < (int)sorted_ranks.size(); ++i) {
                    int       sp_idx       = sorted_ranks[i].first;
                    long long current_free = (long long)sorted_ranks[i].second * kvcache_block_size_;

                    // Alloc = Current - Target
                    long long alloc = current_free - final_target_free_tokens;

                    if (alloc < 0)
                        alloc = 0;
                    if (alloc > current_free)
                        alloc = current_free;  // Safety cap

                    block_ctx.num_dispatched_tokens[sp_idx] = (int)alloc;
                    allocated_sum += alloc;
                }

                // 5. Handle integer division remainders
                long long remainder = total_tokens_needed - allocated_sum;
                int       idx       = 0;

                // If we allocated too few (remainder > 0), distribute to the richest ranks
                while (remainder > 0) {
                    block_ctx.num_dispatched_tokens[sorted_ranks[idx].first]++;
                    remainder--;
                    idx = (idx + 1) % k;
                }

                // If we allocated too many (remainder < 0), take back from richest ranks
                // (This can happen if target calculation slightly overshoots due to integer math)
                while (remainder < 0) {
                    if (block_ctx.num_dispatched_tokens[sorted_ranks[idx].first] > 0) {
                        block_ctx.num_dispatched_tokens[sorted_ranks[idx].first]--;
                        remainder++;
                    }
                    idx = (idx + 1) % k;
                }
            }
            else {
                // =================================================================
                // Standard Feature: Uniform Split
                // =================================================================
                if (fixed_sp_size_ > 0) {
                    int participating_ranks = static_cast<int>(top_most_free_ranks.size());
                    if (participating_ranks <= 0) {
                        return false;
                    }
                    int base_tokens  = seq.num_tokens / participating_ranks;
                    int extra_tokens = seq.num_tokens % participating_ranks;
                    for (int i = 0; i < participating_ranks; ++i) {
                        int sp_idx                              = top_most_free_ranks[i];
                        block_ctx.num_dispatched_tokens[sp_idx] = base_tokens + (i < extra_tokens ? 1 : 0);
                    }
                }
                else {
                    int total_token_unalloc = seq.num_tokens;
                    for (int sp_idx : top_most_free_ranks) {
                        int tokens_to_dispatch =
                            std::min(total_token_unalloc, target_num_segments_per_rank * segment_size_);
                        block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
                        total_token_unalloc -= tokens_to_dispatch;
                    }
                }
            }

            // Reservation Check
            std::vector<int> master_req_counts(attention_sp_, 0);
            for (const auto& running_seq : running) {
                int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                if (m_idx >= 0 && m_idx < attention_sp_)
                    master_req_counts[m_idx]++;
            }
            for (const auto& [m_idx, count] : num_seqs) {
                if (m_idx >= 0 && m_idx < attention_sp_)
                    master_req_counts[m_idx] += count;
            }
            master_req_counts[master_rank]++;

            bool memory_check_passed = true;
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (block_ctx.num_dispatched_tokens[sp_idx] > 0 || sp_idx == master_rank) {

                    if (fixed_sp_size_ == 0 && sp_idx != master_rank && block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                            memory_check_passed = false;
                            break;
                        }
                    }

                    int free_blocks           = block_manager[sp_idx]->num_free_blocks();
                    int prefill_tokens        = block_ctx.num_dispatched_tokens[sp_idx];
                    int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

                    double needed_float              = master_req_counts[sp_idx] * reserved_blocks_per_req_;
                    int    reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

                    if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) {
                        memory_check_passed = false;
                        break;
                    }

                    if (!block_manager[sp_idx]->can_allocate(seq)) {
                        memory_check_passed = false;
                        break;
                    }
                }
            }

            if (memory_check_passed) {
                return true;
            }
        }
        return false;
    }
    else {
        // ==========================================
        //  Naive Strategy (SP=1)
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

        int num_tokens            = seq.num_tokens;
        int num_segments          = (num_tokens + segment_size_ - 1) / segment_size_;
        int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int num_ranks             = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        if (num_segments_per_rank == 0)
            num_segments_per_rank = 1;
        if (num_ranks == 0)
            num_ranks = 1;

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) {
            return false;
        }

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) {
            return false;
        }

        std::vector<std::pair<int, int>> rank_free_count;
        for (const auto& [rank, bm] : block_manager) {
            if (rank != master_rank) {
                rank_free_count.push_back({rank, bm->num_free_blocks()});
            }
        }

        std::sort(rank_free_count.begin(),
                  rank_free_count.end(),
                  [](const std::pair<int, int>& a, const std::pair<int, int>& b) { return a.second > b.second; });

        std::vector<int> top_most_free_ranks;
        int              ranks_to_pick = std::min((int)rank_free_count.size(), num_ranks - 1);
        for (int i = 0; i < ranks_to_pick; ++i) {
            top_most_free_ranks.push_back(rank_free_count[i].first);
        }
        top_most_free_ranks.push_back(master_rank);

        block_ctx.master_sp_idx_ = master_rank;
        int total_token_unalloc  = seq.num_tokens;

        for (int sp_idx : top_most_free_ranks) {
            int tokens_to_dispatch = std::min(total_token_unalloc, num_segments_per_rank * segment_size_);
            block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
            total_token_unalloc -= tokens_to_dispatch;
        }

        std::vector<int> master_req_counts(attention_sp_, 0);

        for (const auto& running_seq : running) {
            int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (m_idx >= 0 && m_idx < attention_sp_) {
                master_req_counts[m_idx]++;
            }
        }

        for (const auto& [m_idx, count] : num_seqs) {
            if (m_idx >= 0 && m_idx < attention_sp_) {
                master_req_counts[m_idx] += count;
            }
        }

        master_req_counts[master_rank]++;

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int free_blocks = block_manager[sp_idx]->num_free_blocks();

            int prefill_tokens        = block_ctx.num_dispatched_tokens[sp_idx];
            int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

            double needed_float              = master_req_counts[sp_idx] * reserved_blocks_per_req_;
            int    reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

            if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) {
                return false;
            }
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (!block_manager[sp_idx]->can_allocate(seq)) {
                return false;
            }
        }

        return true;
    }
}

void SPStateManager::allocate(Sequence& seq)
{
    cached_running_state_.reset();
    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        master_seq_counts_[master_sp_idx]++;
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        // [修改] 更新 Recv 计数
        // 只有确实分到了 token 且不是 Master 的才算 Receiver
        if (block_ctx.num_dispatched_tokens[sp_idx] > 0) {
            if (sp_idx != master_sp_idx) {
                num_recv_seqs_per_sp_[sp_idx]++;
            }
        }

        if (sp_idx != master_sp_idx) {
            block_manager[sp_idx]->allocate(seq);
        }
    }
    block_manager[master_sp_idx]->allocate(seq);

    num_running_seqs_++;
    num_running_tokens_ += seq.num_tokens;
}

void SPStateManager::allocate_ls_initial(Sequence& seq)
{
    cached_running_state_.reset();
    auto& ctx    = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master = ctx.master_sp_idx_;
    if (master < 0 || master >= attention_sp_) {
        throw std::runtime_error("LS initial master is out of range");
    }
    master_seq_counts_[master]++;
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (ctx.num_dispatched_tokens[sp_idx] > 0 && sp_idx != master) {
            num_recv_seqs_per_sp_[sp_idx]++;
        }
        block_manager[sp_idx]->allocate_uncached(seq);
    }
    num_running_seqs_++;
    num_running_tokens_ += seq.num_tokens;
}

void SPStateManager::allocate_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch,
                                               int failure_after_allocations_for_test)
{
    if (batch.empty()) {
        throw std::runtime_error("LS initial batch must not be empty");
    }

    std::unordered_set<uint64_t> sequence_ids;
    std::vector<int>             needed_blocks(attention_sp_, 0);
    std::vector<int>             master_delta(attention_sp_, 0);
    std::vector<int>             recv_delta(attention_sp_, 0);
    int                          running_token_delta = 0;

    // Validate the entire transaction and aggregate its capacity demand before
    // mutating any block manager or role counter.
    for (const auto& seq : batch) {
        if (!seq || !sequence_ids.insert(seq->seq_id).second) {
            throw std::runtime_error("LS initial batch contains a null or duplicate sequence");
        }
        const auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
        if (ctx.master_sp_idx_ < 0 || ctx.master_sp_idx_ >= attention_sp_) {
            throw std::runtime_error("LS initial master is out of range");
        }
        if (static_cast<int>(ctx.num_dispatched_tokens.size()) != attention_sp_
            || static_cast<int>(ctx.sp_block_table.size()) != attention_sp_ || !ctx.block_location.empty()) {
            throw std::runtime_error("LS initial sequence has an invalid active block context");
        }

        int dispatched = 0;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int tokens = ctx.num_dispatched_tokens[sp_idx];
            if (tokens < 0 || !ctx.sp_block_table[sp_idx].empty()) {
                throw std::runtime_error("LS initial sequence has a dirty or negative placement");
            }
            dispatched += tokens;
            needed_blocks[sp_idx] += (tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
            if (tokens > 0 && sp_idx != ctx.master_sp_idx_) {
                recv_delta[sp_idx]++;
            }
        }
        if (dispatched != seq->num_tokens) {
            throw std::runtime_error("LS initial placement does not cover the complete sequence");
        }
        master_delta[ctx.master_sp_idx_]++;
        running_token_delta += seq->num_tokens;
    }
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (block_manager.at(sp_idx)->num_free_blocks() < needed_blocks[sp_idx]) {
            throw std::runtime_error("LS initial batch lost its prevalidated block capacity");
        }
    }

    int allocated_sequences = 0;
    try {
        if (failure_after_allocations_for_test == 0) {
            throw std::runtime_error("injected LS initial batch allocation failure");
        }
        for (const auto& seq : batch) {
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                block_manager.at(sp_idx)->allocate_uncached(*seq);
            }
            allocated_sequences++;
            if (failure_after_allocations_for_test >= 0 && allocated_sequences == failure_after_allocations_for_test) {
                throw std::runtime_error("injected LS initial batch allocation failure");
            }
        }
    }
    catch (...) {
        // Scan every table, including the currently failing rank. This also
        // covers an exception thrown after a BlockManager partially mutated a
        // table but before the sequence-level allocation completed.
        for (const auto& seq : batch) {
            if (!seq) {
                continue;
            }
            auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (!ctx.sp_block_table[sp_idx].empty()) {
                    block_manager.at(sp_idx)->deallocate(*seq, BlockContextSlot::ACTIVE);
                }
            }
            ctx.block_location.clear();
        }
        throw;
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        master_seq_counts_[sp_idx] += master_delta[sp_idx];
        num_recv_seqs_per_sp_[sp_idx] += recv_delta[sp_idx];
    }
    num_running_seqs_ += static_cast<int>(batch.size());
    num_running_tokens_ += running_token_delta;
    cached_running_state_.reset();
}

void SPStateManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    cached_running_state_.reset();
    auto& block_ctx     = seq.block_ctx(slot);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        block_manager[sp_idx]->deallocate(seq, slot);
    }

    // Commit counters only after every rank released successfully. Together
    // with BlockManager's retry-safe table updates, an exceptional deallocate
    // never applies the logical counter delta twice.
    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_ && master_seq_counts_[master_sp_idx] > 0) {
        master_seq_counts_[master_sp_idx]--;
    }
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (block_ctx.num_dispatched_tokens[sp_idx] > 0 && sp_idx != master_sp_idx) {
            num_recv_seqs_per_sp_[sp_idx]--;
        }
    }

    for (auto& table : block_ctx.sp_block_table) {
        table.clear();
    }
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);
    block_ctx.pending_token_present_   = false;
    block_ctx.pending_token_target_sp_ = -1;

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens;
}

}  // namespace nanodeploy
