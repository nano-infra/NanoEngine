#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <set>

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
    const std::string trimmed = trim_copy(text);
    if (trimmed.empty()) {
        return intervals;
    }

    std::stringstream ss(trimmed);
    std::string item;
    int prev_high = std::numeric_limits<int>::min();
    while (std::getline(ss, item, ';')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        const auto colon = item.find(':');
        const auto dash = item.find('-', colon == std::string::npos ? 0 : colon + 1);
        if (colon == std::string::npos || dash == std::string::npos) {
            throw std::runtime_error("Invalid dynamic_sp_bucket_policy item: " + item);
        }
        SPBucketInterval interval;
        interval.sp_size = std::stoi(trim_copy(item.substr(0, colon)));
        interval.seq_len_low = std::stoi(trim_copy(item.substr(colon + 1, dash - colon - 1)));
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
        case DynamicSPSizeStrategy::LongShort:
            return "long_short";
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
                               int                fixed_sp_segments) :
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
    long_request_sp_size_(dynamic_sp_long_request_size),
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
    fixed_sp_segments_(fixed_sp_segments)
{
    // Initialize Strategy
    if (sp_master_selector == "LeastBatch") {
        master_selector_ = SPMasterSelector::LeastBatch;
    } else if (sp_master_selector == "LeastCache") {
        master_selector_ = SPMasterSelector::LeastCache;
    } else {
        master_selector_ = SPMasterSelector::RoundRobin;
    }

    if (dynamic_sp_size_strategy == "legacy") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Legacy;
    } else if (
        dynamic_sp_size_strategy == "long_short"
        || dynamic_sp_size_strategy == "long_short_sp8") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::LongShort;
    } else if (dynamic_sp_size_strategy == "bucket") {
        dynamic_sp_size_strategy_ = DynamicSPSizeStrategy::Bucket;
    } else {
        throw std::runtime_error(
            "Unsupported dynamic_sp_size_strategy: " + dynamic_sp_size_strategy);
    }
    dynamic_sp_bucket_policy_ = parse_bucket_policy(dynamic_sp_bucket_policy, attention_sp_);

    // Initialize Running Load Counter
    master_seq_counts_.assign(attention_sp_, 0);

    for (int i = 0; i < attention_sp; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }

    initialize_dummy_seqs();

    std::cerr << "[SPStateManager] Initialized with attention_sp=" << attention_sp_ 
              << ", kvcache_block_size=" << kvcache_block_size_
              << ", reserved_blocks_per_req=" << reserved_blocks_per_req_ 
              << ", segment_size=" << segment_size_
              << ", fixed_sp_segments=" << fixed_sp_segments_
              << ", dynamic_sp_size_strategy=" << dynamic_sp_size_strategy_name(dynamic_sp_size_strategy_)
              << ", dynamic_sp_long_request_threshold=" << long_request_sp_threshold_
              << ", dynamic_sp_long_request_size=" << long_request_sp_size_
              << ", enable_dynamic_sp_bucket_policy=" << enable_dynamic_sp_bucket_policy_
              << std::endl;

    if (attention_sp_ <= 0) {
        throw std::runtime_error("attention_sp must be positive to prevent division by zero");
    }
    if (kvcache_block_size_ <= 0) {
        throw std::runtime_error("kvcache_block_size must be positive to prevent division by zero");
    }
    if (long_request_sp_size_ < 1 || long_request_sp_size_ > attention_sp_) {
        throw std::runtime_error("dynamic_sp_long_request_size must be in [1, attention_sp]");
    }
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
        int idx = sp_rr_counter_;
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
    return 0; // Fallback
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

void SPStateManager::add_communication(PlanningState&            state,
                                       int                       master_sp_idx,
                                       const std::vector<int>&   dispatched_tokens) const
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
    state.recv_counts = num_recv_seqs_per_sp_;
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

std::optional<SPStateManager::DecodeBatchPlan> SPStateManager::plan_decode_batch(
    const std::vector<std::shared_ptr<Sequence>>& pending_seqs) const
{
    auto ceil_div = [](int x, int y) -> int { return (x + y - 1) / y; };

    auto allowed_sp_sizes = [&]() {
        std::vector<int> sizes;
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
        breakdown.max_tokens = static_cast<double>(*std::max_element(state.tokens.begin(), state.tokens.end()));
        breakdown.max_q_bytes = max_send_or_recv(state.send_q, state.recv_q);
        breakdown.max_res_bytes = max_send_or_recv(state.send_res, state.recv_res);
        breakdown.max_lse_bytes = max_send_or_recv(state.send_lse, state.recv_lse);
        breakdown.attention = cost_model_.attention.predict(breakdown.max_tokens);
        breakdown.q = breakdown.max_q_bytes > 0.0 ? cost_model_.q.predict(breakdown.max_q_bytes) : 0.0;
        breakdown.res =
            breakdown.max_res_bytes > 0.0 ? cost_model_.res.predict(breakdown.max_res_bytes) : 0.0;
        breakdown.lse =
            breakdown.max_lse_bytes > 0.0 ? cost_model_.lse.predict(breakdown.max_lse_bytes) : 0.0;
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

        int best_idx = -1;
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
                score = std::make_tuple(-state.free_blocks[sp_idx], state.master_counts[sp_idx], state.tokens[sp_idx], sp_idx);
            } else {
                score = std::make_tuple(state.master_counts[sp_idx], state.tokens[sp_idx], -state.free_blocks[sp_idx], sp_idx);
            }

            if (best_idx == -1 || score < best_score) {
                best_idx = sp_idx;
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

    auto waterfill = [&](const Sequence& seq,
                         const PlanningState& state,
                         const std::vector<int>& participants) {
        std::vector<int> split(attention_sp_, 0);
        int k = static_cast<int>(participants.size());
        if (k == 0 || seq.num_tokens < k) {
            return split;
        }

        std::vector<long long> loads;
        loads.reserve(k);
        for (int sp_idx : participants) {
            loads.push_back(state.tokens[sp_idx]);
        }

        long long low = *std::min_element(loads.begin(), loads.end()) + 1;
        long long high = *std::max_element(loads.begin(), loads.end()) + seq.num_tokens;
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
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        std::vector<int> allocation(k, 1);
        int used_tokens = 0;
        for (int i = 0; i < k; ++i) {
            allocation[i] = (int)std::max(1LL, best_level - loads[i]);
            used_tokens += allocation[i];
        }

        int remaining = seq.num_tokens - used_tokens;
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

    auto placement_feasible = [&](const Sequence& seq,
                                  const PlannedPlacement& placement,
                                  const PlanningState& state) {
        int master_sp_idx = placement.master_sp_idx;
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

            if (sp_idx != master_sp_idx && state.recv_counts[sp_idx] + 1 > max_num_recv_seqs_) {
                return false;
            }

            int prefill_blocks_needed = ceil_div(dispatched, kvcache_block_size_);
            int projected_master_count = state.master_counts[sp_idx] + (sp_idx == master_sp_idx ? 1 : 0);
            int reservation_blocks_needed = (int)std::ceil(projected_master_count * reserved_blocks_per_req_);
            if (state.free_blocks[sp_idx] < prefill_blocks_needed + reservation_blocks_needed) {
                return false;
            }
        }

        return total_dispatched == seq.num_tokens;
    };

    auto apply_placement = [&](const Sequence& seq,
                               const PlannedPlacement& placement,
                               PlanningState& state) {
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

    auto build_placement = [&](const Sequence& seq,
                               const PlanningState& state_template,
                               int target_sp,
                               std::optional<int> fixed_master = std::nullopt)
        -> std::optional<PlannedPlacement> {
        PlanningState state = state_template;
        int master_sp_idx = fixed_master.has_value() ? *fixed_master : choose_master(state, seq);
        if (master_sp_idx < 0) {
            return std::nullopt;
        }

        auto participants = select_instances(state_template, master_sp_idx, target_sp);
        if ((int)participants.size() != target_sp) {
            return std::nullopt;
        }

        PlannedPlacement placement;
        placement.master_sp_idx = master_sp_idx;
        placement.num_dispatched_tokens = waterfill(seq, state_template, participants);
        if (!placement_feasible(seq, placement, state_template)) {
            return std::nullopt;
        }
        return placement;
    };

    PlanningState running_state =
        cached_running_state_.has_value() ? *cached_running_state_ : build_running_state_snapshot();
    PlanningState baseline_state = running_state;
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

    std::vector<PlannedPlacement> best_placements = baseline_placements;
    LatencyBreakdown best_latency = estimate_latency(baseline_state);
    int best_overflow = total_overflow(baseline_state, running_state.group_max);
    int best_extra_participants = count_extra_participants(baseline_placements);

    for (int request_idx : overflow_request_ids) {
        int current_sp = count_active_ranks(baseline_placements[request_idx]);

        for (int target_sp : allowed_sp_sizes) {
            if (target_sp <= current_sp) {
                continue;
            }

            PlanningState candidate_state = running_state;
            std::vector<PlannedPlacement> candidate_placements = baseline_placements;

            for (size_t idx = 0; idx < baseline_placements.size(); ++idx) {
                if ((int)idx == request_idx) {
                    continue;
                }
                apply_placement(*pending_seqs[idx], candidate_placements[idx], candidate_state);
            }

            auto repaired = build_placement(
                *pending_seqs[request_idx],
                candidate_state,
                target_sp,
                baseline_placements[request_idx].master_sp_idx);
            if (!repaired.has_value()) {
                continue;
            }

            candidate_placements[request_idx] = *repaired;
            apply_placement(*pending_seqs[request_idx], *repaired, candidate_state);

            LatencyBreakdown candidate_latency = estimate_latency(candidate_state);
            int candidate_overflow = total_overflow(candidate_state, running_state.group_max);
            int candidate_extra_participants = count_extra_participants(candidate_placements);

            if (better_candidate(candidate_latency,
                                 candidate_overflow,
                                 candidate_extra_participants,
                                 best_latency,
                                 best_overflow,
                                 best_extra_participants)) {
                best_latency = candidate_latency;
                best_overflow = candidate_overflow;
                best_extra_participants = candidate_extra_participants;
                best_placements = std::move(candidate_placements);
            }
        }
    }

    DecodeBatchPlan plan;
    plan.placements = std::move(best_placements);
    plan.latency = best_latency;
    plan.max_tokens = static_cast<int>(best_latency.max_tokens);
    plan.total_overflow = best_overflow;
    plan.extra_participants = best_extra_participants;
    return plan;
}

void SPStateManager::apply_planned_placement(Sequence& seq, const PlannedPlacement& placement)
{
    auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    block_ctx.master_sp_idx_ = placement.master_sp_idx;
    block_ctx.num_dispatched_tokens = placement.num_dispatched_tokens;
    block_ctx.block_location.clear();
    block_ctx.sp_block_table.assign(attention_sp_, {});

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
            int num_segments = (num_tokens + debug_segment_size - 1) / debug_segment_size;
            
            // SP Size is determined by number of segments
            int target_num_ranks = std::min(num_segments, attention_sp_);
            if (target_num_ranks == 0) target_num_ranks = 1;
            
            // Select Master Rank using Round Robin
            SPMasterSelector original_selector = master_selector_;
            master_selector_ = SPMasterSelector::RoundRobin;
            int master_rank = select_master_rank();
            master_selector_ = original_selector; // Restore original selector
            
            // Check Master Rank capacity
            if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) {
                return false;
            }
            
            auto it_tokens = num_batched_tokens.find(master_rank);
            int current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
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
                      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                          return a.second > b.second;
                      });
            
            // Select participating ranks (non-master ranks first)
            std::vector<int> participating_ranks;
            int non_master_ranks_needed = std::min((int)rank_free_count.size(), target_num_ranks - 1);
            for (int i = 0; i < non_master_ranks_needed; ++i) {
                participating_ranks.push_back(rank_free_count[i].first);
            }
            participating_ranks.push_back(master_rank); // Master is always included
            
            // Initialize token dispatch
            block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);
            block_ctx.master_sp_idx_ = master_rank;
            
            // Allocate tokens: non-master ranks get full segments, master gets the remainder (tail segment)
            int remaining_tokens = num_tokens;
            int segments_assigned_to_non_master = std::min(non_master_ranks_needed, num_segments - 1);
            
            // Assign full segments to non-master ranks
            for (int i = 0; i < segments_assigned_to_non_master; ++i) {
                int sp_idx = participating_ranks[i];
                block_ctx.num_dispatched_tokens[sp_idx] = debug_segment_size;
                remaining_tokens -= debug_segment_size;
            }
            
            // Assign remaining tokens (tail segment) to master rank
            block_ctx.num_dispatched_tokens[master_rank] = remaining_tokens;
            
            // Reservation Check
            std::vector<int> master_req_counts(attention_sp_, 0);
            for (const auto& running_seq : running) {
                int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx]++;
            }
            for (const auto& [m_idx, count] : num_seqs) {
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx] += count;
            }
            master_req_counts[master_rank]++;
            
            // Memory check
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (block_ctx.num_dispatched_tokens[sp_idx] > 0 || sp_idx == master_rank) {
                    
                    if (sp_idx != master_rank && block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                            return false;
                        }
                    }
                    
                    int free_blocks = block_manager[sp_idx]->num_free_blocks();
                    int prefill_tokens = block_ctx.num_dispatched_tokens[sp_idx];
                    int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
                    
                    double needed_float = master_req_counts[sp_idx] * reserved_blocks_per_req_;
                    int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));
                    
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
        // Use fixed_sp_segments if set, otherwise calculate based on segment_size
        int num_segments          = (fixed_sp_segments_ > 0) ? fixed_sp_segments_ : (num_tokens + segment_size_ - 1) / segment_size_;
        int num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int initial_num_ranks     = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        if (num_segments_per_rank == 0) num_segments_per_rank = 1;
        if (initial_num_ranks == 0) initial_num_ranks = 1;

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) return false;

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) return false;

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

        int start_ranks = initial_num_ranks;
        int end_ranks   = enable_dynamic_sp_size_ ? attention_sp_ : initial_num_ranks;
        bool recompute_segments_for_forced_sp = false;
        if (dynamic_sp_size_strategy_ == DynamicSPSizeStrategy::Bucket) {
            const int forced_num_ranks = std::max(
                1,
                std::min(
                    attention_sp_,
                    select_bucket_sp_size(seq.num_tokens).value_or(initial_num_ranks)));
            start_ranks = forced_num_ranks;
            end_ranks = forced_num_ranks;
            recompute_segments_for_forced_sp = true;
        } else if (dynamic_sp_size_strategy_ == DynamicSPSizeStrategy::LongShort) {
            const bool is_long_request = seq.num_prompt_tokens > long_request_sp_threshold_;
            const int forced_num_ranks = is_long_request ? long_request_sp_size_ : 1;
            start_ranks = forced_num_ranks;
            end_ranks = forced_num_ranks;
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
            if (enable_non_uniform_split_) {
                // 1. Collect free blocks info for all participating ranks (including master)
                std::vector<std::pair<int, int>> sorted_ranks; // {sp_idx, free_blocks}
                for (int sp_idx : top_most_free_ranks) {
                    sorted_ranks.push_back({sp_idx, block_manager[sp_idx]->num_free_blocks()});
                }
                
                // 2. Sort participating ranks by free blocks descending (richest first)
                std::sort(sorted_ranks.begin(), sorted_ranks.end(),
                          [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                              return a.second > b.second;
                          });

                long long total_tokens_needed = seq.num_tokens;
                long long final_target_free_tokens = 0;
                int k = 0; // Number of ranks contributing to "water-filling"

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
                    long long target_free = remaining_after_alloc / k;

                    // If we are at the last rank, or if the calculated target level is 
                    // higher than the next rank's capacity (meaning next rank doesn't need to help),
                    // then we found our split point.
                    if (k == (int)sorted_ranks.size()) {
                        final_target_free_tokens = target_free;
                        break;
                    } else {
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
                    int sp_idx = sorted_ranks[i].first;
                    long long current_free = (long long)sorted_ranks[i].second * kvcache_block_size_;
                    
                    // Alloc = Current - Target
                    long long alloc = current_free - final_target_free_tokens;
                    
                    if (alloc < 0) alloc = 0;
                    if (alloc > current_free) alloc = current_free; // Safety cap

                    block_ctx.num_dispatched_tokens[sp_idx] = (int)alloc;
                    allocated_sum += alloc;
                }

                // 5. Handle integer division remainders
                long long remainder = total_tokens_needed - allocated_sum;
                int idx = 0;
                
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
                int total_token_unalloc = seq.num_tokens;
                for (int sp_idx : top_most_free_ranks) {
                    int tokens_to_dispatch                  = std::min(total_token_unalloc, target_num_segments_per_rank * segment_size_);
                    block_ctx.num_dispatched_tokens[sp_idx] = tokens_to_dispatch;
                    total_token_unalloc -= tokens_to_dispatch;
                }
            }

            // Reservation Check
            std::vector<int> master_req_counts(attention_sp_, 0);
            for (const auto& running_seq : running) {
                int m_idx = running_seq->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx]++;
            }
            for (const auto& [m_idx, count] : num_seqs) {
                if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx] += count;
            }
            master_req_counts[master_rank]++;

            bool memory_check_passed = true;
            for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                if (block_ctx.num_dispatched_tokens[sp_idx] > 0 || sp_idx == master_rank) {
                    
                    if (sp_idx != master_rank && block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                            memory_check_passed = false;
                            break;
                        }
                    }

                    int free_blocks = block_manager[sp_idx]->num_free_blocks();
                    int prefill_tokens = block_ctx.num_dispatched_tokens[sp_idx];
                    int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

                    double needed_float = master_req_counts[sp_idx] * reserved_blocks_per_req_;
                    int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

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
        // Use fixed_sp_segments if set, otherwise calculate based on segment_size
        int num_segments          = (fixed_sp_segments_ > 0) ? fixed_sp_segments_ : (num_tokens + segment_size_ - 1) / segment_size_;
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
            int tokens_to_dispatch                  = std::min(total_token_unalloc, num_segments_per_rank * segment_size_);
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
            
            int prefill_tokens = block_ctx.num_dispatched_tokens[sp_idx];
            int prefill_blocks_needed = (prefill_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

            double needed_float = master_req_counts[sp_idx] * reserved_blocks_per_req_;
            int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

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

void SPStateManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    cached_running_state_.reset();
    auto& block_ctx     = seq.block_ctx(slot);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        if (master_seq_counts_[master_sp_idx] > 0) {
            master_seq_counts_[master_sp_idx]--;
        }
    }

    // [修改] 在清理 block_ctx 之前，先减少 Recv 计数
    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        if (block_ctx.num_dispatched_tokens[sp_idx] > 0) {
            if (sp_idx != master_sp_idx) {
                num_recv_seqs_per_sp_[sp_idx]--;
            }
        }
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
        block_manager[sp_idx]->deallocate(seq, slot);
    }

    block_ctx.sp_block_table.assign(attention_sp_, {});
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens;
}

}  // namespace nanodeploy
