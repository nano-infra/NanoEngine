#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"

namespace nanodeploy {

namespace {

class SmallDinic {
public:
    struct Edge {
        int to       = 0;
        int reverse  = 0;
        int capacity = 0;
    };

    explicit SmallDinic(int nodes): graph_(nodes), level_(nodes), next_edge_(nodes) {}

    void add_edge(int from, int to, int capacity)
    {
        int forward_reverse = static_cast<int>(graph_[to].size());
        int reverse_reverse = static_cast<int>(graph_[from].size());
        graph_[from].push_back({to, forward_reverse, capacity});
        graph_[to].push_back({from, reverse_reverse, 0});
    }

    int max_flow(int source, int sink, int limit)
    {
        int flow = 0;
        while (flow < limit && build_levels(source, sink)) {
            std::fill(next_edge_.begin(), next_edge_.end(), 0);
            while (flow < limit) {
                int pushed = send_flow(source, sink, limit - flow);
                if (pushed == 0) {
                    break;
                }
                flow += pushed;
            }
        }
        return flow;
    }

    const std::vector<Edge>& edges(int node) const
    {
        return graph_[node];
    }

private:
    bool build_levels(int source, int sink)
    {
        std::fill(level_.begin(), level_.end(), -1);
        std::queue<int> queue;
        level_[source] = 0;
        queue.push(source);
        while (!queue.empty()) {
            int node = queue.front();
            queue.pop();
            for (const auto& edge : graph_[node]) {
                if (edge.capacity > 0 && level_[edge.to] < 0) {
                    level_[edge.to] = level_[node] + 1;
                    queue.push(edge.to);
                }
            }
        }
        return level_[sink] >= 0;
    }

    int send_flow(int node, int sink, int available)
    {
        if (node == sink) {
            return available;
        }
        for (int& edge_idx = next_edge_[node]; edge_idx < static_cast<int>(graph_[node].size()); ++edge_idx) {
            auto& edge = graph_[node][edge_idx];
            if (edge.capacity <= 0 || level_[edge.to] != level_[node] + 1) {
                continue;
            }
            int pushed = send_flow(edge.to, sink, std::min(available, edge.capacity));
            if (pushed == 0) {
                continue;
            }
            edge.capacity -= pushed;
            graph_[edge.to][edge.reverse].capacity += pushed;
            return pushed;
        }
        return 0;
    }

    std::vector<std::vector<Edge>> graph_;
    std::vector<int>               level_;
    std::vector<int>               next_edge_;
};

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

bool block_context_equal(const BlockContext& lhs, const BlockContext& rhs) noexcept
{
    return lhs.engine_id_ == rhs.engine_id_ && lhs.dp_idx_ == rhs.dp_idx_
           && lhs.master_sp_idx_ == rhs.master_sp_idx_ && lhs.attention_sp_ == rhs.attention_sp_
           && lhs.attention_dp_ == rhs.attention_dp_ && lhs.pending_token_present_ == rhs.pending_token_present_
           && lhs.pending_token_target_sp_ == rhs.pending_token_target_sp_
           && lhs.block_location == rhs.block_location && lhs.sp_block_table == rhs.sp_block_table
           && lhs.num_dispatched_tokens == rhs.num_dispatched_tokens;
}

struct PreparedRankMutation {
    int                                 rank = -1;
    BlockManager::PreparedBlockMutation mutation;

    PreparedRankMutation(int rank, BlockManager::PreparedBlockMutation&& mutation) noexcept:
        rank(rank), mutation(std::move(mutation))
    {
    }

    PreparedRankMutation(const PreparedRankMutation&)            = delete;
    PreparedRankMutation& operator=(const PreparedRankMutation&) = delete;
    PreparedRankMutation(PreparedRankMutation&&) noexcept         = default;
    PreparedRankMutation& operator=(PreparedRankMutation&&) noexcept = default;
};

struct PreparedRankRebalance {
    int                                  rank = -1;
    BlockManager::PreparedBlockRebalance rebalance;

    PreparedRankRebalance(int rank, BlockManager::PreparedBlockRebalance&& rebalance) noexcept:
        rank(rank), rebalance(std::move(rebalance))
    {
    }

    PreparedRankRebalance(const PreparedRankRebalance&)            = delete;
    PreparedRankRebalance& operator=(const PreparedRankRebalance&) = delete;
    PreparedRankRebalance(PreparedRankRebalance&&) noexcept         = default;
    PreparedRankRebalance& operator=(PreparedRankRebalance&&) noexcept = default;
};

}  // namespace

struct SPStateManager::PreparedLSInitialBatch::Impl {
    SPStateManager*                                manager = nullptr;
    std::vector<std::shared_ptr<Sequence>>          sequences;
    std::vector<BlockContext>                       original_contexts;
    std::vector<BlockContext>                       shadow_contexts;
    std::vector<int>                                original_num_tokens;
    std::vector<size_t>                             original_token_sizes;
    std::vector<SequenceStatus>                     original_statuses;
    std::vector<PreparedRankMutation>               allocations;
    std::vector<int>                                master_counts_before;
    std::vector<int>                                master_counts_after;
    std::vector<int>                                recv_counts_before;
    std::vector<int>                                recv_counts_after;
    int                                             running_seqs_before   = 0;
    int                                             running_seqs_after    = 0;
    int                                             running_tokens_before = 0;
    int                                             running_tokens_after  = 0;
};

struct SPStateManager::PreparedLSRelease::Impl {
    SPStateManager*                  manager = nullptr;
    std::shared_ptr<Sequence>        sequence;
    BlockContext                    original_context;
    BlockContext                    shadow_context;
    int                             original_num_tokens   = 0;
    size_t                          original_token_size   = 0;
    int                             original_cached_tokens = 0;
    SequenceStatus                  original_status       = SequenceStatus::WAITING;
    std::vector<PreparedRankMutation> releases;
    std::vector<int>                master_counts_before;
    std::vector<int>                master_counts_after;
    std::vector<int>                recv_counts_before;
    std::vector<int>                recv_counts_after;
    int                             running_seqs_before   = 0;
    int                             running_seqs_after    = 0;
    int                             running_tokens_before = 0;
    int                             running_tokens_after  = 0;
};

struct SPStateManager::PreparedLSIterationMasterPlan::Impl {
    SPStateManager*                       manager = nullptr;
    std::vector<std::shared_ptr<Sequence>> sequences;
    std::vector<BlockContext>              original_contexts;
    std::vector<BlockContext>              shadow_contexts;
    std::vector<int>                       original_num_tokens;
    std::vector<size_t>                    original_token_sizes;
    std::vector<int>                       original_cached_tokens;
    std::vector<int>                       original_last_tokens;
    std::vector<SequenceStatus>            original_statuses;
    std::vector<PreparedRankRebalance>      rebalances;
    std::vector<int>                       master_counts_before;
    std::vector<int>                       master_counts_after;
    std::vector<int>                       recv_counts_before;
    std::vector<int>                       recv_counts_after;
    int                                    running_seqs_before   = 0;
    int                                    running_tokens_before = 0;
};

static_assert(std::is_nothrow_swappable_v<BlockContext>,
              "formal LS publication requires BlockContext swap to be noexcept");

SPStateManager::PreparedLSInitialBatch::PreparedLSInitialBatch(std::unique_ptr<Impl> impl) noexcept:
    impl_(std::move(impl)), state_(State::PREPARED)
{
}

SPStateManager::PreparedLSInitialBatch::PreparedLSInitialBatch(PreparedLSInitialBatch&& other) noexcept
{
    take_from(std::move(other));
}

SPStateManager::PreparedLSInitialBatch&
SPStateManager::PreparedLSInitialBatch::operator=(PreparedLSInitialBatch&& other) noexcept
{
    if (this != &other) {
        abort_noexcept();
        take_from(std::move(other));
    }
    return *this;
}

SPStateManager::PreparedLSInitialBatch::~PreparedLSInitialBatch() noexcept
{
    abort_noexcept();
}

void SPStateManager::PreparedLSInitialBatch::take_from(PreparedLSInitialBatch&& other) noexcept
{
    impl_  = std::move(other.impl_);
    state_ = std::exchange(other.state_, State::ABORTED);
}

bool SPStateManager::PreparedLSInitialBatch::validate_precommit_noexcept() const noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager
        || impl_->sequences.size() != impl_->original_contexts.size()
        || impl_->sequences.size() != impl_->original_num_tokens.size()
        || impl_->sequences.size() != impl_->original_token_sizes.size()
        || impl_->sequences.size() != impl_->original_statuses.size()) {
        return false;
    }
    const auto& manager = *impl_->manager;
    if (manager.master_seq_counts_ != impl_->master_counts_before
        || manager.num_recv_seqs_per_sp_ != impl_->recv_counts_before
        || manager.num_running_seqs_ != impl_->running_seqs_before
        || manager.num_running_tokens_ != impl_->running_tokens_before) {
        return false;
    }
    for (size_t idx = 0; idx < impl_->sequences.size(); ++idx) {
        const auto& sequence = impl_->sequences[idx];
        if (!sequence || sequence->num_tokens != impl_->original_num_tokens[idx]
            || sequence->token_ids.size() != impl_->original_token_sizes[idx]
            || sequence->status != impl_->original_statuses[idx]
            || !block_context_equal(sequence->block_ctx(BlockContextSlot::ACTIVE),
                                    impl_->original_contexts[idx])) {
            return false;
        }
    }
    return std::all_of(impl_->allocations.begin(), impl_->allocations.end(), [](const auto& allocation) {
        return allocation.mutation.state() == BlockManager::PreparedBlockMutation::State::PREPARED;
    });
}

void SPStateManager::PreparedLSInitialBatch::commit_noexcept() noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager) {
        return;
    }
    auto& manager = *impl_->manager;
    for (size_t idx = 0; idx < impl_->sequences.size(); ++idx) {
        using std::swap;
        swap(impl_->sequences[idx]->block_ctx(BlockContextSlot::ACTIVE), impl_->shadow_contexts[idx]);
    }
    for (auto& allocation : impl_->allocations) {
        allocation.mutation.commit_noexcept();
    }
    manager.master_seq_counts_.swap(impl_->master_counts_after);
    manager.num_recv_seqs_per_sp_.swap(impl_->recv_counts_after);
    manager.num_running_seqs_   = impl_->running_seqs_after;
    manager.num_running_tokens_ = impl_->running_tokens_after;
    manager.cached_running_state_.reset();
    state_ = State::COMMITTED;
}

void SPStateManager::PreparedLSInitialBatch::abort_noexcept() noexcept
{
    if (state_ != State::PREPARED) {
        return;
    }
    if (impl_) {
        for (auto& allocation : impl_->allocations) {
            allocation.mutation.abort_noexcept();
        }
    }
    state_ = State::ABORTED;
}

SPStateManager::PreparedLSRelease::PreparedLSRelease(std::unique_ptr<Impl> impl) noexcept:
    impl_(std::move(impl)), state_(State::PREPARED)
{
}

SPStateManager::PreparedLSRelease::PreparedLSRelease(PreparedLSRelease&& other) noexcept
{
    take_from(std::move(other));
}

SPStateManager::PreparedLSRelease&
SPStateManager::PreparedLSRelease::operator=(PreparedLSRelease&& other) noexcept
{
    if (this != &other) {
        abort_noexcept();
        take_from(std::move(other));
    }
    return *this;
}

SPStateManager::PreparedLSRelease::~PreparedLSRelease() noexcept
{
    abort_noexcept();
}

void SPStateManager::PreparedLSRelease::take_from(PreparedLSRelease&& other) noexcept
{
    impl_  = std::move(other.impl_);
    state_ = std::exchange(other.state_, State::ABORTED);
}

bool SPStateManager::PreparedLSRelease::validate_precommit_noexcept() const noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager || !impl_->sequence) {
        return false;
    }
    const auto& manager = *impl_->manager;
    const auto& sequence = impl_->sequence;
    if (manager.master_seq_counts_ != impl_->master_counts_before
        || manager.num_recv_seqs_per_sp_ != impl_->recv_counts_before
        || manager.num_running_seqs_ != impl_->running_seqs_before
        || manager.num_running_tokens_ != impl_->running_tokens_before
        || sequence->num_tokens != impl_->original_num_tokens
        || sequence->token_ids.size() != impl_->original_token_size
        || sequence->num_cached_tokens != impl_->original_cached_tokens
        || sequence->status != impl_->original_status
        || !block_context_equal(sequence->block_ctx(BlockContextSlot::ACTIVE), impl_->original_context)) {
        return false;
    }
    return std::all_of(impl_->releases.begin(), impl_->releases.end(), [](const auto& release) {
        return release.mutation.state() == BlockManager::PreparedBlockMutation::State::PREPARED;
    });
}

void SPStateManager::PreparedLSRelease::commit_noexcept() noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager || !impl_->sequence) {
        return;
    }
    auto& manager = *impl_->manager;
    using std::swap;
    swap(impl_->sequence->block_ctx(BlockContextSlot::ACTIVE), impl_->shadow_context);
    for (auto& release : impl_->releases) {
        release.mutation.commit_noexcept();
    }
    manager.master_seq_counts_.swap(impl_->master_counts_after);
    manager.num_recv_seqs_per_sp_.swap(impl_->recv_counts_after);
    manager.num_running_seqs_   = impl_->running_seqs_after;
    manager.num_running_tokens_ = impl_->running_tokens_after;
    manager.cached_running_state_.reset();
    impl_->sequence->num_cached_tokens = 0;
    state_ = State::COMMITTED;
}

void SPStateManager::PreparedLSRelease::abort_noexcept() noexcept
{
    if (state_ != State::PREPARED) {
        return;
    }
    if (impl_) {
        for (auto& release : impl_->releases) {
            release.mutation.abort_noexcept();
        }
    }
    state_ = State::ABORTED;
}

SPStateManager::PreparedLSIterationMasterPlan::PreparedLSIterationMasterPlan(std::unique_ptr<Impl> impl) noexcept:
    impl_(std::move(impl)), state_(State::PREPARED)
{
}

SPStateManager::PreparedLSIterationMasterPlan::PreparedLSIterationMasterPlan(
    PreparedLSIterationMasterPlan&& other) noexcept
{
    take_from(std::move(other));
}

SPStateManager::PreparedLSIterationMasterPlan&
SPStateManager::PreparedLSIterationMasterPlan::operator=(PreparedLSIterationMasterPlan&& other) noexcept
{
    if (this != &other) {
        abort_noexcept();
        take_from(std::move(other));
    }
    return *this;
}

SPStateManager::PreparedLSIterationMasterPlan::~PreparedLSIterationMasterPlan() noexcept
{
    abort_noexcept();
}

void SPStateManager::PreparedLSIterationMasterPlan::take_from(PreparedLSIterationMasterPlan&& other) noexcept
{
    impl_  = std::move(other.impl_);
    state_ = std::exchange(other.state_, State::ABORTED);
}

bool SPStateManager::PreparedLSIterationMasterPlan::validate_precommit_noexcept() const noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager
        || impl_->sequences.size() != impl_->original_contexts.size()
        || impl_->sequences.size() != impl_->original_num_tokens.size()
        || impl_->sequences.size() != impl_->original_token_sizes.size()
        || impl_->sequences.size() != impl_->original_cached_tokens.size()
        || impl_->sequences.size() != impl_->original_last_tokens.size()
        || impl_->sequences.size() != impl_->original_statuses.size()) {
        return false;
    }
    const auto& manager = *impl_->manager;
    if (manager.master_seq_counts_ != impl_->master_counts_before
        || manager.num_recv_seqs_per_sp_ != impl_->recv_counts_before
        || manager.num_running_seqs_ != impl_->running_seqs_before
        || manager.num_running_tokens_ != impl_->running_tokens_before) {
        return false;
    }
    for (size_t idx = 0; idx < impl_->sequences.size(); ++idx) {
        const auto& sequence = impl_->sequences[idx];
        if (!sequence || sequence->num_tokens != impl_->original_num_tokens[idx]
            || sequence->token_ids.size() != impl_->original_token_sizes[idx]
            || sequence->num_cached_tokens != impl_->original_cached_tokens[idx]
            || sequence->last_token != impl_->original_last_tokens[idx]
            || sequence->status != impl_->original_statuses[idx]
            || !block_context_equal(sequence->block_ctx(BlockContextSlot::ACTIVE),
                                    impl_->original_contexts[idx])) {
            return false;
        }
    }
    return std::all_of(impl_->rebalances.begin(), impl_->rebalances.end(), [](const auto& rank) {
        return rank.rebalance.state() == BlockManager::PreparedBlockRebalance::State::PREPARED;
    });
}

void SPStateManager::PreparedLSIterationMasterPlan::commit_noexcept() noexcept
{
    if (state_ != State::PREPARED || !impl_ || !impl_->manager) {
        return;
    }
    auto& manager = *impl_->manager;
    for (size_t idx = 0; idx < impl_->sequences.size(); ++idx) {
        using std::swap;
        swap(impl_->sequences[idx]->block_ctx(BlockContextSlot::ACTIVE), impl_->shadow_contexts[idx]);
    }
    for (auto& rank : impl_->rebalances) {
        rank.rebalance.commit_noexcept();
    }
    // This transaction may be composed with a prepared initial admission in
    // the same pool step. The admission is published first and contributes new
    // absolute role counters; applying the precomputed iteration delta keeps
    // those contributions instead of overwriting them with the iteration's
    // stable-state snapshot.
    for (size_t rank = 0; rank < impl_->master_counts_before.size(); ++rank) {
        manager.master_seq_counts_[rank] += impl_->master_counts_after[rank] - impl_->master_counts_before[rank];
        manager.num_recv_seqs_per_sp_[rank] += impl_->recv_counts_after[rank] - impl_->recv_counts_before[rank];
    }
    manager.cached_running_state_.reset();
    state_ = State::COMMITTED;
}

void SPStateManager::PreparedLSIterationMasterPlan::abort_noexcept() noexcept
{
    if (state_ != State::PREPARED) {
        return;
    }
    if (impl_) {
        for (auto& rank : impl_->rebalances) {
            rank.rebalance.abort_noexcept();
        }
    }
    state_ = State::ABORTED;
}

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
    // LoongServe consumes the idle-instance stack from the end. Nano exposes
    // deterministic SP ranks, so the Decode-only adapter uses rank descending.
    std::stable_sort(extras.begin(), extras.end(), std::greater<int>());

    bool      compute_scaled  = false;
    bool      memory_scaled   = false;
    bool      receiver_scaled = false;
    const int max_restarts    = attention_sp_ + 1;

    // Source compute-bound expansion is decided once from the complete real
    // step-entry membership. It uses integer floor division exactly; suffix
    // chunking below must not request extra ranks after part of the batch has
    // already been assigned.
    while (!current_allocation.empty() && !extras.empty()
           && static_cast<int>(requests.size()) / static_cast<int>(current_allocation.size()) > batch_per_master) {
        int rank = extras.front();
        extras.erase(extras.begin());
        current_allocation.push_back(rank);
        result.new_allocation_ranks.push_back(rank);
        compute_scaled = true;
    }

    auto set_scale_reason = [&](LSDecodeMasterPlan& plan) {
        plan.scale_reason.clear();
        auto append_reason = [&](const std::string& reason) {
            if (!plan.scale_reason.empty()) {
                plan.scale_reason += "+";
            }
            plan.scale_reason += reason;
        };
        if (compute_scaled) {
            append_reason("compute");
        }
        if (memory_scaled) {
            append_reason("memory");
        }
        if (receiver_scaled) {
            append_reason("receiver");
        }
        if (plan.scale_reason.empty()) {
            plan.scale_reason = "none";
        }
    };

    struct SourceGreedyAttempt {
        bool             success       = false;
        bool             request_scale = false;
        bool             memory_scale  = false;
        std::string      failure_reason;
        std::vector<int> master_ranks;
        std::vector<int> master_batch_sizes;
        std::vector<int> sequence_master_ranks;
    };

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

        auto run_source_greedy = [&](const std::vector<size_t>& planning_order) {
            SourceGreedyAttempt attempt;
            attempt.sequence_master_ranks.assign(requests.size(), -1);
            std::vector<int> planned_remote_recv(attention_sp_, 0);
            size_t           remaining_begin = 0;
            size_t           candidate_begin = 0;

            auto ordered_suffix = [&](size_t begin) {
                std::vector<std::shared_ptr<Sequence>> suffix;
                suffix.reserve(planning_order.size() - begin);
                for (size_t order_idx = begin; order_idx < planning_order.size(); ++order_idx) {
                    suffix.push_back(requests[planning_order[order_idx]]);
                }
                return suffix;
            };

            auto receiver_prefix_capacity = [&](int rank, size_t begin) {
                std::vector<int> simulated = planned_remote_recv;
                int              accepted  = 0;
                for (size_t order_idx = begin; order_idx < planning_order.size(); ++order_idx) {
                    size_t request_idx = planning_order[order_idx];
                    bool   feasible    = true;
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

            while (remaining_begin < planning_order.size()) {
                auto                suffix = ordered_suffix(remaining_begin);
                std::vector<size_t> append_capable;
                int                 compute_capable = 0;
                for (size_t idx = candidate_begin; idx < candidates.size(); ++idx) {
                    int rank = candidates[idx];
                    if (estimate_pending_append_capacity(rank, suffix, requests) <= 0) {
                        continue;
                    }
                    compute_capable++;
                    if (receiver_prefix_capacity(rank, remaining_begin) > 0) {
                        append_capable.push_back(idx);
                    }
                }

                if (append_capable.empty()) {
                    if (enable_memory_scale_up && !extras.empty()) {
                        attempt.request_scale = true;
                        attempt.memory_scale  = true;
                    }
                    else {
                        attempt.failure_reason = "source-greedy append/receiver prefix exhausted";
                    }
                    return attempt;
                }

                int n_left    = static_cast<int>(append_capable.size());
                int remaining = static_cast<int>(planning_order.size() - remaining_begin);
                size_t rank_pos  = append_capable.front();
                int    rank      = candidates[rank_pos];
                int    capacity  = estimate_pending_append_capacity(rank, suffix, requests);
                capacity         = std::min(capacity, receiver_prefix_capacity(rank, remaining_begin));
                int target_chunk = std::max(remaining / n_left, batch_per_master);
                int chunk        = std::min({remaining, target_chunk, capacity});
                candidate_begin  = rank_pos + 1;
                if (chunk <= 0) {
                    continue;
                }
                attempt.master_ranks.push_back(rank);
                attempt.master_batch_sizes.push_back(chunk);
                for (int i = 0; i < chunk; ++i) {
                    size_t request_idx                         = planning_order[remaining_begin + i];
                    attempt.sequence_master_ranks[request_idx] = rank;
                    for (int owner = 0; owner < attention_sp_; ++owner) {
                        if (owner != rank
                            && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, owner) > 0) {
                            planned_remote_recv[owner]++;
                        }
                    }
                }
                remaining_begin += chunk;
            }
            attempt.success = true;
            return attempt;
        };

        std::vector<size_t> identity_order(requests.size());
        std::iota(identity_order.begin(), identity_order.end(), 0);
        auto attempt = run_source_greedy(identity_order);
        auto return_attempt = [&](SourceGreedyAttempt&& successful, const std::string& strategy) {
            result.success               = true;
            result.assignment_strategy   = strategy;
            result.allocation            = current_allocation;
            result.master_ranks          = std::move(successful.master_ranks);
            result.master_batch_sizes    = std::move(successful.master_batch_sizes);
            result.sequence_master_ranks = std::move(successful.sequence_master_ranks);
            set_scale_reason(result);
        };

        if (attempt.success) {
            return_attempt(std::move(attempt), "source_greedy");
            return result;
        }

        // The source-greedy fast path consumes a contiguous request prefix per
        // candidate. That ordering is not a correctness constraint and can
        // reject a feasible owner-local assignment. Retry once with stable
        // current-master buckets while preserving candidate and chunk policy.
        std::vector<std::vector<size_t>> owner_buckets(candidates.size());
        for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
            int    previous_master = requests[request_idx]->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            size_t bucket          = candidates.size();
            for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                if (candidates[candidate_idx] == previous_master) {
                    bucket = candidate_idx;
                    break;
                }
            }
            if (bucket == candidates.size() && !candidates.empty()) {
                bucket          = 0;
                int best_tokens = -1;
                for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                    int tokens = requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE,
                                                                              candidates[candidate_idx]);
                    if (tokens > best_tokens) {
                        best_tokens = tokens;
                        bucket      = candidate_idx;
                    }
                }
            }
            if (bucket < owner_buckets.size()) {
                owner_buckets[bucket].push_back(request_idx);
            }
        }
        std::vector<size_t> owner_order;
        owner_order.reserve(requests.size());
        for (const auto& bucket : owner_buckets) {
            owner_order.insert(owner_order.end(), bucket.begin(), bucket.end());
        }
        if (owner_order.size() == requests.size() && owner_order != identity_order) {
            auto owner_attempt = run_source_greedy(owner_order);
            if (owner_attempt.success) {
                return_attempt(std::move(owner_attempt), "owner_bucket_repair");
                return result;
            }
        }

        // Merging two previously legal groups should not make their existing
        // master placement illegal merely because request order changed. Keep
        // the sticky placement when the authoritative validator accepts it.
        LSDecodeMasterPlan sticky = result;
        sticky.success            = true;
        sticky.failure_reason.clear();
        sticky.assignment_strategy = "sticky_repair";
        sticky.allocation          = current_allocation;
        sticky.master_ranks.clear();
        sticky.master_batch_sizes.clear();
        sticky.sequence_master_ranks.assign(requests.size(), -1);
        std::vector<int> sticky_load(attention_sp_, 0);
        bool             sticky_complete = true;
        for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
            int master = requests[request_idx]->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (std::find(current_allocation.begin(), current_allocation.end(), master) == current_allocation.end()) {
                sticky_complete = false;
                break;
            }
            sticky.sequence_master_ranks[request_idx] = master;
            sticky_load[master]++;
        }
        if (sticky_complete) {
            for (int rank : candidates) {
                if (sticky_load[rank] > 0) {
                    sticky.master_ranks.push_back(rank);
                    sticky.master_batch_sizes.push_back(sticky_load[rank]);
                }
            }
            set_scale_reason(sticky);
            if (validate_iteration_master_plan(requests, sticky)) {
                return sticky;
            }
        }

        // Exact receiver feasibility fallback. Receiver load on rank r is
        // owner_count[r] minus requests assigned locally to r, so rank r needs
        // at least max(0, owner_count[r] - max_num_recv_seqs_) local-owner
        // assignments. A capacitated bipartite matching over these quota slots
        // is complete for receiver and per-master count constraints.
        const int        batch_size   = static_cast<int>(requests.size());
        const int        metadata_cap = std::min(max_num_seqs_, max_num_batched_tokens_);
        std::vector<int> owner_count(attention_sp_, 0);
        for (const auto& seq : requests) {
            for (int rank = 0; rank < attention_sp_; ++rank) {
                if (seq->committed_context_len(BlockContextSlot::ACTIVE, rank) > 0) {
                    owner_count[rank]++;
                }
            }
        }

        std::vector<int> candidate_pos(attention_sp_, -1);
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            candidate_pos[candidates[idx]] = static_cast<int>(idx);
        }
        std::vector<int> lower(candidates.size(), 0);
        bool             receiver_rank_missing = false;
        bool             receiver_restart      = false;
        int              total_lower           = 0;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            int required = std::max(0, owner_count[rank] - max_num_recv_seqs_);
            if (required == 0) {
                continue;
            }
            if (candidate_pos[rank] < 0) {
                auto extra = std::find(extras.begin(), extras.end(), rank);
                if (extra != extras.end()) {
                    extras.erase(extra);
                    current_allocation.push_back(rank);
                    result.new_allocation_ranks.push_back(rank);
                    receiver_scaled  = true;
                    receiver_restart = true;
                }
                else {
                    receiver_rank_missing = true;
                }
                break;
            }
            lower[candidate_pos[rank]] = required;
            total_lower += required;
        }
        if (receiver_restart) {
            continue;
        }
        if (receiver_rank_missing || total_lower > batch_size) {
            result.failure_reason = "receiver capacity proven infeasible: owner-local quota exceeds request supply";
            return result;
        }
        if (std::any_of(lower.begin(), lower.end(), [&](int required) { return required > metadata_cap; })) {
            result.failure_reason = "decode metadata capacity cannot cover receiver-feasible assignment";
            return result;
        }
        if (static_cast<int>(candidates.size()) * metadata_cap < batch_size) {
            if (!extras.empty()) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                compute_scaled = true;
                continue;
            }
            result.failure_reason = "decode metadata capacity cannot cover receiver-feasible assignment";
            return result;
        }

        auto append_cost = [&](size_t request_idx, int rank) {
            int committed = requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, rank);
            int before    = (committed + kvcache_block_size_ - 1) / kvcache_block_size_;
            int after     = (committed + 2 + kvcache_block_size_ - 1) / kvcache_block_size_;
            return after - before;
        };

        auto reclaimable_free_blocks = [&](int rank) {
            auto manager = block_manager.find(rank);
            if (manager == block_manager.end()) {
                return 0;
            }
            int free_blocks = manager->second->num_free_blocks();
            for (const auto& seq : requests) {
                int committed        = seq->committed_context_len(BlockContextSlot::ACTIVE, rank);
                int committed_blocks = (committed + kvcache_block_size_ - 1) / kvcache_block_size_;
                int table_blocks     = static_cast<int>(seq->block_table(BlockContextSlot::ACTIVE, rank).size());
                free_blocks += std::max(0, table_blocks - committed_blocks);
            }
            return free_blocks;
        };

        // With the current +2-token reservation and block_size >= 2, per-request
        // append cost is binary. Cost-0 requests are necessarily local owners,
        // so append and receiver quotas are nested and can be solved exactly by
        // one capacity-flow assignment below. Compute an existence upper bound
        // using the cheapest possible subset, not the old worst-prefix bound.
        std::vector<int> upper(candidates.size(), 0);
        std::vector<int> free_blocks(candidates.size(), 0);
        std::vector<int> cheap_supply(candidates.size(), 0);
        for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
            int rank                   = candidates[candidate_idx];
            free_blocks[candidate_idx] = reclaimable_free_blocks(rank);
            int cheap_count            = 0;
            for (size_t request_idx = 0; request_idx < requests.size(); ++request_idx) {
                if (append_cost(request_idx, rank) == 0) {
                    cheap_count++;
                }
            }
            cheap_supply[candidate_idx] = cheap_count;
            if (kvcache_block_size_ < 2) {
                std::vector<size_t> worst_indices(requests.size());
                std::iota(worst_indices.begin(), worst_indices.end(), 0);
                std::stable_sort(worst_indices.begin(), worst_indices.end(), [&](size_t lhs, size_t rhs) {
                    return append_cost(lhs, rank) > append_cost(rhs, rank);
                });
                std::vector<std::shared_ptr<Sequence>> worst_requests;
                worst_requests.reserve(requests.size());
                for (size_t request_idx : worst_indices) {
                    worst_requests.push_back(requests[request_idx]);
                }
                upper[candidate_idx] =
                    std::min(metadata_cap, estimate_pending_append_capacity(rank, worst_requests, requests));
                continue;
            }
            for (int load = 0; load <= metadata_cap; ++load) {
                int minimum_append_blocks = std::max(0, load - cheap_count);
                int reserve_headroom      = static_cast<int>(std::ceil(load * reserved_blocks_per_req_));
                if (minimum_append_blocks + reserve_headroom <= free_blocks[candidate_idx]) {
                    upper[candidate_idx] = load;
                }
            }
        }
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            if (lower[idx] > upper[idx]) {
                result.failure_reason = "append capacity cannot satisfy required receiver-local quota";
                return result;
            }
        }
        if (std::accumulate(upper.begin(), upper.end(), 0) < batch_size) {
            if (enable_memory_scale_up && !extras.empty()) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                memory_scaled = true;
                continue;
            }
            result.failure_reason = "append capacity cannot cover receiver-feasible assignment";
            return result;
        }

        std::vector<int> nominal(candidates.size(), 0);
        int              nominal_remaining = batch_size;
        for (size_t idx = 0; idx < candidates.size() && nominal_remaining > 0; ++idx) {
            int ranks_left = static_cast<int>(candidates.size() - idx);
            int chunk      = std::min(nominal_remaining, std::max(nominal_remaining / ranks_left, batch_per_master));
            nominal[idx]   = chunk;
            nominal_remaining -= chunk;
        }

        std::vector<int> target_load    = lower;
        int              load_remaining = batch_size - total_lower;
        for (size_t idx = 0; idx < candidates.size() && load_remaining > 0; ++idx) {
            int desired = std::max(0, std::min(nominal[idx], upper[idx]) - target_load[idx]);
            int add     = std::min(load_remaining, desired);
            target_load[idx] += add;
            load_remaining -= add;
        }
        for (size_t idx = 0; idx < candidates.size() && load_remaining > 0; ++idx) {
            int add = std::min(load_remaining, upper[idx] - target_load[idx]);
            target_load[idx] += add;
            load_remaining -= add;
        }
        if (load_remaining != 0) {
            if (enable_memory_scale_up && !extras.empty()) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                memory_scaled = true;
                continue;
            }
            result.failure_reason = "append capacity cannot realize receiver-feasible master loads";
            return result;
        }

        const int  flow_source       = 0;
        const int  flow_request_base = 1;
        const int  flow_rank_base    = flow_request_base + batch_size;
        const int  flow_sink         = flow_rank_base + static_cast<int>(candidates.size());
        SmallDinic quota_flow(flow_sink + 1);
        for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
            quota_flow.add_edge(flow_source, flow_request_base + request_idx, 1);
            for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                int rank = candidates[candidate_idx];
                if (lower[candidate_idx] > 0
                    && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, rank) > 0) {
                    quota_flow.add_edge(
                        flow_request_base + request_idx, flow_rank_base + static_cast<int>(candidate_idx), 1);
                }
            }
        }
        for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
            quota_flow.add_edge(flow_rank_base + static_cast<int>(candidate_idx), flow_sink, lower[candidate_idx]);
        }
        int matched_lower = quota_flow.max_flow(flow_source, flow_sink, total_lower);
        if (matched_lower != total_lower) {
            result.failure_reason = "receiver capacity proven infeasible: owner-local quota matching failed";
            return result;
        }

        struct JointAssignmentAttempt {
            bool             success = false;
            std::vector<int> sequence_master_ranks;
        };
        constexpr int categories_per_rank = 3;
        auto solve_category_assignment = [&](const std::vector<std::array<int, categories_per_rank>>& category_capacity,
                                             int mandatory_category_count) {
            JointAssignmentAttempt assignment;
            const int              assignment_source        = 0;
            const int              assignment_request_base  = 1;
            const int              assignment_category_base = assignment_request_base + batch_size;
            const int              assignment_sink =
                assignment_category_base + categories_per_rank * static_cast<int>(candidates.size());
            SmallDinic assignment_flow(assignment_sink + 1);
            for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
                assignment_flow.add_edge(assignment_source, assignment_request_base + request_idx, 1);
            }

            auto add_category_range = [&](int category_begin, int category_end) {
                int added_capacity = 0;
                for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                    for (int category = category_begin; category < category_end; ++category) {
                        int capacity = category_capacity[candidate_idx][category];
                        if (capacity <= 0) {
                            continue;
                        }
                        int node =
                            assignment_category_base + categories_per_rank * static_cast<int>(candidate_idx) + category;
                        assignment_flow.add_edge(node, assignment_sink, capacity);
                        added_capacity += capacity;
                    }
                }

                for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
                    int                 request_node = assignment_request_base + request_idx;
                    std::vector<size_t> candidate_order;
                    int previous = requests[request_idx]->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
                    if (previous >= 0 && previous < attention_sp_ && candidate_pos[previous] >= 0) {
                        candidate_order.push_back(static_cast<size_t>(candidate_pos[previous]));
                    }
                    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                        if (candidate_order.empty() || candidate_order.front() != candidate_idx) {
                            candidate_order.push_back(candidate_idx);
                        }
                    }
                    for (size_t candidate_idx : candidate_order) {
                        int rank = candidates[candidate_idx];
                        int node_base =
                            assignment_category_base + categories_per_rank * static_cast<int>(candidate_idx);
                        for (int category = category_begin; category < category_end; ++category) {
                            if (category_capacity[candidate_idx][category] <= 0) {
                                continue;
                            }
                            bool eligible =
                                category == 2 || (category == 0 && append_cost(request_idx, rank) == 0)
                                || (category == 1
                                    && requests[request_idx]->committed_context_len(BlockContextSlot::ACTIVE, rank)
                                           > 0);
                            if (eligible) {
                                assignment_flow.add_edge(request_node, node_base + category, 1);
                            }
                        }
                    }
                }
                return added_capacity;
            };

            int mandatory_capacity = add_category_range(0, mandatory_category_count);
            if (mandatory_capacity > batch_size
                || assignment_flow.max_flow(assignment_source, assignment_sink, mandatory_capacity)
                       != mandatory_capacity) {
                return assignment;
            }
            int optional_capacity = add_category_range(mandatory_category_count, categories_per_rank);
            int remaining         = batch_size - mandatory_capacity;
            if (optional_capacity < remaining
                || assignment_flow.max_flow(assignment_source, assignment_sink, remaining) != remaining) {
                return assignment;
            }

            assignment.sequence_master_ranks.assign(requests.size(), -1);
            for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
                for (const auto& edge : assignment_flow.edges(assignment_request_base + request_idx)) {
                    if (edge.to < assignment_category_base || edge.to >= assignment_sink || edge.capacity != 0) {
                        continue;
                    }
                    int candidate_idx = (edge.to - assignment_category_base) / categories_per_rank;
                    assignment.sequence_master_ranks[request_idx] = candidates[candidate_idx];
                    break;
                }
                if (assignment.sequence_master_ranks[request_idx] < 0) {
                    assignment.sequence_master_ranks.clear();
                    return assignment;
                }
            }
            assignment.success = true;
            return assignment;
        };

        auto try_target_load = [&](const std::vector<int>& load_vector) {
            std::vector<std::array<int, categories_per_rank>> category_capacity(candidates.size());
            for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                int load           = load_vector[candidate_idx];
                int required_cheap = 0;
                if (kvcache_block_size_ >= 2) {
                    int reserve_headroom = static_cast<int>(std::ceil(load * reserved_blocks_per_req_));
                    int cost_one_budget  = std::max(0, free_blocks[candidate_idx] - reserve_headroom);
                    required_cheap       = std::max(0, load - cost_one_budget);
                }
                int required_owner = std::max(0, lower[candidate_idx] - required_cheap);
                int general        = load - std::max(lower[candidate_idx], required_cheap);
                if (required_cheap > load || general < 0) {
                    return JointAssignmentAttempt{};
                }
                category_capacity[candidate_idx] = {required_cheap, required_owner, general};
            }
            return solve_category_assignment(category_capacity, categories_per_rank);
        };

        auto exact_assignment     = try_target_load(target_load);
        bool target_load_repaired = false;
        int  load_search_states   = 0;
        if (!exact_assignment.success) {
            // The source-style target load is a preference, not a feasibility
            // constraint. Search variable load bounds exactly. For any relaxed
            // assignment that exceeds rank r's append budget, every legal
            // solution either has a smaller load on r or supplies at least the
            // newly derived number of cheap (cost-0) requests there. Those two
            // branches are exhaustive and each strictly tightens one bound.
            struct LoadSearchNode {
                std::vector<int> upper_load;
                std::vector<int> cheap_lower;
            };
            std::vector<LoadSearchNode> load_stack;
            std::set<std::vector<int>>  visited_bounds;
            load_stack.push_back({upper, std::vector<int>(candidates.size(), 0)});

            auto bounds_key = [](const LoadSearchNode& node) {
                std::vector<int> key = node.upper_load;
                key.insert(key.end(), node.cheap_lower.begin(), node.cheap_lower.end());
                return key;
            };
            visited_bounds.insert(bounds_key(load_stack.back()));

            while (!load_stack.empty() && !exact_assignment.success) {
                auto node = std::move(load_stack.back());
                load_stack.pop_back();
                load_search_states++;

                int                                               minimum_total = 0;
                int                                               maximum_total = 0;
                bool                                              bounds_valid  = true;
                std::vector<std::array<int, categories_per_rank>> category_capacity(candidates.size());
                for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                    int mandatory_cheap = node.cheap_lower[candidate_idx];
                    int mandatory_owner = std::max(0, lower[candidate_idx] - mandatory_cheap);
                    int minimum_load    = std::max(lower[candidate_idx], mandatory_cheap);
                    int maximum_load    = node.upper_load[candidate_idx];
                    if (mandatory_cheap > cheap_supply[candidate_idx] || minimum_load > maximum_load) {
                        bounds_valid = false;
                        break;
                    }
                    category_capacity[candidate_idx] = {mandatory_cheap, mandatory_owner, maximum_load - minimum_load};
                    minimum_total += minimum_load;
                    maximum_total += maximum_load;
                }
                if (!bounds_valid || minimum_total > batch_size || maximum_total < batch_size) {
                    continue;
                }

                auto relaxed_assignment = solve_category_assignment(category_capacity, 2);
                if (!relaxed_assignment.success) {
                    continue;
                }

                std::vector<int> actual_load(candidates.size(), 0);
                std::vector<int> actual_cheap(candidates.size(), 0);
                std::vector<int> actual_append_blocks(candidates.size(), 0);
                for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
                    int rank          = relaxed_assignment.sequence_master_ranks[request_idx];
                    int candidate_idx = candidate_pos[rank];
                    actual_load[candidate_idx]++;
                    int cost = append_cost(request_idx, rank);
                    actual_append_blocks[candidate_idx] += cost;
                    if (cost == 0) {
                        actual_cheap[candidate_idx]++;
                    }
                }

                int violating_candidate = -1;
                int largest_deficit     = 0;
                for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
                    int reserve_headroom =
                        static_cast<int>(std::ceil(actual_load[candidate_idx] * reserved_blocks_per_req_));
                    int deficit = actual_append_blocks[candidate_idx] + reserve_headroom - free_blocks[candidate_idx];
                    if (deficit > largest_deficit) {
                        largest_deficit     = deficit;
                        violating_candidate = static_cast<int>(candidate_idx);
                    }
                }
                if (violating_candidate < 0) {
                    exact_assignment     = std::move(relaxed_assignment);
                    target_load_repaired = true;
                    break;
                }

                int candidate_idx = violating_candidate;
                int load          = actual_load[candidate_idx];
                if (load > 0) {
                    LoadSearchNode smaller_load            = node;
                    smaller_load.upper_load[candidate_idx] = std::min(smaller_load.upper_load[candidate_idx], load - 1);
                    auto key                               = bounds_key(smaller_load);
                    if (visited_bounds.insert(key).second) {
                        load_stack.push_back(std::move(smaller_load));
                    }
                }

                if (kvcache_block_size_ >= 2) {
                    int reserve_headroom = static_cast<int>(std::ceil(load * reserved_blocks_per_req_));
                    int needed_cheap     = std::max(0, load + reserve_headroom - free_blocks[candidate_idx]);
                    if (needed_cheap > node.cheap_lower[candidate_idx] && needed_cheap <= cheap_supply[candidate_idx]) {
                        LoadSearchNode more_cheap             = node;
                        more_cheap.cheap_lower[candidate_idx] = needed_cheap;
                        auto key                              = bounds_key(more_cheap);
                        if (visited_bounds.insert(key).second) {
                            // LIFO ordering explores the cheap-preserving
                            // branch before reducing the preferred load.
                            load_stack.push_back(std::move(more_cheap));
                        }
                    }
                }
                else if (actual_cheap[candidate_idx] != 0) {
                    // With block size 1 append cost is constant, so the
                    // precomputed upper bound alone is exact.
                    throw std::runtime_error("unexpected cheap append with block size 1");
                }
            }
        }
        if (!exact_assignment.success) {
            if (enable_memory_scale_up && !extras.empty()) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                memory_scaled = true;
                continue;
            }
            result.failure_reason = "append/receiver joint capacity proven infeasible after "
                                    + std::to_string(load_search_states) + " variable-load states";
            return result;
        }

        LSDecodeMasterPlan exact = result;
        exact.success            = true;
        exact.failure_reason.clear();
        exact.assignment_strategy = target_load_repaired ? "receiver_append_flow_load_repair" : "receiver_append_flow";
        exact.allocation          = current_allocation;
        exact.master_ranks.clear();
        exact.master_batch_sizes.clear();
        exact.sequence_master_ranks = std::move(exact_assignment.sequence_master_ranks);
        std::vector<int> exact_load(attention_sp_, 0);
        for (int request_idx = 0; request_idx < batch_size; ++request_idx) {
            exact_load[exact.sequence_master_ranks[request_idx]]++;
        }
        for (int rank : candidates) {
            if (exact_load[rank] > 0) {
                exact.master_ranks.push_back(rank);
                exact.master_batch_sizes.push_back(exact_load[rank]);
            }
        }
        set_scale_reason(exact);
        std::string exact_error;
        if (!validate_iteration_master_plan(requests, exact, &exact_error)) {
            if (enable_memory_scale_up && !extras.empty() && exact_error.find("append capacity") != std::string::npos) {
                int rank = extras.front();
                extras.erase(extras.begin());
                current_allocation.push_back(rank);
                result.new_allocation_ranks.push_back(rank);
                memory_scaled = true;
                continue;
            }
            result.failure_reason = "receiver quota matching failed validation: " + exact_error;
            return result;
        }
        return exact;
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
    if (plan.master_ranks.size() != plan.master_batch_sizes.size()) {
        return fail("master rank/count telemetry size mismatch");
    }

    std::vector<int> declared_master_load(attention_sp_, 0);
    int              declared_total = 0;
    for (size_t idx = 0; idx < plan.master_ranks.size(); ++idx) {
        int rank  = plan.master_ranks[idx];
        int count = plan.master_batch_sizes[idx];
        if (rank < 0 || rank >= attention_sp_
            || std::find(plan.allocation.begin(), plan.allocation.end(), rank) == plan.allocation.end()) {
            return fail("declared master is outside group allocation");
        }
        if (count <= 0 || declared_master_load[rank] != 0) {
            return fail("declared master ranks must be unique with positive counts");
        }
        declared_master_load[rank] = count;
        declared_total += count;
    }
    if (declared_total != static_cast<int>(requests.size())) {
        return fail("declared master batch sizes do not cover request list");
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
        if (static_cast<int>(assigned[rank].size()) != declared_master_load[rank]) {
            return fail("declared master batch sizes disagree with sequence assignments");
        }
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

SPStateManager::PreparedLSIterationMasterPlan
SPStateManager::prepare_iteration_master_plan(const std::vector<std::shared_ptr<Sequence>>& requests,
                                              const LSDecodeMasterPlan&                     plan)
{
    std::unordered_set<int> allocation_ranks;
    allocation_ranks.reserve(plan.allocation.size());
    for (int rank : plan.allocation) {
        if (rank < 0 || rank >= attention_sp_ || !allocation_ranks.insert(rank).second) {
            throw std::runtime_error("prepared LS iteration plan has an invalid canonical allocation");
        }
    }

    // validate_iteration_master_plan assumes dimensionally valid ACTIVE
    // contexts. Establish those bounds first so malformed stable metadata is a
    // deterministic rejection rather than an out-of-bounds read.
    for (const auto& sequence : requests) {
        if (!sequence) {
            throw std::runtime_error("prepared LS iteration request is null");
        }
        const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
        if (context.engine_id_ != engine_id_ || context.attention_sp_ != attention_sp_
            || (dp_idx_ >= 0 && context.dp_idx_ != dp_idx_)
            || static_cast<int>(context.num_dispatched_tokens.size()) != attention_sp_
            || static_cast<int>(context.sp_block_table.size()) != attention_sp_) {
            throw std::runtime_error("prepared LS iteration ACTIVE context has invalid dimensions");
        }
    }

    std::string validation_error;
    if (!validate_iteration_master_plan(requests, plan, &validation_error)) {
        throw std::runtime_error("prepared LS iteration plan validation failed: " + validation_error);
    }

    auto impl     = std::make_unique<PreparedLSIterationMasterPlan::Impl>();
    impl->manager = this;
    impl->sequences.reserve(requests.size());
    impl->original_contexts.reserve(requests.size());
    impl->shadow_contexts.reserve(requests.size());
    impl->original_num_tokens.reserve(requests.size());
    impl->original_token_sizes.reserve(requests.size());
    impl->original_cached_tokens.reserve(requests.size());
    impl->original_last_tokens.reserve(requests.size());
    impl->original_statuses.reserve(requests.size());
    impl->rebalances.reserve(attention_sp_);
    impl->master_counts_before   = master_seq_counts_;
    impl->master_counts_after    = master_seq_counts_;
    impl->recv_counts_before     = num_recv_seqs_per_sp_;
    impl->recv_counts_after      = num_recv_seqs_per_sp_;
    impl->running_seqs_before    = num_running_seqs_;
    impl->running_tokens_before  = num_running_tokens_;

    struct ExtraBlockTarget {
        size_t sequence_idx = 0;
        int    rank         = -1;
    };
    std::vector<std::vector<int>>              releases(attention_sp_);
    std::vector<std::vector<ExtraBlockTarget>> extra_targets(attention_sp_);
    for (int rank = 0; rank < attention_sp_; ++rank) {
        releases[rank].reserve(requests.size());
        extra_targets[rank].reserve(requests.size());
    }

    std::unordered_set<const Sequence*> unique_sequences;
    std::unordered_set<uint64_t>        unique_sequence_ids;
    unique_sequences.reserve(requests.size());
    unique_sequence_ids.reserve(requests.size());

    for (size_t sequence_idx = 0; sequence_idx < requests.size(); ++sequence_idx) {
        const auto& sequence = requests[sequence_idx];
        const auto& context  = sequence->block_ctx(BlockContextSlot::ACTIVE);
        const int   target   = plan.sequence_master_ranks[sequence_idx];
        if (!unique_sequences.insert(sequence.get()).second
            || !unique_sequence_ids.insert(sequence->seq_id).second) {
            throw std::runtime_error("prepared LS iteration request list contains duplicate identity");
        }
        if (sequence->status != SequenceStatus::RUNNING || sequence->num_tokens < 0
            || sequence->token_ids.size() != static_cast<size_t>(sequence->num_tokens)
            || !context.pending_token_present_ || context.pending_token_target_sp_ < 0
            || context.pending_token_target_sp_ >= attention_sp_
            || context.master_sp_idx_ != context.pending_token_target_sp_) {
            throw std::runtime_error("prepared LS iteration request has an invalid stable frontier");
        }

        std::vector<std::pair<int, int>> table_locations;
        table_locations.reserve(context.block_location.size());
        int64_t total_dispatched = 0;
        int64_t total_final_blocks = 0;

        BlockContext shadow = context;
        shadow.master_sp_idx_           = target;
        shadow.pending_token_present_   = true;
        shadow.pending_token_target_sp_ = target;
        shadow.block_location.clear();

        for (int rank = 0; rank < attention_sp_; ++rank) {
            const int dispatched = context.num_dispatched_tokens[rank];
            if (dispatched < 0) {
                throw std::runtime_error("prepared LS iteration has a negative dispatched-token count");
            }
            total_dispatched += dispatched;
            const int committed = dispatched
                                  - (rank == context.pending_token_target_sp_ ? 1 : 0);
            if (committed < 0) {
                throw std::runtime_error("prepared LS iteration pending frontier exceeds dispatched tokens");
            }

            const int64_t historical_blocks =
                (static_cast<int64_t>(committed) + kvcache_block_size_ - 1) / kvcache_block_size_;
            const int64_t old_max_blocks =
                (static_cast<int64_t>(committed)
                 + (rank == context.pending_token_target_sp_ ? 2 : 0)
                 + kvcache_block_size_ - 1)
                / kvcache_block_size_;
            const int64_t final_blocks =
                (static_cast<int64_t>(committed) + (rank == target ? 2 : 0)
                 + kvcache_block_size_ - 1)
                / kvcache_block_size_;
            const auto& old_table = context.sp_block_table[rank];
            if (historical_blocks > static_cast<int64_t>(old_table.size())
                || static_cast<int64_t>(old_table.size()) > old_max_blocks
                || final_blocks < historical_blocks
                || final_blocks > std::numeric_limits<int>::max()) {
                throw std::runtime_error("prepared LS iteration block table is not an exact pending reservation");
            }

            std::unordered_set<int> rank_table_ids;
            rank_table_ids.reserve(old_table.size());
            for (int block_id : old_table) {
                if (block_id < 0
                    || block_id >= static_cast<int>(block_manager.at(rank)->blocks().size())
                    || block_manager.at(rank)->blocks()[block_id].ref_count <= 0
                    || !rank_table_ids.insert(block_id).second) {
                    throw std::runtime_error("prepared LS iteration block table contains invalid ownership");
                }
                table_locations.emplace_back(rank, block_id);
            }

            auto& shadow_table = shadow.sp_block_table[rank];
            shadow_table.clear();
            shadow_table.reserve(static_cast<size_t>(final_blocks));
            shadow_table.insert(shadow_table.end(),
                                old_table.begin(),
                                old_table.begin() + static_cast<std::ptrdiff_t>(historical_blocks));
            for (size_t block_idx = static_cast<size_t>(historical_blocks);
                 block_idx < old_table.size();
                 ++block_idx) {
                releases[rank].push_back(old_table[block_idx]);
            }
            for (int64_t block_idx = historical_blocks; block_idx < final_blocks; ++block_idx) {
                extra_targets[rank].push_back({sequence_idx, rank});
            }
            shadow.num_dispatched_tokens[rank] = committed + (rank == target ? 1 : 0);
            total_final_blocks += final_blocks;
        }

        auto published_locations =
            std::vector<std::pair<int, int>>(context.block_location.begin(), context.block_location.end());
        std::sort(table_locations.begin(), table_locations.end());
        std::sort(published_locations.begin(), published_locations.end());
        if (table_locations != published_locations || total_dispatched != sequence->num_tokens) {
            throw std::runtime_error("prepared LS iteration ACTIVE block metadata is inconsistent");
        }
        shadow.block_location.reserve(static_cast<size_t>(total_final_blocks));

        impl->sequences.push_back(sequence);
        impl->original_contexts.push_back(context);
        impl->shadow_contexts.push_back(std::move(shadow));
        impl->original_num_tokens.push_back(sequence->num_tokens);
        impl->original_token_sizes.push_back(sequence->token_ids.size());
        impl->original_cached_tokens.push_back(sequence->num_cached_tokens);
        impl->original_last_tokens.push_back(sequence->last_token);
        impl->original_statuses.push_back(sequence->status);

        const int old_master = context.master_sp_idx_;
        if (impl->master_counts_after[old_master] <= 0) {
            throw std::runtime_error("prepared LS iteration master counters are inconsistent");
        }
        impl->master_counts_after[old_master]--;
        impl->master_counts_after[target]++;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            const int committed = context.num_dispatched_tokens[rank]
                                  - (rank == context.pending_token_target_sp_ ? 1 : 0);
            const bool was_receiver = rank != old_master && committed > 0;
            const bool is_receiver  = rank != target && committed > 0;
            if (was_receiver && !is_receiver) {
                if (impl->recv_counts_after[rank] <= 0) {
                    throw std::runtime_error("prepared LS iteration receiver counters are inconsistent");
                }
                impl->recv_counts_after[rank]--;
            }
            else if (!was_receiver && is_receiver) {
                impl->recv_counts_after[rank]++;
            }
        }
    }

    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (impl->master_counts_after[rank] < 0 || impl->master_counts_after[rank] > max_num_seqs_
            || impl->recv_counts_after[rank] < 0 || impl->recv_counts_after[rank] > max_num_recv_seqs_) {
            throw std::runtime_error("prepared LS iteration role metadata capacity is exceeded");
        }
    }

    // All shadow/container allocation is complete. Rank-local prepare may now
    // reserve ownership; if a later rank fails, already-prepared RAII owners in
    // impl abort exactly while stable ACTIVE contexts and counters stay intact.
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (releases[rank].empty() && extra_targets[rank].empty()) {
            continue;
        }
        auto rebalance = block_manager.at(rank)->prepare_rebalance(
            releases[rank], static_cast<int>(extra_targets[rank].size()));
        const auto& allocation_ids = rebalance.allocation_block_ids();
        if (allocation_ids.size() != extra_targets[rank].size()) {
            throw std::runtime_error("prepared LS iteration rebalance returned the wrong block count");
        }
        for (size_t block_idx = 0; block_idx < allocation_ids.size(); ++block_idx) {
            const auto& target = extra_targets[rank][block_idx];
            impl->shadow_contexts[target.sequence_idx].sp_block_table[target.rank].push_back(
                allocation_ids[block_idx]);
        }
        impl->rebalances.emplace_back(rank, std::move(rebalance));
    }

    // Rebuild a complete deterministic rank-major location table only after
    // every rank table has its final prepared IDs. Capacity was reserved above.
    for (auto& shadow : impl->shadow_contexts) {
        for (int rank = 0; rank < attention_sp_; ++rank) {
            for (int block_id : shadow.sp_block_table[rank]) {
                shadow.block_location.emplace_back(rank, block_id);
            }
        }
    }


    return PreparedLSIterationMasterPlan(std::move(impl));
}

bool SPStateManager::validate_ls_pool_step_composition_noexcept(
    const PreparedLSInitialBatch* initial, const PreparedLSIterationMasterPlan* iteration) const noexcept
{
    if (!initial && !iteration) {
        return true;
    }
    if ((initial
         && (initial->state_ != PreparedLSInitialBatch::State::PREPARED || !initial->impl_
             || initial->impl_->manager != this || !initial->validate_precommit_noexcept()))
        || (iteration
            && (iteration->state_ != PreparedLSIterationMasterPlan::State::PREPARED || !iteration->impl_
                || iteration->impl_->manager != this || !iteration->validate_precommit_noexcept()))) {
        return false;
    }

    const auto& initial_master = initial ? initial->impl_->master_counts_after : master_seq_counts_;
    const auto& initial_recv   = initial ? initial->impl_->recv_counts_after : num_recv_seqs_per_sp_;
    if (initial_master.size() != master_seq_counts_.size() || initial_recv.size() != num_recv_seqs_per_sp_.size()) {
        return false;
    }

    if (iteration
        && (iteration->impl_->master_counts_before.size() != master_seq_counts_.size()
            || iteration->impl_->master_counts_after.size() != master_seq_counts_.size()
            || iteration->impl_->recv_counts_before.size() != num_recv_seqs_per_sp_.size()
            || iteration->impl_->recv_counts_after.size() != num_recv_seqs_per_sp_.size())) {
        return false;
    }

    for (size_t rank = 0; rank < master_seq_counts_.size(); ++rank) {
        int64_t final_master = initial_master[rank];
        int64_t final_recv   = initial_recv[rank];
        if (iteration) {
            final_master += static_cast<int64_t>(iteration->impl_->master_counts_after[rank])
                            - iteration->impl_->master_counts_before[rank];
            final_recv += static_cast<int64_t>(iteration->impl_->recv_counts_after[rank])
                          - iteration->impl_->recv_counts_before[rank];
        }
        if (final_master < 0 || final_master > max_num_seqs_ || final_recv < 0 || final_recv > max_num_recv_seqs_) {
            return false;
        }
    }

    const int64_t final_running_seqs   = initial ? initial->impl_->running_seqs_after : num_running_seqs_;
    const int64_t final_running_tokens = initial ? initial->impl_->running_tokens_after : num_running_tokens_;
    if (final_running_seqs < 0 || final_running_seqs > std::numeric_limits<int>::max() || final_running_tokens < 0
        || final_running_tokens > std::numeric_limits<int>::max()) {
        return false;
    }

    if (initial && iteration) {
        // noexcept validation must remain allocation-free. Pool batches are
        // bounded by max_num_seqs_, so the quadratic identity check is both
        // deterministic and small.
        for (size_t lhs = 0; lhs < initial->impl_->sequences.size(); ++lhs) {
            const auto& sequence = initial->impl_->sequences[lhs];
            if (!sequence) {
                return false;
            }
            for (size_t rhs = 0; rhs < lhs; ++rhs) {
                if (initial->impl_->sequences[rhs].get() == sequence.get()) {
                    return false;
                }
            }
        }
        for (const auto& sequence : iteration->impl_->sequences) {
            if (!sequence) {
                return false;
            }
            for (const auto& admitted : initial->impl_->sequences) {
                if (admitted.get() == sequence.get()) {
                    return false;
                }
            }
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

    plan->block_manager_lifetime_guards.reserve(retained_ranks.size() + 1);
    plan->block_manager_lifetime_guards.push_back(block_manager.at(source_rank));
    for (int rank : retained_ranks) {
        plan->block_manager_lifetime_guards.push_back(block_manager.at(rank));
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
    }

    // The source contract fills sequences in canonical group order. Rank
    // preference is global exact capacity first, then current group KV, then
    // rank ID; it must not be biased toward the current Decode master.
    std::vector<int> destination_used_tokens = group_used_kv_tokens(sequences);
    bool             found_source_kv         = false;
    for (const auto& sequence : sequences) {
        const auto& ctx = sequence->block_ctx(BlockContextSlot::ACTIVE);

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

        auto movable_capacity = [&](int rank) {
            int committed = sequence->committed_context_len(BlockContextSlot::ACTIVE, rank);
            int frontier  = ctx.pending_token_present_ && ctx.pending_token_target_sp_ == rank ? 2 : 0;
            int blocks    = static_cast<int>(ctx.sp_block_table[rank].size()) + free_blocks[rank];
            return std::max(0, blocks * kvcache_block_size_ - committed - frontier);
        };

        std::vector<int> candidates = retained_ranks;
        std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
            int lhs_capacity = movable_capacity(lhs);
            int rhs_capacity = movable_capacity(rhs);
            if (lhs_capacity != rhs_capacity) {
                return lhs_capacity > rhs_capacity;
            }
            if (destination_used_tokens[lhs] != destination_used_tokens[rhs]) {
                return destination_used_tokens[lhs] < destination_used_tokens[rhs];
            }
            return lhs < rhs;
        });

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
            destination_used_tokens[rank] += moved;
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
    std::vector<int> staged_master_counts(attention_sp_, 0);
    std::vector<int> staged_remote_recv(attention_sp_, 0);
    for (const auto& sequence : running) {
        if (!sequence || sequence->status != SequenceStatus::RUNNING) {
            continue;
        }
        const auto  staged = staged_contexts.find(sequence.get());
        const auto& ctx =
            staged == staged_contexts.end() ? sequence->block_ctx(BlockContextSlot::ACTIVE) : *staged->second;
        if (ctx.master_sp_idx_ < 0 || ctx.master_sp_idx_ >= attention_sp_) {
            return reject("KV consolidation encountered an invalid Decode master");
        }
        staged_master_counts[ctx.master_sp_idx_]++;
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

    size_t move_count = 0;
    for (const auto& draft : drafts) {
        int source_cursor = 0;
        for (const auto& assignment : draft.assignments) {
            int destination_cursor = assignment.dst_logical_start;
            int assignment_left    = assignment.num_tokens;
            while (assignment_left > 0) {
                int length = std::min({assignment_left,
                                       kvcache_block_size_ - source_cursor % kvcache_block_size_,
                                       kvcache_block_size_ - destination_cursor % kvcache_block_size_});
                source_cursor += length;
                destination_cursor += length;
                assignment_left -= length;
                ++move_count;
            }
        }
    }
    plan->moves.reserve(move_count);
    std::vector<LSKVConsolidationPlan::SequenceStage> prepared_stages;
    prepared_stages.reserve(drafts.size());

    for (auto& draft : drafts) {
        const auto stage_reservation_count = static_cast<size_t>(std::count_if(
            draft.additional_blocks.begin(), draft.additional_blocks.end(), [](int count) { return count > 0; }));
        draft.stage.destination_allocations.reserve(stage_reservation_count);
        for (int rank : retained_ranks) {
            int count = draft.additional_blocks[rank];
            if (count == 0) {
                continue;
            }
            auto& table = draft.stage.staged_context.sp_block_table[rank];
            table.reserve(table.size() + static_cast<size_t>(count));
            auto allocation = block_manager.at(rank)->prepare_allocate_uncached(count);
            const auto& block_ids = allocation.block_ids();
            table.insert(table.end(), block_ids.begin(), block_ids.end());
            draft.stage.destination_allocations.emplace_back(rank, std::move(allocation));
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
        draft.stage.source_release.emplace(
            block_manager.at(source_rank)->prepare_release(draft.stage.source_blocks));
        plan->num_tokens += draft.source_tokens;
        prepared_stages.push_back(std::move(draft.stage));
    }

    plan->sequence_stages      = std::move(prepared_stages);
    plan->master_seq_counts_before = master_seq_counts_;
    plan->master_seq_counts_after  = std::move(staged_master_counts);
    plan->recv_seq_counts_before   = num_recv_seqs_per_sp_;
    plan->recv_seq_counts_after    = std::move(staged_remote_recv);
    plan->running_seqs_before      = num_running_seqs_;
    plan->running_tokens_before    = num_running_tokens_;
    plan->success = true;
    plan->state   = LSKVConsolidationPlan::State::RESERVED;
    return plan;
}

bool SPStateManager::commit_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan) noexcept
{
    if (!plan || !plan->success || plan->state != LSKVConsolidationPlan::State::DISPATCHED
        || master_seq_counts_ != plan->master_seq_counts_before
        || num_recv_seqs_per_sp_ != plan->recv_seq_counts_before
        || num_running_seqs_ != plan->running_seqs_before
        || num_running_tokens_ != plan->running_tokens_before) {
        return false;
    }

    for (const auto& snapshot : plan->sequence_snapshots) {
        if (!snapshot.sequence || snapshot.sequence->status != snapshot.status
            || !block_context_equal(snapshot.sequence->block_ctx(BlockContextSlot::ACTIVE), snapshot.context)) {
            return false;
        }
    }
    for (const auto& stage : plan->sequence_stages) {
        if (!stage.source_release.has_value()
            || stage.source_release->state() != BlockManager::PreparedBlockMutation::State::PREPARED
            || std::any_of(stage.destination_allocations.begin(),
                           stage.destination_allocations.end(),
                           [](const auto& allocation) {
                               return allocation.mutation.state()
                                      != BlockManager::PreparedBlockMutation::State::PREPARED;
                           })) {
            return false;
        }
    }

    for (auto& stage : plan->sequence_stages) {
        using std::swap;
        swap(stage.sequence->block_ctx(BlockContextSlot::ACTIVE), stage.staged_context);
    }
    for (auto& stage : plan->sequence_stages) {
        for (auto& allocation : stage.destination_allocations) {
            allocation.mutation.commit_noexcept();
        }
        stage.source_release->commit_noexcept();
    }
    master_seq_counts_.swap(plan->master_seq_counts_after);
    num_recv_seqs_per_sp_.swap(plan->recv_seq_counts_after);
    cached_running_state_.reset();
    plan->state = LSKVConsolidationPlan::State::COMMITTED;
    return true;
}

void SPStateManager::abort_kv_consolidation(const std::shared_ptr<LSKVConsolidationPlan>& plan) noexcept
{
    if (!plan || plan->state != LSKVConsolidationPlan::State::RESERVED) {
        return;
    }
    for (auto& stage : plan->sequence_stages) {
        for (auto& allocation : stage.destination_allocations) {
            allocation.mutation.abort_noexcept();
        }
        if (stage.source_release.has_value()) {
            stage.source_release->abort_noexcept();
        }
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

SPStateManager::PreparedLSInitialBatch
SPStateManager::prepare_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch,
                                         const std::vector<BlockContext>&              placement_contexts)
{
    if (batch.empty()) {
        throw std::runtime_error("prepared LS initial batch must not be empty");
    }
    if (batch.size() != placement_contexts.size()) {
        throw std::runtime_error("prepared LS initial batch placement count mismatch");
    }

    auto impl     = std::make_unique<PreparedLSInitialBatch::Impl>();
    impl->manager = this;
    impl->sequences.reserve(batch.size());
    impl->original_contexts.reserve(batch.size());
    impl->shadow_contexts.reserve(batch.size());
    impl->original_num_tokens.reserve(batch.size());
    impl->original_token_sizes.reserve(batch.size());
    impl->original_statuses.reserve(batch.size());
    impl->allocations.reserve(attention_sp_);

    impl->master_counts_before   = master_seq_counts_;
    impl->master_counts_after    = master_seq_counts_;
    impl->recv_counts_before     = num_recv_seqs_per_sp_;
    impl->recv_counts_after      = num_recv_seqs_per_sp_;
    impl->running_seqs_before    = num_running_seqs_;
    impl->running_seqs_after     = num_running_seqs_;
    impl->running_tokens_before  = num_running_tokens_;
    impl->running_tokens_after   = num_running_tokens_;

    std::unordered_set<uint64_t> sequence_ids;
    sequence_ids.reserve(batch.size());
    std::vector<std::vector<int>> blocks_per_sequence(batch.size(), std::vector<int>(attention_sp_, 0));
    std::vector<int>              needed_blocks(attention_sp_, 0);
    int64_t                       running_token_delta = 0;

    // Finish every allocating shadow/container operation before reserving the
    // first physical block. token_ids capacity is semantically invisible and
    // makes the caller's fixed dummy append allocation-free after commit.
    for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
        const auto& sequence  = batch[seq_idx];
        const auto& placement = placement_contexts[seq_idx];
        if (!sequence || !sequence_ids.insert(sequence->seq_id).second) {
            throw std::runtime_error("prepared LS initial batch contains a null or duplicate sequence");
        }
        if (sequence->num_tokens < 0 || sequence->token_ids.size() != static_cast<size_t>(sequence->num_tokens)) {
            throw std::runtime_error("prepared LS initial sequence token metadata is inconsistent");
        }
        if (placement.engine_id_ != engine_id_ || placement.attention_sp_ != attention_sp_
            || (dp_idx_ >= 0 && placement.dp_idx_ != dp_idx_)
            || placement.master_sp_idx_ < 0 || placement.master_sp_idx_ >= attention_sp_
            || static_cast<int>(placement.num_dispatched_tokens.size()) != attention_sp_
            || static_cast<int>(placement.sp_block_table.size()) != attention_sp_
            || !placement.block_location.empty() || placement.pending_token_present_
            || placement.pending_token_target_sp_ != -1) {
            throw std::runtime_error("prepared LS initial placement context is invalid");
        }
        if (std::any_of(placement.sp_block_table.begin(), placement.sp_block_table.end(), [](const auto& table) {
                return !table.empty();
            })) {
            throw std::runtime_error("prepared LS initial placement already contains physical blocks");
        }

        int64_t dispatched = 0;
        int64_t location_capacity = 0;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            const int tokens = placement.num_dispatched_tokens[rank];
            if (tokens < 0) {
                throw std::runtime_error("prepared LS initial placement contains a negative token count");
            }
            dispatched += tokens;
            const int64_t tokens_with_dummy_headroom =
                static_cast<int64_t>(tokens) + (rank == placement.master_sp_idx_ ? 1 : 0);
            const int64_t block_count =
                (tokens_with_dummy_headroom + kvcache_block_size_ - 1) / kvcache_block_size_;
            if (block_count > std::numeric_limits<int>::max()
                || needed_blocks[rank] > std::numeric_limits<int>::max() - block_count) {
                throw std::runtime_error("prepared LS initial block demand overflows integer accounting");
            }
            blocks_per_sequence[seq_idx][rank] = static_cast<int>(block_count);
            needed_blocks[rank] += static_cast<int>(block_count);
            location_capacity += block_count;
        }
        if (dispatched != sequence->num_tokens) {
            throw std::runtime_error("prepared LS initial placement does not cover the complete prompt");
        }

        sequence->token_ids.reserve(sequence->token_ids.size() + 1);
        impl->sequences.push_back(sequence);
        impl->original_contexts.push_back(sequence->block_ctx(BlockContextSlot::ACTIVE));
        impl->original_num_tokens.push_back(sequence->num_tokens);
        impl->original_token_sizes.push_back(sequence->token_ids.size());
        impl->original_statuses.push_back(sequence->status);

        BlockContext shadow = placement;
        shadow.block_location.clear();
        shadow.block_location.reserve(static_cast<size_t>(location_capacity));
        shadow.sp_block_table.clear();
        shadow.sp_block_table.resize(attention_sp_);
        for (int rank = 0; rank < attention_sp_; ++rank) {
            shadow.sp_block_table[rank].reserve(blocks_per_sequence[seq_idx][rank]);
        }
        impl->shadow_contexts.push_back(std::move(shadow));

        impl->master_counts_after[placement.master_sp_idx_]++;
        for (int rank = 0; rank < attention_sp_; ++rank) {
            if (placement.num_dispatched_tokens[rank] > 0 && rank != placement.master_sp_idx_) {
                impl->recv_counts_after[rank]++;
            }
        }
        // The fixed dummy token is already included in the publication target
        // even though the shadow context remains prompt-only until the caller's
        // allocation-free append/mark operations.
        running_token_delta += static_cast<int64_t>(sequence->num_tokens) + 1;
    }

    if (batch.size() > static_cast<size_t>(std::numeric_limits<int>::max() - impl->running_seqs_after)
        || running_token_delta > std::numeric_limits<int>::max() - impl->running_tokens_after) {
        throw std::runtime_error("prepared LS initial running counter overflows integer accounting");
    }
    impl->running_seqs_after += static_cast<int>(batch.size());
    impl->running_tokens_after += static_cast<int>(running_token_delta);
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (impl->master_counts_after[rank] > max_num_seqs_) {
            throw std::runtime_error("prepared LS initial master metadata capacity is exceeded");
        }
        if (impl->recv_counts_after[rank] > max_num_recv_seqs_) {
            throw std::runtime_error("prepared LS initial receiver metadata capacity is exceeded");
        }
        if (block_manager.at(rank)->num_free_blocks() < needed_blocks[rank]) {
            throw std::runtime_error("prepared LS initial batch lost its exact block capacity");
        }
    }

    // Each rank gets one exact transaction-owned ID list. Sequence shadow
    // tables consume that list in stable batch order; no publication occurs.
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (needed_blocks[rank] == 0) {
            continue;
        }
        auto mutation = block_manager.at(rank)->prepare_allocate_uncached(needed_blocks[rank]);
        const auto& block_ids = mutation.block_ids();
        size_t      block_pos = 0;
        for (size_t seq_idx = 0; seq_idx < batch.size(); ++seq_idx) {
            auto& shadow = impl->shadow_contexts[seq_idx];
            for (int block_idx = 0; block_idx < blocks_per_sequence[seq_idx][rank]; ++block_idx) {
                const int block_id = block_ids[block_pos++];
                shadow.sp_block_table[rank].push_back(block_id);
                shadow.block_location.emplace_back(rank, block_id);
            }
        }
        if (block_pos != block_ids.size()) {
            std::terminate();
        }
        impl->allocations.emplace_back(rank, std::move(mutation));
    }

    return PreparedLSInitialBatch(std::move(impl));
}

SPStateManager::PreparedLSInitialBatch
SPStateManager::prepare_ls_initial_batch(const std::vector<std::shared_ptr<Sequence>>& batch)
{
    std::vector<BlockContext> placement_contexts;
    placement_contexts.reserve(batch.size());
    for (const auto& sequence : batch) {
        if (!sequence) {
            throw std::runtime_error("prepared LS initial batch contains a null sequence");
        }
        placement_contexts.push_back(sequence->block_ctx(BlockContextSlot::ACTIVE));
    }
    return prepare_ls_initial_batch(batch, placement_contexts);
}

SPStateManager::PreparedLSRelease
SPStateManager::prepare_ls_release(const std::shared_ptr<Sequence>& sequence, BlockContextSlot slot)
{
    if (slot != BlockContextSlot::ACTIVE) {
        throw std::runtime_error("formal LS prepared release only supports the ACTIVE context");
    }
    if (!sequence) {
        throw std::runtime_error("prepared LS release sequence is null");
    }
    const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
    if (context.engine_id_ != engine_id_ || context.attention_sp_ != attention_sp_
        || (dp_idx_ >= 0 && context.dp_idx_ != dp_idx_)
        || context.master_sp_idx_ < 0 || context.master_sp_idx_ >= attention_sp_
        || static_cast<int>(context.num_dispatched_tokens.size()) != attention_sp_
        || static_cast<int>(context.sp_block_table.size()) != attention_sp_) {
        throw std::runtime_error("prepared LS release ACTIVE context is invalid");
    }
    if (sequence->num_tokens < 0 || sequence->token_ids.size() != static_cast<size_t>(sequence->num_tokens)) {
        throw std::runtime_error("prepared LS release token metadata is inconsistent");
    }

    // Validate that every published table ID has exactly one matching location
    // entry before any BlockManager is reserved.
    std::vector<std::pair<int, int>> table_locations;
    table_locations.reserve(context.block_location.size());
    for (int rank = 0; rank < attention_sp_; ++rank) {
        for (int block_id : context.sp_block_table[rank]) {
            table_locations.emplace_back(rank, block_id);
        }
    }
    auto published_locations = std::vector<std::pair<int, int>>(context.block_location.begin(),
                                                                 context.block_location.end());
    std::sort(table_locations.begin(), table_locations.end());
    std::sort(published_locations.begin(), published_locations.end());
    if (table_locations != published_locations) {
        throw std::runtime_error("prepared LS release block tables and locations disagree");
    }

    auto impl                    = std::make_unique<PreparedLSRelease::Impl>();
    impl->manager                = this;
    impl->sequence               = sequence;
    impl->original_context       = context;
    impl->original_num_tokens    = sequence->num_tokens;
    impl->original_token_size    = sequence->token_ids.size();
    impl->original_cached_tokens = sequence->num_cached_tokens;
    impl->original_status        = sequence->status;
    impl->master_counts_before   = master_seq_counts_;
    impl->master_counts_after    = master_seq_counts_;
    impl->recv_counts_before     = num_recv_seqs_per_sp_;
    impl->recv_counts_after      = num_recv_seqs_per_sp_;
    impl->running_seqs_before    = num_running_seqs_;
    impl->running_seqs_after     = num_running_seqs_;
    impl->running_tokens_before  = num_running_tokens_;
    impl->running_tokens_after   = num_running_tokens_;
    impl->releases.reserve(attention_sp_);

    impl->shadow_context = BlockContext(engine_id_, attention_sp_, context.attention_dp_);
    impl->shadow_context.dp_idx_ = sequence->assigned_dp >= 0 ? sequence->assigned_dp : context.dp_idx_;
    impl->shadow_context.master_sp_idx_ = -1;

    const int master = context.master_sp_idx_;
    if (impl->master_counts_after[master] <= 0 || impl->running_seqs_after <= 0
        || impl->running_tokens_after < sequence->num_tokens) {
        throw std::runtime_error("prepared LS release logical counters are inconsistent");
    }
    impl->master_counts_after[master]--;
    for (int rank = 0; rank < attention_sp_; ++rank) {
        if (context.num_dispatched_tokens[rank] > 0 && rank != master) {
            if (impl->recv_counts_after[rank] <= 0) {
                throw std::runtime_error("prepared LS release receiver counter is inconsistent");
            }
            impl->recv_counts_after[rank]--;
        }
    }
    impl->running_seqs_after--;
    impl->running_tokens_after -= sequence->num_tokens;

    for (int rank = 0; rank < attention_sp_; ++rank) {
        const auto& block_ids = context.sp_block_table[rank];
        if (block_ids.empty()) {
            continue;
        }
        auto mutation = block_manager.at(rank)->prepare_release(
            std::vector<int>(block_ids.begin(), block_ids.end()));
        impl->releases.emplace_back(rank, std::move(mutation));
    }
    return PreparedLSRelease(std::move(impl));
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
