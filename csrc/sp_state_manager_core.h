#pragma once

#include "block_manager_core.h"

#include <algorithm>
#include <deque>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class NDSPStateManager {
public:
    static constexpr int segment_size = 1024;

    NDSPStateManager(std::optional<std::string> engine_id,
                     int                        attention_sp,
                     int                        num_kvcache_blocks,
                     int                        kvcache_block_size,
                     int                        max_num_seqs,
                     int                        max_num_batched_tokens):
        engine_id_(std::move(engine_id)),
        attention_sp_(attention_sp),
        max_num_seqs_(max_num_seqs),
        max_num_batched_tokens_(max_num_batched_tokens)
    {
        if (attention_sp_ <= 0)
            throw std::invalid_argument("attention_sp must be > 0");

        block_managers_.reserve((size_t)attention_sp_);
        for (int i = 0; i < attention_sp_; ++i) {
            block_managers_.emplace_back(engine_id_, i, num_kvcache_blocks, kvcache_block_size);
        }

        initialize_dummy_seqs();
    }

    bool is_empty() const
    {
        return running_.empty();
    }

    int attention_sp() const { return attention_sp_; }

    // Expose underlying BlockManagers for Python-side compatibility
    NDCacheBlockManager& block_manager_at(int sp_idx)
    {
        if (sp_idx < 0 || sp_idx >= attention_sp_)
            throw py::index_error("block_manager index out of range");
        return block_managers_.at((size_t)sp_idx);
    }

    const NDCacheBlockManager& block_manager_at(int sp_idx) const
    {
        if (sp_idx < 0 || sp_idx >= attention_sp_)
            throw py::index_error("block_manager index out of range");
        return block_managers_.at((size_t)sp_idx);
    }

    // ---------------------------------------------------------------------
    // Dummy seqs
    // ---------------------------------------------------------------------

    std::vector<std::shared_ptr<Sequence>> dummy_seqs() const
    {
        return dummy_seqs_;
    }

    std::shared_ptr<Sequence> dummy_seq_at(int sp_idx) const
    {
        if (sp_idx < 0 || sp_idx >= (int)dummy_seqs_.size())
            throw py::index_error("dummy_seqs index out of range");
        return dummy_seqs_[(size_t)sp_idx];
    }

    // ---------------------------------------------------------------------
    // Running queue ops (used by Python Scheduler)
    // ---------------------------------------------------------------------

    bool running_has_any() const
    {
        return !running_.empty();
    }

    size_t running_size() const
    {
        return running_.size();
    }

    void running_append(const std::shared_ptr<Sequence>& seq)
    {
        running_.push_back(seq);
    }

    std::shared_ptr<Sequence> running_popleft()
    {
        if (running_.empty())
            throw py::index_error("running queue is empty");
        auto seq = running_.front();
        running_.pop_front();
        return seq;
    }

    std::shared_ptr<Sequence> running_pop()
    {
        if (running_.empty())
            throw py::index_error("running queue is empty");
        auto seq = running_.back();
        running_.pop_back();
        return seq;
    }

    void running_extendleft(const py::list& seqs)
    {
        // Mirror: deque.extendleft(reversed(seqs))
        for (ssize_t i = (ssize_t)seqs.size() - 1; i >= 0; --i) {
            running_.push_front(seqs[i].cast<std::shared_ptr<Sequence>>());
        }
    }

    void running_remove_seq_ids(const py::set& seq_ids)
    {
        std::unordered_set<std::string> ids;
        ids.reserve((size_t)seq_ids.size());
        for (auto item : seq_ids) {
            ids.insert(item.cast<std::string>());
        }

        std::deque<std::shared_ptr<Sequence>> filtered;
        filtered.resize(0);
        for (auto& seq : running_) {
            if (ids.find(seq->seq_id) == ids.end()) {
                filtered.push_back(seq);
            }
        }
        running_.swap(filtered);
    }

    py::list running_snapshot() const
    {
        py::list L;
        for (auto& seq : running_) {
            L.append(seq);
        }
        return L;
    }

    // ---------------------------------------------------------------------
    // Cache/block manager operations
    // ---------------------------------------------------------------------

    bool can_append(const std::shared_ptr<Sequence>& seq, int num_tokens = 1)
    {
        auto& ctx = seq->block_ctx(engine_id_);
        int   master = ctx.master_sp_idx;
        return block_managers_.at((size_t)master).can_append(*seq, num_tokens);
    }

    void may_append(const std::shared_ptr<Sequence>& seq, int num_tokens = 1)
    {
        auto& ctx = seq->block_ctx(engine_id_);
        int   master = ctx.master_sp_idx;
        block_managers_.at((size_t)master).may_append(*seq, num_tokens);
    }

    bool can_allocate(const std::shared_ptr<Sequence>& seq, const py::dict& num_seqs, const py::dict& num_batched_tokens)
    {
        auto& ctx = seq->block_ctx(engine_id_);
        ctx.num_dispatched_tokens.clear();

        int64_t num_segments = (seq->num_tokens + segment_size - 1) / segment_size;
        int64_t num_segments_per_rank = (num_segments + attention_sp_ - 1) / attention_sp_;
        int64_t num_ranks = (num_segments + num_segments_per_rank - 1) / num_segments_per_rank;

        int master_rank = rr_counter_ % attention_sp_;
        rr_counter_ += 1;

        int master_seq_count = dict_get_int(num_seqs, master_rank);
        if (master_seq_count >= max_num_seqs_) {
            return false;
        }

        int master_token_count = dict_get_int(num_batched_tokens, master_rank);
        if (master_token_count + (int)seq->num_tokens >= max_num_batched_tokens_) {
            return false;
        }

        std::vector<std::pair<int, int>> rank_free_count;
        rank_free_count.reserve((size_t)attention_sp_);
        for (int rank = 0; rank < attention_sp_; ++rank) {
            if (rank == master_rank)
                continue;
            rank_free_count.push_back({rank, block_managers_[(size_t)rank].free_count()});
        }

        std::sort(rank_free_count.begin(), rank_free_count.end(), [](auto& a, auto& b) { return a.second < b.second; });

        std::vector<int> selected_ranks;
        selected_ranks.reserve((size_t)attention_sp_);
        int take = (int)std::max<int64_t>(0, num_ranks - 1);
        for (int i = 0; i < take && i < (int)rank_free_count.size(); ++i) {
            selected_ranks.push_back(rank_free_count[(size_t)i].first);
        }
        selected_ranks.push_back(master_rank);

        ctx.master_sp_idx = master_rank;

        int64_t total_token_unalloc = seq->num_tokens;
        for (int sp_idx : selected_ranks) {
            int64_t alloc = std::min<int64_t>(total_token_unalloc, num_segments_per_rank * segment_size);
            ctx.num_dispatched_tokens[sp_idx] = (int)alloc;
            total_token_unalloc -= (num_segments_per_rank * segment_size);
        }

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (!block_managers_[(size_t)sp_idx].can_allocate(*seq))
                return false;
        }
        return true;
    }

    void allocate(const std::shared_ptr<Sequence>& seq)
    {
        auto& ctx = seq->block_ctx(engine_id_);
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            if (sp_idx != ctx.master_sp_idx) {
                block_managers_[(size_t)sp_idx].allocate(*seq);
            }
        }
        block_managers_[(size_t)ctx.master_sp_idx].allocate(*seq);
    }

    std::pair<std::vector<std::shared_ptr<Sequence>>, std::vector<std::shared_ptr<Sequence>>>
    schedule_decode(int loop_count)
    {
        std::vector<std::shared_ptr<Sequence>> scheduled;
        std::vector<std::shared_ptr<Sequence>> preempted;

        while (!running_.empty()) {
            auto seq = running_.front();
            running_.pop_front();

            while (!can_append(seq, loop_count)) {
                if (!running_.empty()) {
                    auto victim = running_.back();
                    running_.pop_back();

                    deallocate(victim);
                    victim->status = SequenceStatus::WAITING;
                    victim->num_checkpointed_tokens = victim->token_ids.size();
                    preempted.push_back(victim);
                } else {
                    deallocate(seq);
                    seq->status = SequenceStatus::WAITING;
                    seq->num_checkpointed_tokens = seq->token_ids.size();
                    preempted.push_back(seq);
                    seq = nullptr;
                    break;
                }
            }

            if (seq) {
                may_append(seq, loop_count);
                scheduled.push_back(seq);
            }
        }

        for (auto it = scheduled.rbegin(); it != scheduled.rend(); ++it) {
            running_.push_front(*it);
        }

        return {scheduled, preempted};
    }

    void deallocate(const std::shared_ptr<Sequence>& seq)
    {
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            block_managers_[(size_t)sp_idx].deallocate(*seq);
        }
        auto& ctx = seq->block_ctx(engine_id_);
        ctx.sp_block_table.clear();
        ctx.block_location.clear();
        ctx.num_dispatched_tokens.clear();
    }

private:
    static int dict_get_int(const py::dict& d, int key)
    {
        py::handle k = py::int_(key);
        auto it = d.contains(k);
        if (!it)
            return 0;
        return d[k].cast<int>();
    }

    void initialize_dummy_seqs()
    {
        // Keep it simple: deterministic-ish tokens.
        std::mt19937 rng(0);
        std::uniform_int_distribution<int> dist(0, 7999);

        dummy_seqs_.clear();
        dummy_seqs_.reserve((size_t)attention_sp_);

        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            auto dummy = std::make_shared<Sequence>(std::vector<int>{dist(rng)}, py::none(), engine_id_, sp_idx);
            dummy->append_token(dist(rng), engine_id_, sp_idx);
            block_managers_[(size_t)sp_idx].allocate(*dummy);
            dummy_seqs_.push_back(dummy);
        }
    }

private:
    std::optional<std::string>          engine_id_;
    int                                 attention_sp_ = 1;
    int                                 max_num_seqs_ = 0;
    int                                 max_num_batched_tokens_ = 0;

    std::vector<NDCacheBlockManager>    block_managers_;

    std::deque<std::shared_ptr<Sequence>> running_;
    std::vector<std::shared_ptr<Sequence>> dummy_seqs_;

    int rr_counter_ = 0;
};

// Binding hook (implemented in sp_state_manager_binding.cpp)
void bind_sp_state_manager(py::module& m);
