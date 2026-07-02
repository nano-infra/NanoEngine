#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

#include "dlengine/csrc/sequence/sequence.h"
#include "sequence_generated.h"

#include "group_manager.h"

namespace dlengine {

namespace {
// Env-gated diagnostics for session-scoped GDN state caching. Set
// DLENGINE_LOG_SESSION_CACHE=1 to trace park/adopt decisions (why a warm
// session does or does not get reused) to stderr.
bool session_cache_debug()
{
    static const bool on = [] {
        const char* v = std::getenv("DLENGINE_LOG_SESSION_CACHE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

// Longest common prefix length of two token id vectors (diagnostic only).
int common_prefix_len(const std::vector<int>& a, const std::vector<int>& b)
{
    int n = static_cast<int>(std::min(a.size(), b.size()));
    int i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}
}  // namespace

GroupManager::GroupManager(const std::string& engine_id,
                           int                group_size,
                           int                num_kvcache_blocks,
                           int                kvcache_block_size,
                           int                max_num_seqs,
                           int                max_num_batched_tokens):
    gdn_state_manager_(engine_id, 0, max_num_seqs),
    hisparse_slot_manager_(engine_id, 0, max_num_seqs),
    engine_id_(engine_id),
    group_size_(group_size),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    kvcache_block_size_(kvcache_block_size),
    num_kvcache_blocks_(num_kvcache_blocks),
    num_running_seqs_per_group_(group_size, 0),
    num_running_tokens_per_group_(group_size, 0)
{
    for (int i = 0; i < group_size; ++i) {
        block_manager[i] = std::make_shared<BlockManager>(engine_id, i, num_kvcache_blocks, kvcache_block_size);
    }
    initialize_dummy_seqs();
}

void GroupManager::initialize_dummy_seqs()
{
    // Use a fixed seed for reproducibility or random device
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 7999);

    for (int group_id = 0; group_id < group_size_; ++group_id) {
        std::vector<int> token_ids = {dis(gen)};

        SamplingParams sp;
        sp.temperature = 1.0;
        sp.max_tokens  = 256;
        sp.ignore_eos  = false;

        auto dummy_seq = std::make_shared<Sequence>(token_ids, sp);
        dummy_seq->active(engine_id_, group_size_, 1, num_kvcache_blocks_);
        dummy_seq->block_ctx().master_group_id = group_id;

        dummy_seq->append_token(dis(gen), BlockContextSlot::ACTIVE, group_id);

        block_manager[group_id]->allocate(*dummy_seq);
        dummy_seqs.push_back(dummy_seq);
    }
}

int GroupManager::next_group_id()
{
    int idx           = group_rr_counter_;
    group_rr_counter_ = (group_rr_counter_ + 1) % group_size_;
    return idx;
}

bool GroupManager::can_append(Sequence& seq, int num_tokens)
{
    int master_group_id = seq.block_ctx(BlockContextSlot::ACTIVE).master_group_id;
    if (block_manager.find(master_group_id) == block_manager.end()) {
        return false;
    }
    return block_manager[master_group_id]->can_append(seq, num_tokens);
}

bool GroupManager::may_append(Sequence& seq, int num_tokens)
{
    int master_group_id = seq.block_ctx(BlockContextSlot::ACTIVE).master_group_id;
    if (block_manager.find(master_group_id) != block_manager.end()) {
        return block_manager[master_group_id]->may_append(seq, num_tokens);
    }
    return false;
}

bool GroupManager::can_allocate(Sequence&                           seq,
                                const std::unordered_map<int, int>& num_seqs,
                                const std::unordered_map<int, int>& num_batched_tokens)
{
    if (seq.block_ctx(BlockContextSlot::ACTIVE).hisparse_slot < 0 && !hisparse_slot_manager_.can_allocate()) {
        return false;
    }
    // Step 1: Determine min required ranks (Initial SP Size)
    int num_tokens           = seq.num_tokens();
    int num_segments         = (num_tokens + segment_size - 1) / segment_size;
    int initial_ranks_needed = std::max(1, std::min(group_size_, num_segments));

    // Step 2 & 3: Prepare and sort all ranks
    struct RankStatus {
        int       id;
        long long current_kv_load;
        int       current_batch_load;
        int       free_blocks;
    };

    std::vector<RankStatus> all_ranks;
    all_ranks.reserve(group_size_);

    for (int i = 0; i < group_size_; ++i) {
        long long tokens = num_running_tokens_per_group_[i];
        if (num_batched_tokens.count(i))
            tokens += num_batched_tokens.at(i);

        int seqs = num_running_seqs_per_group_[i];
        if (num_seqs.count(i))
            seqs += num_seqs.at(i);

        int free_blks = block_manager[i]->num_free_blocks();
        all_ranks.push_back({i, tokens, seqs, free_blks});
    }

    // Keep batch-first sorting strategy
    std::sort(all_ranks.begin(), all_ranks.end(), [](const RankStatus& a, const RankStatus& b) {
        if (a.current_batch_load != b.current_batch_load) {
            return a.current_batch_load < b.current_batch_load;
        }
        return a.free_blocks > b.free_blocks;
    });

    int needed_blocks = (num_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;

    // Outer loop: Adaptively increase SP Size
    for (int current_sp_size = initial_ranks_needed; current_sp_size <= group_size_; ++current_sp_size) {

        // Step 4: Select participants for current size
        std::vector<RankStatus> participants;
        std::vector<RankStatus> candidates_pool;

        participants.reserve(current_sp_size);
        candidates_pool.reserve(group_size_ - current_sp_size);

        long long total_free_blocks_capacity = 0;

        for (int i = 0; i < group_size_; ++i) {
            if (i < current_sp_size) {
                participants.push_back(all_ranks[i]);
                total_free_blocks_capacity += all_ranks[i].free_blocks;
            }
            else {
                candidates_pool.push_back(all_ranks[i]);
            }
        }

        // Step 4.5: Swap participants if capacity insufficient
        auto sort_pool_by_mem_desc = [](const RankStatus& a, const RankStatus& b) {
            return a.free_blocks > b.free_blocks;
        };
        std::sort(candidates_pool.begin(), candidates_pool.end(), sort_pool_by_mem_desc);

        bool capacity_check_passed = true;
        while (total_free_blocks_capacity < needed_blocks) {
            if (candidates_pool.empty()) {
                capacity_check_passed = false;
                break;
            }

            auto min_mem_it = std::min_element(
                participants.begin(), participants.end(), [](const RankStatus& a, const RankStatus& b) {
                    return a.free_blocks < b.free_blocks;
                });

            const auto& rich_candidate = candidates_pool.front();

            if (rich_candidate.free_blocks <= min_mem_it->free_blocks) {
                capacity_check_passed = false;
                break;
            }

            total_free_blocks_capacity -= min_mem_it->free_blocks;
            total_free_blocks_capacity += rich_candidate.free_blocks;

            *min_mem_it = rich_candidate;
            candidates_pool.erase(candidates_pool.begin());
        }

        if (!capacity_check_passed) {
            continue;  // Try next SP Size
        }

        // Step 5: Water-filling allocation
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        block_ctx.num_dispatched_tokens.assign(group_size_, 0);

        std::vector<long long> simulated_kv_loads;
        std::vector<int>       alloc_counts(participants.size(), 0);
        for (const auto& p : participants)
            simulated_kv_loads.push_back(p.current_kv_load);

        int       tokens_remaining  = num_tokens;
        const int CHUNK_SIZE        = kvcache_block_size_;
        bool      water_fill_failed = false;

        while (tokens_remaining > 0) {
            auto min_it = std::min_element(simulated_kv_loads.begin(), simulated_kv_loads.end());
            int  idx    = std::distance(simulated_kv_loads.begin(), min_it);

            int attempt_alloc        = std::min(CHUNK_SIZE, tokens_remaining);
            int rank_capacity_tokens = participants[idx].free_blocks * kvcache_block_size_;

            if (alloc_counts[idx] + attempt_alloc > rank_capacity_tokens) {
                *min_it = std::numeric_limits<long long>::max();  // Mark rank as full

                bool all_full = true;
                for (auto val : simulated_kv_loads) {
                    if (val != std::numeric_limits<long long>::max()) {
                        all_full = false;
                        break;
                    }
                }
                if (all_full) {
                    water_fill_failed = true;
                    break;
                }
                continue;
            }

            simulated_kv_loads[idx] += attempt_alloc;
            alloc_counts[idx] += attempt_alloc;
            tokens_remaining -= attempt_alloc;
        }

        if (water_fill_failed) {
            continue;  // Try larger SP Size
        }

        // Step 6 & 7: Success - write results and return

        // 1. Fill dispatch results
        for (size_t i = 0; i < participants.size(); ++i) {
            int rank_id                              = participants[i].id;
            block_ctx.num_dispatched_tokens[rank_id] = alloc_counts[i];
        }

        // 2. Select master (least loaded)
        auto min_batch_it =
            std::min_element(participants.begin(), participants.end(), [](const RankStatus& a, const RankStatus& b) {
                return a.current_batch_load < b.current_batch_load;
            });
        int master_rank           = min_batch_it->id;
        block_ctx.master_group_id = master_rank;

        // 3. Final physical check
        if (min_batch_it->current_batch_load + 1 > max_num_seqs_) {
            continue;  // Max seq limit reached, try larger SP Size
        }

        bool             physical_check_ok = true;
        std::vector<int> prefix_hints(group_size_, -1);
        for (size_t i = 0; i < participants.size(); ++i) {
            int rank_id = participants[i].id;
            int hits    = block_manager[rank_id]->can_allocate(seq);
            if (hits < 0) {
                physical_check_ok = false;
                break;
            }
            prefix_hints[rank_id] = hits;
        }

        if (physical_check_ok) {
            // *** Success! Store hints for allocate() ***
            cached_prefix_hints_ = std::move(prefix_hints);
            return true;
        }

        // Physical check failed, try larger SP Size
    }

    // All SP sizes failed
    return false;
}

std::optional<AllocResult> GroupManager::try_allocate(Sequence&                           seq,
                                                      const std::unordered_map<int, int>& num_seqs,
                                                      const std::unordered_map<int, int>& num_batched_tokens)
{
    int orig_num_tokens = seq.num_tokens();

    // Budget pre-check: skip expensive can_allocate if no tokens can be scheduled
    int budget = max_num_batched_tokens_;
    for (auto& [sp, tok] : num_batched_tokens)
        budget = std::min(budget, max_num_batched_tokens_ - tok);
    if (budget <= 0)
        return std::nullopt;

    // Fast path: a warm session from a previous turn can be adopted as a
    // chunked-prefill continuation (no recompute of the shared prefix, GDN
    // recurrent state reused in place). Only valid for the seq's *first* chunk
    // (num_checkpointed == num_prompt); for later chunks the seq is already
    // running and must not be re-adopted.
    if (session_cache_.enabled() && seq.num_checkpointed_tokens() <= seq.num_prompt_tokens()
        && seq.num_cached_tokens() == 0) {
        if (auto adopted = try_adopt_session(seq, budget)) {
            return adopted;
        }
    }

    // Set num_tokens to full prompt length for block allocation.
    // PD separation guarantees no decode traffic competes for KV cache,
    // so locking all blocks at admission eliminates mid-prefill OOM.
    int full_len = std::max(seq.num_prompt_tokens(), seq.num_checkpointed_tokens());
    seq.set_num_tokens(full_len);

    // Under KV pressure, free warm parked sessions (LRU first) to make room
    // rather than rejecting admission outright. Each evicted session returns
    // its KV blocks + GDN slot to the free pools.
    while (!can_allocate(seq, num_seqs, num_batched_tokens)) {
        auto evicted = session_cache_.evict_lru();
        if (!evicted) {
            seq.set_num_tokens(orig_num_tokens);
            return std::nullopt;
        }
        free_parked(*evicted);
    }

    // Physical block allocation (sets seq.num_cached_tokens via prefix matching)
    allocate(seq);

    // Compute chunk boundary using actual prefix cache hits.
    // BlockManager::allocate caps num_cached_tokens at num_tokens - 1,
    // so full_len - num_cached >= 1 is guaranteed (combined with budget >= 1).
    int num_cached = seq.num_cached_tokens();
    int new_tokens = std::min(budget, full_len - num_cached);
    int chunk_end  = num_cached + new_tokens;

    seq.set_num_tokens(chunk_end);

    // For chunked sequences, restrict dispatch to master SP rank
    if (chunk_end < full_len) {
        auto& block_ctx    = seq.block_ctx(BlockContextSlot::ACTIVE);
        int   master_group = block_ctx.master_group_id;
        std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);
        block_ctx.num_dispatched_tokens[master_group] = chunk_end;
    }

    return AllocResult{chunk_end, new_tokens};
}

void GroupManager::allocate(Sequence& seq)
{
    auto& block_ctx       = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_group_id = block_ctx.master_group_id;

    // Use prefix hints cached by can_allocate (if available) to skip
    // redundant hash scans inside BlockManager::allocate.
    auto hints = std::move(cached_prefix_hints_);
    cached_prefix_hints_.clear();

    auto get_hint = [&](int group_id) -> int {
        return (group_id < static_cast<int>(hints.size())) ? hints[group_id] : -1;
    };

    for (int group_id = 0; group_id < group_size_; ++group_id) {
        if (group_id != master_group_id) {
            block_manager[group_id]->allocate(seq, get_hint(group_id));
        }
    }
    block_manager[master_group_id]->allocate(seq, get_hint(master_group_id));

    hisparse_slot_manager_.allocate(seq);

    // Assign a GDN state slot (index into conv/recurrent state buffers).
    // state_manager_ is a free-list over [0, max_num_seqs_); slot max_num_seqs_
    // is the reserved dummy slot and is never allocated here.
    gdn_state_manager_.allocate(seq);

    // DSv4: reserve compressed-KV pages for each compression ratio.  The
    // reservation is sized by the sequence's max possible compressed token
    // count (= (prompt + max_new_tokens) / ratio, rounded up to page_size).
    if (!compressed_block_managers_.empty()) {
        SamplingParams sparams          = seq.sampling_params();
        int            max_total_tokens = seq.num_prompt_tokens() + sparams.max_tokens;
        for (auto& [ratio, mgr] : compressed_block_managers_) {
            int max_compressed_tokens = (max_total_tokens + ratio - 1) / ratio;
            int ps                    = mgr->page_size();
            int needed                = (max_compressed_tokens + ps - 1) / ps;
            if (needed > 0) {
                mgr->allocate(seq, needed);
            }
        }
    }

    num_running_seqs_++;
    num_running_tokens_ += seq.num_tokens();
    num_running_seqs_per_group_[master_group_id]++;
    num_running_tokens_per_group_[master_group_id] += seq.num_tokens();
}

void GroupManager::configure_compressed_pools(const CachePlan& cache_plan)
{
    compressed_block_managers_.clear();
    if (cache_plan.has_hca() && cache_plan.hca.compression_ratio > 0) {
        const auto& hca = cache_plan.hca;
        compressed_block_managers_.emplace(
            hca.compression_ratio,
            std::make_unique<CompressedBlockManager>(
                engine_id_, hca.compression_ratio, hca.num_pages, hca.page_size, hca.max_blocks_per_seq));
    }
    if (cache_plan.has_csa() && cache_plan.csa.compression_ratio > 0) {
        const auto& csa = cache_plan.csa;
        compressed_block_managers_.emplace(
            csa.compression_ratio,
            std::make_unique<CompressedBlockManager>(
                engine_id_, csa.compression_ratio, csa.num_pages, csa.page_size, csa.max_blocks_per_seq));
    }
}

void GroupManager::deallocate(Sequence& seq, BlockContextSlot slot)
{
    for (int group_id = 0; group_id < group_size_; ++group_id) {
        block_manager[group_id]->deallocate(seq, slot);
    }

    hisparse_slot_manager_.deallocate(seq, slot);

    // Free the GDN state slot so it can be reused by future sequences.
    gdn_state_manager_.deallocate(seq, slot);

    // DSv4: return all compressed pages owned by this seq to their pools.
    for (auto& [ratio, mgr] : compressed_block_managers_) {
        mgr->deallocate(seq, slot);
    }

    auto& block_ctx       = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_group_id = block_ctx.master_group_id;
    block_ctx.group_block_table.clear();
    block_ctx.block_location.clear();
    std::fill(block_ctx.num_dispatched_tokens.begin(), block_ctx.num_dispatched_tokens.end(), 0);

    num_running_seqs_--;
    num_running_tokens_ -= seq.num_tokens();
    num_running_seqs_per_group_[master_group_id]--;
    num_running_tokens_per_group_[master_group_id] -= seq.num_tokens();
}

// --- Session-scoped GatedDeltaNet state caching ---------------------------

void GroupManager::set_session_cache_slots(int capacity)
{
    if (capacity < 0) {
        capacity = 0;
    }
    session_cache_.set_capacity(capacity);

    // Resize the GDN state-slot free list to cover the extra parked slots.
    // The GPU pool's active region is [0, max_num_seqs_ + capacity); the
    // dummy/backup slots live above it and are never handed out here. Safe to
    // rebuild because this runs once at init, before any allocate().
    gdn_state_manager_.reset(max_num_seqs_ + capacity);
}

void GroupManager::free_parked(const ParkedSession& entry)
{
    if (!entry.group_block_tables.empty()) {
        // group_size_ == 1 when caching is enabled, but free every recorded
        // group defensively in case that invariant ever relaxes.
        for (int gid = 0; gid < static_cast<int>(entry.group_block_tables.size()); ++gid) {
            auto it = block_manager.find(gid);
            if (it != block_manager.end() && it->second) {
                it->second->release_block_ids(entry.group_block_tables[gid]);
            }
        }
    }
    gdn_state_manager_.free_slot(entry.state_slot);
}

std::optional<AllocResult> GroupManager::try_adopt_session(Sequence& seq, int budget)
{
    if (!session_cache_.enabled() || group_size_ != 1) {
        return std::nullopt;
    }
    uint64_t key = seq.affinity_key();
    if (key == 0) {
        return std::nullopt;
    }

    const ParkedSession* e = session_cache_.peek(key);
    if (e == nullptr) {
        if (session_cache_debug()) {
            std::fprintf(stderr,
                         "[session-cache] MISS seq_id=%llu key=%llu (no parked session) parked=%d\n",
                         (unsigned long long)seq.seq_id(),
                         (unsigned long long)key,
                         session_cache_.size());
        }
        return std::nullopt;
    }

    const int full_len = std::max(seq.num_prompt_tokens(), seq.num_checkpointed_tokens());

    // The parked GDN slot holds the recurrent state for exactly e->length
    // tokens, so num_cached_tokens MUST equal e->length for the continuation to
    // be correct. We therefore need at least one *new* token (full_len >
    // e->length); a same-or-shorter prompt can't reuse this state — drop it.
    const auto& toks      = seq.token_ids();
    bool        prefix_ok = e->length > 0 && e->length < full_len && static_cast<int>(toks.size()) >= e->length
                     && std::equal(e->token_ids.begin(), e->token_ids.end(), toks.begin());
    if (!prefix_ok) {
        if (session_cache_debug()) {
            int lcp = common_prefix_len(e->token_ids, toks);
            std::fprintf(stderr,
                         "[session-cache] REJECT seq_id=%llu key=%llu parked_len=%d full_len=%d "
                         "common_prefix=%d (need exact prefix of length parked_len, and full_len > "
                         "parked_len). Dropping stale entry.\n",
                         (unsigned long long)seq.seq_id(),
                         (unsigned long long)key,
                         e->length,
                         full_len,
                         lcp);
        }
        if (auto dead = session_cache_.take(key)) {
            free_parked(*dead);
        }
        return std::nullopt;
    }

    // Capacity check: blocks needed to grow group-0's table from the cached
    // prefix up to the full prompt (all blocks locked at admission).
    const int have_blocks = (e->length + kvcache_block_size_ - 1) / kvcache_block_size_;
    const int need_blocks = (full_len + kvcache_block_size_ - 1) / kvcache_block_size_;
    const int extra       = std::max(0, need_blocks - have_blocks);
    if (block_manager[0]->num_free_blocks() < extra) {
        if (session_cache_debug()) {
            std::fprintf(stderr,
                         "[session-cache] DEFER seq_id=%llu key=%llu need_extra_blocks=%d free=%d "
                         "(leaving parked)\n",
                         (unsigned long long)seq.seq_id(),
                         (unsigned long long)key,
                         extra,
                         block_manager[0]->num_free_blocks());
        }
        // Leave the entry parked; the cold path (or a later retry) handles it.
        return std::nullopt;
    }

    // Commit: take ownership of the parked resources.
    ParkedSession ent    = std::move(*session_cache_.take(key));
    const int     master = 0;  // group_size_ == 1

    auto& bctx           = seq.block_ctx(BlockContextSlot::ACTIVE);
    bctx.master_group_id = master;

    auto& table = seq.block_table(BlockContextSlot::ACTIVE, master);
    table       = ent.group_block_tables.empty() ? std::vector<int>{} : ent.group_block_tables[0];

    bctx.block_location.clear();
    bctx.block_location.reserve(table.size());
    for (int bid : table) {
        bctx.block_location.emplace_back(master, bid);
    }

    if (static_cast<int>(bctx.num_dispatched_tokens.size()) < group_size_) {
        bctx.num_dispatched_tokens.assign(group_size_, 0);
    }
    bctx.num_dispatched_tokens[master] = ent.length;  // cached extent (for may_append)

    seq.set_state_slot(BlockContextSlot::ACTIVE, ent.state_slot);

    // Grow the block table to cover the full prompt (lock all blocks now).
    block_manager[master]->may_append(seq, full_len - ent.length);

    // Continuation cursor: the prefix is already resident, so the forward only
    // recomputes [ent.length, chunk_end).
    seq.set_num_tokens(full_len);
    seq.set_num_cached_tokens(ent.length);

    // Update running counters (mirror allocate(): counts the full prompt).
    num_running_seqs_++;
    num_running_tokens_ += full_len;
    num_running_seqs_per_group_[master]++;
    num_running_tokens_per_group_[master] += full_len;

    // Compute this chunk's boundary from the per-step budget.
    int new_tokens = std::min(budget, full_len - ent.length);
    int chunk_end  = ent.length + new_tokens;
    seq.set_num_tokens(chunk_end);
    bctx.num_dispatched_tokens[master] = chunk_end;

    if (session_cache_debug()) {
        std::fprintf(stderr,
                     "[session-cache] ADOPT seq_id=%llu key=%llu reused=%d/%d tokens (%.1f%%) "
                     "chunk_end=%d slot=%d\n",
                     (unsigned long long)seq.seq_id(),
                     (unsigned long long)key,
                     ent.length,
                     full_len,
                     full_len > 0 ? (100.0 * ent.length / full_len) : 0.0,
                     chunk_end,
                     ent.state_slot);
    }

    return AllocResult{chunk_end, new_tokens};
}

void GroupManager::park_or_deallocate(Sequence& seq)
{
    auto& bctx = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   slot = seq.state_slot(BlockContextSlot::ACTIVE);
    int   len  = seq.num_tokens();

    auto& table0 = seq.block_table(BlockContextSlot::ACTIVE, 0);

    // Eligibility: caching on, single-SP, has a session key, a GDN slot and a
    // non-empty block table. Anything else falls back to a plain free.
    bool eligible = session_cache_.enabled() && group_size_ == 1 && seq.affinity_key() != 0 && slot >= 0 && len > 0
                    && !table0.empty();
    if (!eligible) {
        deallocate(seq);
        return;
    }

    ParkedSession ent;
    ent.affinity_key    = seq.affinity_key();
    ent.state_slot      = slot;
    ent.master_group_id = bctx.master_group_id;

    const auto& toks     = seq.token_ids();
    int         copy_len = std::min<int>(len, static_cast<int>(toks.size()));
    ent.length           = copy_len;
    ent.token_ids.assign(toks.begin(), toks.begin() + copy_len);
    ent.group_block_tables.resize(1);
    ent.group_block_tables[0] = table0;  // copy the retained block ids

    // Detach the resources from the sequence WITHOUT freeing them (ownership
    // moves to the parked entry). Clearing the table prevents a later
    // deallocate() on this seq object from double-freeing the parked blocks.
    table0.clear();
    bctx.block_location.clear();
    std::fill(bctx.num_dispatched_tokens.begin(), bctx.num_dispatched_tokens.end(), 0);
    seq.set_state_slot(BlockContextSlot::ACTIVE, -1);
    seq.set_num_cached_tokens(0);

    // DSv4 compressed pages aren't part of the parked state — return them.
    for (auto& [ratio, mgr] : compressed_block_managers_) {
        mgr->deallocate(seq, BlockContextSlot::ACTIVE);
    }

    // The sequence leaves the running set (counters mirror deallocate()).
    num_running_seqs_--;
    num_running_tokens_ -= len;
    num_running_seqs_per_group_[ent.master_group_id]--;
    num_running_tokens_per_group_[ent.master_group_id] -= len;

    if (session_cache_debug()) {
        std::fprintf(stderr,
                     "[session-cache] PARK seq_id=%llu key=%llu len=%d slot=%d blocks=%d (warm=%d)\n",
                     (unsigned long long)seq.seq_id(),
                     (unsigned long long)ent.affinity_key,
                     ent.length,
                     ent.state_slot,
                     static_cast<int>(ent.group_block_tables[0].size()),
                     session_cache_.size() + 1);
    }

    // Park; free any entry it displaces (same-key replacement or LRU eviction).
    auto displaced = session_cache_.put(std::move(ent));
    for (auto& d : displaced) {
        free_parked(d);
    }
}

}  // namespace dlengine
