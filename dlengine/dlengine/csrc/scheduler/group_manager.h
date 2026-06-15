#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dlengine/csrc/cache/block_manager.h"
#include "dlengine/csrc/cache/compressed_block_manager.h"
#include "dlengine/csrc/cache/gdn_state_manager.h"
#include "dlengine/csrc/sequence/sequence.h"

#include "session_state_cache.h"

namespace dlengine {

enum class RoutingStrategy {
    RoundRobin,
    LeastBatch,
    LeastCache,
    // Session/prefix-aware: route a sequence to the DP rank that already holds
    // the longest cached prefix of its prompt (and, when available, the rank
    // that served the same client session), so multi-turn conversations keep
    // hitting a warm prefix cache. Falls back to least-cache load balancing.
    SessionPrefix
};

struct AllocResult {
    int chunk_end;   // num_tokens boundary for the current batch
    int new_tokens;  // budget consumed (= chunk_end - num_cached_tokens)
};

// DSv4: configuration for one compression ratio's page pool.
struct CompressedPoolConfig {
    int ratio;               // e.g. 4 or 128
    int num_pages;           // pool size (number of pages)
    int page_size;           // tokens per page (e.g. 2 for 16-byte alignment)
    int max_blocks_per_seq;  // hard cap per sequence
};

class GroupManager {
public:
    static constexpr int segment_size = 256;

    GroupManager(const std::string& engine_id,
                 int                group_size,
                 int                num_kvcache_blocks,
                 int                kvcache_block_size,
                 int                max_num_seqs,
                 int                max_num_batched_tokens);

    // State queries
    bool is_empty() const
    {
        return running.empty();
    }

    // Block management delegation
    bool can_append(Sequence& seq, int num_tokens = 1);
    bool may_append(Sequence& seq, int num_tokens = 1);

    // Allocation logic
    // num_seqs and num_batched_tokens are maps from dp_idx to count/tokens
    // But wait, in Python:
    // num_seqs: dict[int, int] -> maps master_group to count?
    // Let's check Python code:
    // num_seqs[selected_dp_idx][block_ctx.master_group_id] += 1
    // So passed to can_allocate is num_seqs[selected_dp_idx], which is dict[int, int] (group_id -> count)
    bool can_allocate(Sequence&                           seq,
                      const std::unordered_map<int, int>& num_seqs,
                      const std::unordered_map<int, int>& num_batched_tokens);

    void allocate(Sequence& seq);
    void deallocate(Sequence& seq, BlockContextSlot slot = BlockContextSlot::ACTIVE);

    // Session-scoped GatedDeltaNet state caching ----------------------------
    //
    // When a finished sequence carries a session affinity key, PARK its KV
    // blocks + GDN recurrent-state slot keyed by the session instead of freeing
    // them (park_or_deallocate). The next turn from the same session is then
    // served as a chunked-prefill continuation that skips recomputing the
    // shared prefix (try_adopt_session). Only active when:
    //   * session caching is enabled (capacity > 0), AND
    //   * group_size == 1 (single-SP; multi-SP block layout is not parked), AND
    //   * the sequence has a non-zero affinity_key.
    // On any other path it degrades to a plain deallocate / cold allocate.

    // Set the number of warm sessions to retain (0 disables). Re-sizes the GDN
    // state-slot free list to (max_num_seqs + capacity). Call once at init,
    // before any allocate(), mirroring set_prefix_caching_enabled().
    void set_session_cache_slots(int capacity);

    int num_parked_sessions() const
    {
        return session_cache_.size();
    }

    // Finish hook: park the sequence if eligible, otherwise deallocate().
    void park_or_deallocate(Sequence& seq);

    // Admission hook: if a warm session matches ``seq`` (same affinity key and
    // its prompt extends the parked context), adopt the parked blocks + slot
    // and return the chunk boundary; the caller skips the cold allocate path.
    // ``budget`` is the per-step token budget for this chunk. Returns nullopt
    // when there is no usable warm session (caller falls back to try_allocate's
    // cold path).
    std::optional<AllocResult> try_adopt_session(Sequence& seq, int budget);

    // Atomic budget-check + full-prompt allocation + chunk computation.
    // Internally: saves/restores num_tokens, sets full_len for block allocation,
    // computes chunk boundary from prefix hits + budget, restricts dispatch for
    // chunked sequences to master group.
    // On success: blocks allocated for full prompt, num_tokens = chunk_end,
    //             returns {chunk_end, new_tokens}.
    // On failure: num_tokens restored, no side effects, returns nullopt.
    std::optional<AllocResult> try_allocate(Sequence&                           seq,
                                            const std::unordered_map<int, int>& num_seqs,
                                            const std::unordered_map<int, int>& num_batched_tokens);

    // Load tracking
    /// \brief Returns the total number of sequences currently running on this engine.
    ///
    /// This aggregates the number of active sequences across all block group
    /// (block group) partitions managed by this GroupManager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs() const
    {
        return num_running_seqs_;
    }

    /// \brief Returns the total number of tokens currently being processed.
    ///
    /// The returned value is the sum of running tokens across all running
    /// sequences and all SP partitions in this manager.
    ///
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens() const
    {
        return num_running_tokens_;
    }

    /// \brief Returns the number of running sequences assigned to a given group index.
    ///
    /// \param group_id The zero-based block group index for which to query
    ///               the number of running sequences.
    /// \return The number of currently running sequences mapped to \p group_id.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_seqs_per_group(int group_id) const
    {
        return num_running_seqs_per_group_[group_id];
    }

    /// \brief Returns the number of running tokens assigned to a given group index.
    ///
    /// \param group_id The zero-based block group index for which to query
    ///               the number of running tokens.
    /// \return The number of tokens currently being processed on \p group_id.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    int num_running_tokens_per_group(int group_id) const
    {
        return num_running_tokens_per_group_[group_id];
    }

    // WARNING: This method modifies shared state without thread safety protection.
    // If called concurrently from multiple threads (e.g., in worker_func),
    // this will cause race conditions on the counters.

    /// \brief Adjusts the number of running tokens for a given group index.
    ///
    /// This updates both the global running-token count and the per-group running
    /// token count for the specified \p group_id.
    ///
    /// \param group_id The zero-based block group index whose token count
    ///               should be updated.
    /// \param count  The number of tokens to add. Implementations may pass a
    ///               negative value to decrement the counters when tokens are
    ///               completed or removed.
    ///
    /// \warning No bounds checking is performed on \p group_id; callers must ensure
    ///          that it is within the valid range of group indices for this engine.
    /// \note This class does not provide internal synchronization. Callers must
    ///       ensure external synchronization if accessed from multiple threads.
    void add_running_tokens(int group_id, int count)
    {
        num_running_tokens_ += count;
        num_running_tokens_per_group_[group_id] += count;
    }

    std::unordered_map<int, std::shared_ptr<BlockManager>> block_manager;

    // Cache-aware routing probe: how many leading full blocks of ``seq`` are
    // already warm on this DP rank. Block hashes are token-derived and thus
    // identical across SP groups, so the representative group 0 is sufficient.
    int matched_prefix_blocks(Sequence& seq) const
    {
        auto it = block_manager.find(0);
        if (it == block_manager.end() || !it->second) {
            return 0;
        }
        return it->second->matched_prefix_blocks(seq);
    }

    GDNStateManager gdn_state_manager_;

    // DSv4: per-compression-ratio paged allocator for compressed KV cache.
    // Empty when the model has no compressed layers (Qwen3.5, V3, V3.2).
    // Initialized via configure_compressed_pools(...) after construction.
    // Stored as unique_ptr so map<int, V> doesn't need V to be default-ctible.
    std::unordered_map<int, std::unique_ptr<CompressedBlockManager>> compressed_block_managers_;

    // Configure DSv4 compressed pools.  Replaces any previous configuration.
    // Each entry creates a new CompressedBlockManager for its `ratio`.
    void configure_compressed_pools(const std::vector<CompressedPoolConfig>& configs);

    // Toggle cross-request prefix caching on every block manager in this group
    // (disabled for linear-attention models; see BlockManager header).
    void set_prefix_caching_enabled(bool enabled)
    {
        for (auto& [gid, bm] : block_manager) {
            bm->set_prefix_caching_enabled(enabled);
        }
    }

    std::deque<std::shared_ptr<Sequence>>  running;
    std::vector<std::shared_ptr<Sequence>> dummy_seqs;

    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    void initialize_dummy_seqs();
    int  next_group_id();  // Round-robin counter

    // Free the KV blocks + GDN slot retained by a parked session that is being
    // evicted/replaced (resources are exclusively owned by the entry).
    void free_parked(const ParkedSession& entry);

    // LRU store of warm-session resources (empty / inert when capacity == 0).
    SessionStateCache session_cache_;

    std::string engine_id_;
    int         group_size_;
    int         max_num_seqs_;
    int         max_num_batched_tokens_;

    int kvcache_block_size_;
    int num_kvcache_blocks_;

    int              group_rr_counter_   = 0;
    int              num_running_seqs_   = 0;
    int              num_running_tokens_ = 0;
    std::vector<int> num_running_seqs_per_group_;
    std::vector<int> num_running_tokens_per_group_;

    // Per-group prefix hit counts cached between can_allocate() and allocate().
    // Populated by can_allocate on success; consumed (moved) by allocate.
    std::vector<int> cached_prefix_hints_;
};

}  // namespace dlengine
