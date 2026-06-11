#pragma once

#include <deque>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dlengine/csrc/sequence/sequence.h"

#include "group_manager.h"
#include "thread_pool.h"

namespace dlengine {

// Forward declaration
class MetricsManager;

// Type alias for migration list: vector<pair<sequence pointer, target DP index>>
using MigrationList = std::vector<std::pair<std::shared_ptr<Sequence>, int>>;

// Result of postprocessing sequences after model execution
struct PostprocessResult {
    MigrationList                          migrations;
    std::vector<std::shared_ptr<Sequence>> continuations;  // non-final prefill chunks
};

// Result of a single scheduling step.
// This struct is returned by `schedule()` and summarizes which sequences
// should be executed on each data-parallel (DP) worker (and, if applicable,
// on each group shard) for the current iteration.
struct ScheduleResult {
    // Sequences scheduled per DP worker for this step.
    // Outer index: DP worker index.
    // Inner vector: sequences assigned to that DP worker.
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_seqs;

    // Sequences laid out per (DP, SP) shard for this step.
    // Outer index: DP worker index.
    // Inner vector: sequences assigned to that DP worker after applying
    // group partitioning / layout.
    std::vector<std::vector<std::shared_ptr<Sequence>>> dp_group_seqs;

    // Filtered subset of `dp_group_seqs` that will actually be executed in this
    // iteration (for example, after removing finished / paused sequences or
    // enforcing per-step limits on tokens or sequences).
    // Same indexing convention as `dp_group_seqs`.
    std::vector<std::vector<std::shared_ptr<Sequence>>> filtered_dp_group_seqs;

    // Indicates whether this scheduling step is a prefill step (true) or a
    // decode step (false). Callers can use this to select the appropriate
    // execution path.
    bool is_prefill;

    // Group counts
    std::vector<std::vector<int>> group_send_counts;
    std::vector<std::vector<int>> group_recv_counts;

    // Matrix of Group communication counts.
    // Dimensions: [dp_idx][master_group][participant_group]
    // Value: Number of requests sent from master_group to participant_group.
    // std::vector<std::vector<std::vector<int>>> group_comm_matrix;

    // Matrix of Q communication counts (Master -> Participant).
    // Dimensions: [dp_idx][master_group][participant_group]
    // Value: Number of Q requests sent from master_group to participant_group.
    std::vector<std::vector<std::vector<int>>> group_q_matrix;

    // Matrix of Res communication counts (Participant -> Master).
    // Dimensions: [dp_idx][participant_group][master_group]
    // Value: Number of Res requests sent from participant_group to master_group.
    // std::vector<std::vector<std::vector<int>>> group_res_matrix;

    // Metrics for waiting queue blocks
    int waiting_head_blocks  = 0;
    int waiting_total_blocks = 0;
};

class Scheduler {
public:
    Scheduler(const std::string& engine_id,
              int                num_speculative_tokens,
              int                max_num_seqs,
              int                max_num_batched_tokens,
              int                max_model_len,
              std::vector<int>   eos_ids,
              int                attention_dp,
              int                group_size,
              int                num_kvcache_blocks,
              int                kvcache_block_size,
              const std::string& mode);

    // Queue management
    void add(std::shared_ptr<Sequence> seq);

    // DSv4: configure per-compression-ratio compressed-cache pools across
    // all DP workers.  Must be called once after construction (before any
    // allocate()) when the model has compressed layers.
    void configure_compressed_pools(const std::vector<CompressedPoolConfig>& configs);

    // Main scheduling functions
    ScheduleResult schedule();

    // Postprocessing
    void postprocess(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_group_seqs,
                     const std::vector<std::vector<std::vector<int>>>&          dp_group_token_ids,
                     bool                                                       update_metrics = true,
                     // Per-token logprobs aligned with ``dp_group_token_ids``. When
                     // empty/non-shaped to match, the scheduler skips the
                     // append_token logprob arg (sequences keep empty
                     // ``completion_logprobs``). The shape is
                     // [num_dp_groups][batch_size][per_seq_tokens].
                     const std::vector<std::vector<std::vector<float>>>& dp_group_token_logprobs = {});

    // State queries
    bool is_finished() const;

    // Preemption
    void preempt(int dp_idx, std::shared_ptr<Sequence> seq);

    // Migration management
    void free_to_be_migrated(std::shared_ptr<Sequence> seq);
    void free_to_be_migrated(const std::vector<std::shared_ptr<Sequence>>& seqs);

    // Access to running sequences
    std::deque<std::shared_ptr<Sequence>>&       running(int dp_idx);
    const std::deque<std::shared_ptr<Sequence>>& running(int dp_idx) const;

    // Access to block managers
    std::unordered_map<int, std::shared_ptr<BlockManager>>&       block_manager(int dp_idx);
    const std::unordered_map<int, std::shared_ptr<BlockManager>>& block_manager(int dp_idx) const;

    // Public members exposed to Python
    std::string engine_id_;

    // Number of MTP draft tokens (0 = MTP disabled). Used to size per-step KV
    // reservation: a decode step may append the base token plus up to this many
    // speculative tokens (plus a 1-token margin).
    int num_speculative_tokens_;
    // Tokens to reserve KV space for per scheduling step (1 for plain decode,
    // num_speculative_tokens_ + 2 when MTP is enabled).
    int                     kv_reserve_tokens_;
    int                     max_num_seqs_;
    int                     max_num_batched_tokens_;
    std::unordered_set<int> eos_ids_;

    int attention_dp_;
    int group_size_;

    std::deque<std::shared_ptr<Sequence>>                              waiting;
    std::deque<std::shared_ptr<Sequence>>                              waiting_migration;
    std::deque<std::shared_ptr<Sequence>>                              prefilling;  // mid-prompt sequences
    std::vector<std::shared_ptr<GroupManager>>                         worker_state;
    std::unordered_map<int, std::pair<std::shared_ptr<Sequence>, int>> to_be_migrated;

    // Configuration
    RoutingStrategy routing_strategy = RoutingStrategy::RoundRobin;

private:
    // Internal scheduling logic
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_prefill();
    std::vector<std::vector<std::shared_ptr<Sequence>>> _schedule_decode();

    // Round-robin counter for DP
    int next_dp_idx();

    // Postprocessing internal types
    struct PostprocessTask {
        std::shared_ptr<Sequence> seq;
        const std::vector<int>*   tokens;
        int                       group_id;
        // Optional: per-token logprobs aligned with ``*tokens``. nullptr
        // when the rollout didn't request them; otherwise size matches.
        const std::vector<float>* logprobs = nullptr;
    };

    struct PostprocessWorkerContext {
        std::vector<PostprocessTask>           tasks;
        MigrationList                          migration_candidates;
        std::vector<std::shared_ptr<Sequence>> chunk_continuations;
        std::exception_ptr                     eptr = nullptr;
        int                                    dp_idx;
        void                                   reserve(size_t n)
        {
            tasks.reserve(n);
        }
    };

    // Postprocessing internal methods
    static void postprocess_worker_func(std::shared_ptr<GroupManager>   state_manager,
                                        const PostprocessWorkerContext* ctx,
                                        PostprocessWorkerContext*       result_ctx,
                                        const std::unordered_set<int>&  eos_ids,
                                        bool                            is_prefill,
                                        bool                            update_metrics);

    PostprocessResult
    postprocess_sequences_impl(const std::vector<std::vector<std::shared_ptr<Sequence>>>& dp_group_seqs,
                               const std::vector<std::vector<std::vector<int>>>&          dp_group_token_ids,
                               bool                                                       is_prefill,
                               bool                                                       update_metrics,
                               const std::vector<std::vector<std::vector<float>>>&        dp_group_token_logprobs = {});

    // Configuration
    int max_model_len_;
    int num_kvcache_blocks_;
    int kvcache_block_size_;

    std::string mode_;

    int dp_rr_counter_ = 0;

    std::unique_ptr<ThreadPool> thread_pool_;
};

}  // namespace dlengine
