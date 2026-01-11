#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>

#include "nanodeploy/sequence/sequence.h"

#include "sp_state_manager.h"

namespace nanodeploy {

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
                               bool               enable_non_uniform_split,
                               const std::string& sp_master_selector,
                               const std::string& sp_size_mode,
                               float              initial_avg_prompt_length,
                               float              initial_avg_output_length,
                               int                stats_window_size) :
    engine_id_(engine_id),
    attention_sp_(attention_sp),
    max_num_seqs_(max_num_seqs),
    max_num_batched_tokens_(max_num_batched_tokens),
    max_num_recv_seqs_(max_num_recv_seqs),
    reserved_blocks_per_req_(reserved_blocks_per_req),
    kvcache_block_size_(kvcache_block_size),
    segment_size_(segment_size),
    num_recv_seqs_per_sp_(attention_sp, 0),
    enable_dynamic_sp_size_(enable_dynamic_sp_size),
    enable_non_uniform_split_(enable_non_uniform_split),
    num_kvcache_blocks_(num_kvcache_blocks),
    load_stats_(attention_sp, initial_avg_prompt_length, initial_avg_output_length, stats_window_size)
{
    // Initialize SP Master Selector Strategy (kept for backward compatibility)
    if (sp_master_selector == "LeastBatch") {
        master_selector_ = SPMasterSelector::LeastBatch;
    } else if (sp_master_selector == "LeastCache") {
        master_selector_ = SPMasterSelector::LeastCache;
    } else {
        master_selector_ = SPMasterSelector::RoundRobin;
    }

    // Initialize SP Size Policy
    SPSizeConfig sp_config;
    if (sp_size_mode == "load_aware") {
        sp_config.mode = SPSizeMode::LoadAware;
    } else {
        sp_config.mode = SPSizeMode::Segment;
    }
    sp_config.initial_avg_prompt_length = initial_avg_prompt_length;
    sp_config.initial_avg_output_length = initial_avg_output_length;
    sp_config.stats_window_size         = stats_window_size;
    sp_config.segment_size              = segment_size;
    sp_size_policy_ = SPSizePolicy(sp_config);
    
    // Log SP Size Policy configuration
    std::cout << "[SPStateManager] SP Size Policy initialized:" << std::endl;
    std::cout << "  - sp_size_mode: " << sp_size_mode 
              << " (" << (sp_config.mode == SPSizeMode::LoadAware ? "LoadAware" : "Segment") << ")" << std::endl;
    std::cout << "  - enable_dynamic_sp_size: " << (enable_dynamic_sp_size_ ? "true" : "false") << std::endl;
    std::cout << "  - enable_non_uniform_split: " << (enable_non_uniform_split_ ? "true" : "false") << std::endl;
    if (sp_config.mode == SPSizeMode::LoadAware) {
        std::cout << "  - initial_avg_prompt_length: " << initial_avg_prompt_length << std::endl;
        std::cout << "  - initial_avg_output_length: " << initial_avg_output_length << std::endl;
        std::cout << "  - stats_window_size: " << stats_window_size << std::endl;
    } else {
        std::cout << "  - segment_size: " << segment_size << std::endl;
    }

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
              << ", sp_size_mode=" << sp_size_mode << std::endl;

    if (attention_sp_ <= 0) {
        throw std::runtime_error("attention_sp must be positive to prevent division by zero");
    }
    if (kvcache_block_size_ <= 0) {
        throw std::runtime_error("kvcache_block_size must be positive to prevent division by zero");
    }
}

// === Helper Methods ===

std::vector<int> SPStateManager::get_free_blocks_per_rank() const
{
    std::vector<int> result;
    result.reserve(attention_sp_);
    for (int i = 0; i < attention_sp_; ++i) {
        result.push_back(block_manager.at(i)->num_free_blocks());
    }
    return result;
}

std::vector<int> SPStateManager::get_used_blocks_per_rank() const
{
    std::vector<int> result;
    result.reserve(attention_sp_);
    for (int i = 0; i < attention_sp_; ++i) {
        int free = block_manager.at(i)->num_free_blocks();
        result.push_back(num_kvcache_blocks_ - free);
    }
    return result;
}

std::vector<int> SPStateManager::get_batch_size_per_rank() const
{
    return master_seq_counts_;
}

void SPStateManager::record_waiting_queue_size(int queue_size)
{
    load_stats_.record_waiting_queue_size(queue_size);
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

bool SPStateManager::can_allocate(Sequence&                           seq,
                                  const std::unordered_map<int, int>& num_seqs,
                                  const std::unordered_map<int, int>& num_batched_tokens)
{
    // Record arrival for statistics
    load_stats_.record_arrival();
    
    // =========================================================================
    // Real-time Workload Scan: Identify short request BS and long request load
    // =========================================================================
    int short_seq_count = 0;
    std::vector<int> long_used_per_rank(attention_sp_, 0);

    for (const auto& s : running) {
        if (load_stats_.is_long_request(s->num_prompt_tokens)) {
            const auto& ctx = s->block_ctx(BlockContextSlot::ACTIVE);
            for (int i = 0; i < attention_sp_; ++i) {
                int tokens = ctx.num_dispatched_tokens[i];
                int blocks = (tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
                long_used_per_rank[i] += blocks;
            }
        } else {
            short_seq_count++;
        }
    }
    // Update load statistics with real-time short request batch size
    load_stats_.record_short_batch_size(short_seq_count);

    if (attention_sp_ > 1) {
        // ==========================================
        //  Load-Aware Dynamic SP Size Strategy
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        int num_tokens = seq.num_tokens;

        // Collect current state for SP size decision
        std::vector<int> free_blocks_per_rank = get_free_blocks_per_rank();
        std::vector<int> used_blocks_per_rank = get_used_blocks_per_rank();
        std::vector<int> batch_size_per_rank = get_batch_size_per_rank();

        // Use new SP size policy to determine SP size, master rank AND detailed distribution
        SPSizeDecision decision = sp_size_policy_.determine_sp_size(
            num_tokens,
            free_blocks_per_rank,
            used_blocks_per_rank,
            long_used_per_rank,
            batch_size_per_rank,
            num_kvcache_blocks_,
            load_stats_,
            attention_sp_,
            kvcache_block_size_
        );
        
        // Apply the decision
        block_ctx.master_sp_idx_ = decision.master_rank;
        block_ctx.num_dispatched_tokens = decision.dispatch_tokens;

        // Final sanity check for capacity (reservation and physical)
        // Note: The policy already simulated this, but we double-check here against current master load limits
        int master_rank = decision.master_rank;
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) return false;

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) return false;

        // Perform final memory and reservation check for the selected distribution
        std::vector<int> master_req_counts(attention_sp_, 0);
        for (const auto& rs : running) {
            int m_idx = rs->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx]++;
        }
        for (const auto& [m_idx, count] : num_seqs) {
            if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx] += count;
        }
        master_req_counts[master_rank]++;

        bool memory_check_passed = true;
        for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
            int dispatched = block_ctx.num_dispatched_tokens[sp_idx];
            if (dispatched > 0 || sp_idx == master_rank) {
                // Check per-rank receiver limits
                if (sp_idx != master_rank && dispatched > 0) {
                    if (num_recv_seqs_per_sp_[sp_idx] >= max_num_recv_seqs_) {
                        memory_check_passed = false;
                        break;
                    }
                }

                int free_blocks = block_manager[sp_idx]->num_free_blocks();
                int prefill_blocks_needed = (dispatched + kvcache_block_size_ - 1) / kvcache_block_size_;

                // Standard per-request reservation check
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
            // Log scheduling decision for debugging
            if (enable_scheduling_log_) {
                std::cout << "[SP Schedule] seq_id=" << seq.seq_id
                            << " | tokens=" << num_tokens
                            << " | sp_size=" << decision.sp_size
                            << " | master=" << master_rank
                            << " | reason=" << (decision.due_to_imbalance ? "imbalance" : 
                                                (decision.due_to_pressure ? "pressure" : "default"));
                
                std::cout << " | dispatch=[";
                bool first = true;
                for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
                    if (block_ctx.num_dispatched_tokens[sp_idx] > 0) {
                        if (!first) std::cout << ",";
                        std::cout << "r" << sp_idx << ":" << block_ctx.num_dispatched_tokens[sp_idx];
                        first = false;
                    }
                }
                std::cout << "]";
                
                std::cout << " | free=[";
                for (int i = 0; i < attention_sp_; ++i) {
                    if (i > 0) std::cout << ",";
                    std::cout << block_manager[i]->num_free_blocks();
                }
                std::cout << "]";
                
                if (sp_size_policy_.config().mode == SPSizeMode::LoadAware) {
                    std::cout << " | is_long=" << (load_stats_.is_long_request(num_tokens) ? "Y" : "N")
                                << " | imbalance=" << std::fixed << std::setprecision(2) 
                                << load_stats_.kvcache_imbalance_ratio()
                                << " | long_used_avg=" << std::accumulate(long_used_per_rank.begin(), long_used_per_rank.end(), 0) / attention_sp_
                                << " | exp_wait=" << std::setprecision(1) 
                                << load_stats_.expected_waiting_requests();
                }
                std::cout << std::endl;
            }
            return true;
        }
        return false;
    } 
    else {
        // ==========================================
        //  Naive Strategy (SP=1)
        // ==========================================
        auto& block_ctx = seq.block_ctx(BlockContextSlot::ACTIVE);
        block_ctx.num_dispatched_tokens.assign(attention_sp_, 0);

        int master_rank = select_master_rank();
        if (master_seq_counts_[master_rank] + 1 > max_num_seqs_) return false;

        auto it_tokens              = num_batched_tokens.find(master_rank);
        int  current_batched_tokens = (it_tokens != num_batched_tokens.end()) ? it_tokens->second : 0;
        if (current_batched_tokens + seq.num_tokens >= max_num_batched_tokens_) return false;

        block_ctx.master_sp_idx_ = master_rank;
        block_ctx.num_dispatched_tokens[master_rank] = seq.num_tokens;

        std::vector<int> master_req_counts(attention_sp_, 0);
        for (const auto& s : running) {
            int m_idx = s->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_;
            if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx]++;
        }
        for (const auto& [m_idx, count] : num_seqs) {
            if (m_idx >= 0 && m_idx < attention_sp_) master_req_counts[m_idx] += count;
        }
        master_req_counts[master_rank]++;

        int free_blocks = block_manager[master_rank]->num_free_blocks();
        int prefill_blocks_needed = (seq.num_tokens + kvcache_block_size_ - 1) / kvcache_block_size_;
        double needed_float = master_req_counts[master_rank] * reserved_blocks_per_req_;
        int reservation_blocks_needed = static_cast<int>(std::ceil(needed_float));

        if (free_blocks < prefill_blocks_needed + reservation_blocks_needed) return false;
        if (!block_manager[master_rank]->can_allocate(seq)) return false;

        return true;
    }
}

void SPStateManager::allocate(Sequence& seq)
{
    auto& block_ctx     = seq.block_ctx(BlockContextSlot::ACTIVE);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        master_seq_counts_[master_sp_idx]++;
    }

    for (int sp_idx = 0; sp_idx < attention_sp_; ++sp_idx) {
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
    auto& block_ctx     = seq.block_ctx(slot);
    int   master_sp_idx = block_ctx.master_sp_idx_;

    int actual_sp_size = 0;
    for (int tokens : block_ctx.num_dispatched_tokens) {
        if (tokens > 0) actual_sp_size++;
    }

    int prompt_length = seq.num_prompt_tokens;
    int output_length = seq.num_tokens - seq.num_prompt_tokens;
    load_stats_.record_request(prompt_length, output_length);

    if (actual_sp_size > 1) {
        float kv_imbalance = load_stats_.kvcache_imbalance_ratio();
        std::vector<int> bs_per_rank = get_batch_size_per_rank();
        float bs_sum = std::accumulate(bs_per_rank.begin(), bs_per_rank.end(), 0.0f);
        float bs_avg = bs_sum / attention_sp_;
        float bs_sq_sum = 0;
        for (int bs : bs_per_rank) {
            float diff = static_cast<float>(bs) - bs_avg;
            bs_sq_sum += diff * diff;
        }
        float bs_cv = (bs_avg > 0) ? std::sqrt(bs_sq_sum / attention_sp_) / bs_avg : 0.0f;

        bool was_beneficial = (kv_imbalance > 1.5f) || (bs_cv > 0.3f);
        load_stats_.update_learned_thresholds(actual_sp_size, was_beneficial);
        
        if (enable_scheduling_log_) {
            std::cout << "[SP Feedback] seq_id=" << seq.seq_id 
                      << " | sp_size=" << actual_sp_size 
                      << " | beneficial=" << (was_beneficial ? "Y" : "N")
                      << " | kv_imb=" << std::fixed << std::setprecision(2) << kv_imbalance 
                      << " | bs_cv=" << bs_cv 
                      << " | long_req_thr=" << load_stats_.long_req_threshold() << std::endl;
        }
    }

    if (master_sp_idx >= 0 && master_sp_idx < attention_sp_) {
        if (master_seq_counts_[master_sp_idx] > 0) {
            master_seq_counts_[master_sp_idx]--;
        }
    }

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
