#include "model_runner_utils.h"
#include <algorithm>
#include <iostream>

namespace nanodeploy {

// Helper to mimic Python's updated prepare_block_tables logic (Packed format)
static void build_block_tables_packed(
    const std::vector<Sequence*>& dp_seqs,
    const std::string& engine_id,
    int sp_rank,
    int sp_size,
    std::vector<int>& block_tables_flat,
    int& max_num_blocks
) {
    // 1. Group sequences by master_sp_idx
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(engine_id).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Collect valid block tables and calculate max_num_blocks
    struct ValidBlockTable {
        const std::vector<int>* bt_ptr;
    };
    std::vector<ValidBlockTable> valid_tables;

    max_num_blocks = 0;
    
    // Iterate in sp_idx order as per Python implementation
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            const auto& bt = seq->block_table(engine_id, sp_rank);
            if (!bt.empty()) {
                valid_tables.push_back({&bt});
                if ((int)bt.size() > max_num_blocks) {
                    max_num_blocks = (int)bt.size();
                }
            }
        }
    }

    // 3. Fill flattened table: [num_valid_seqs, max_num_blocks]
    if (valid_tables.empty()) {
        block_tables_flat.clear();
        max_num_blocks = 0;
        return;
    }

    size_t total_size = valid_tables.size() * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (size_t i = 0; i < valid_tables.size(); ++i) {
        const auto& bt = *valid_tables[i].bt_ptr;
        size_t base_offset = i * max_num_blocks;
        for (size_t k = 0; k < bt.size(); ++k) {
            block_tables_flat[base_offset + k] = bt[k];
        }
    }
}

static void build_block_tables_sparse(
    const std::vector<Sequence*>& dp_seqs,
    const std::string& engine_id,
    int sp_rank,
    int sp_size,
    int max_num_seqs,
    std::vector<int>& block_tables_flat,
    int& max_num_blocks
) {
     // 1. Group sequences by master_sp_idx
    std::vector<std::vector<Sequence*>> dp_sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(engine_id).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            dp_sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Calculate max_num_blocks
    max_num_blocks = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (auto* seq : dp_sp_seqs[sp_idx]) {
            int size = (int)seq->block_table(engine_id, sp_rank).size();
            if (size > max_num_blocks) max_num_blocks = size;
        }
    }

    // 3. Fill table: [sp_size, max_num_seqs, max_num_blocks]
    size_t total_size = (size_t)sp_size * max_num_seqs * max_num_blocks;
    block_tables_flat.assign(total_size, -1);

    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& seqs = dp_sp_seqs[sp_idx];
        
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            size_t base_offset = ((size_t)sp_idx * max_num_seqs + seq_id) * max_num_blocks;
            
            if (seq_id < (int)seqs.size()) {
                Sequence* seq = seqs[seq_id];
                const auto& bt = seq->block_table(engine_id, sp_rank);
                for (size_t i = 0; i < bt.size(); ++i) {
                    block_tables_flat[base_offset + i] = bt[i];
                }
            }
        }
    }
}

PrefillMetadata prepare_prefill_cpp(
    const std::vector<Sequence*>& seqs,
    const std::string& engine_id,
    int sp_rank,
    int sp_size,
    int block_size,
    int max_num_seqs
) {
    PrefillMetadata meta;
    meta.cu_seqlens_q.push_back(0);
    meta.cu_seqlens_k.push_back(0);
    
    size_t est_tokens = seqs.size() * 256; 
    meta.input_ids.reserve(est_tokens);
    meta.positions.reserve(est_tokens);
    meta.slot_mapping.reserve(est_tokens);

    for (auto* seq : seqs) {
        if (seq->block_ctx(engine_id).master_sp_idx != sp_rank) {
             continue; 
        }

        int seqlen = seq->num_tokens;
        int num_cached = seq->num_cached_tokens;
        int seqlen_q = seqlen - num_cached;
        int seqlen_k = seqlen;

        const auto& full_tokens = seq->token_ids;
        for (int i = num_cached; i < seqlen; ++i) {
            meta.input_ids.push_back(full_tokens[i]);
            meta.positions.push_back(i);
        }

        meta.cu_seqlens_q.push_back(meta.cu_seqlens_q.back() + seqlen_q);
        meta.cu_seqlens_k.push_back(meta.cu_seqlens_k.back() + seqlen_k);
        meta.max_seqlen_q = std::max(meta.max_seqlen_q, seqlen_q);
        meta.max_seqlen_k = std::max(meta.max_seqlen_k, seqlen_k);

        const auto& bt = seq->block_table(engine_id, sp_rank);
        if (bt.empty()) {
            continue; 
        }

        int num_blocks = seq->num_blocks(engine_id, sp_rank);
        int num_cached_blocks = seq->num_cached_blocks();
        
        for (int i = num_cached_blocks; i < num_blocks; ++i) {
            int block_id = bt[i];
            int start = block_id * block_size;
            int end = (i != num_blocks - 1) 
                      ? start + block_size 
                      : start + seq->last_block_num_tokens(engine_id, sp_rank);
            
            for (int k = start; k < end; ++k) {
                meta.slot_mapping.push_back(k);
            }
        }
    }

    if (meta.cu_seqlens_k.back() > meta.cu_seqlens_q.back()) {
        meta.use_block_tables = true;
        // Prefill currently seems to use the sparse format in Python wrapper
        build_block_tables_sparse(seqs, engine_id, sp_rank, sp_size, max_num_seqs, 
                                  meta.block_tables_flat, meta.max_num_blocks);
    }

    return meta;
}

DecodeMetadata prepare_decode_cpp(
    const std::vector<Sequence*>& dp_seqs,
    const std::string& engine_id,
    int sp_rank,
    int sp_size,
    int block_size,
    int max_num_seqs
) {
    DecodeMetadata meta;
    
    // 1. Group sequences
    std::vector<std::vector<Sequence*>> sp_seqs(sp_size);
    for (auto* seq : dp_seqs) {
        int m_sp = seq->block_ctx(engine_id).master_sp_idx;
        if (m_sp >= 0 && m_sp < sp_size) {
            sp_seqs[m_sp].push_back(seq);
        }
    }

    // 2. Prepare input_ids, positions, slot_mapping (Local SP Rank)
    for (auto* seq : dp_seqs) {
        if (seq->block_ctx(engine_id).master_sp_idx == sp_rank) {
            meta.input_ids.push_back(seq->last_token);
            meta.positions.push_back(seq->num_tokens - 1);
            
            int page_id = seq->last_block_page_id(engine_id, sp_rank);
            int offset = seq->last_block_num_tokens(engine_id, sp_rank);
            meta.slot_mapping.push_back(page_id * block_size + offset - 1);
        }
    }

    // 3. Prepare context_lens and global_context_lens
    meta.context_lens_flat.assign(sp_size * max_num_seqs, 0);
    meta.global_context_lens_flat.assign(sp_size * max_num_seqs, 0);

    // To track valid request counts per SP
    std::vector<int> sp_valid_request_counts(sp_size, 0);

    // context_lens
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        const auto& batch_seqs = sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)batch_seqs.size()) {
                Sequence* seq = batch_seqs[seq_id];
                int ctx_len = seq->context_len(engine_id, sp_rank);
                meta.context_lens_flat[sp_idx * max_num_seqs + seq_id] = ctx_len;
                
                if (ctx_len > 0) {
                    meta.context_lens_for_attn.push_back(ctx_len);
                    sp_valid_request_counts[sp_idx]++;
                }
            }
        }
    }
    
    // Calculate q_output_stride (res_rank_start_loc)
    std::vector<int> res_rank_start_loc(sp_size, 0);
    for (int target_rank = 0; target_rank < sp_size; ++target_rank) {
        int offset = 0;
        
        // 遍历所有在当前 rank 之前的 sender rank
        for (int sender_rank = 0; sender_rank < sp_rank; ++sender_rank) {
            // 统计 sender_rank 发送给 target_rank 的有效请求数
            for (Sequence* seq : sp_seqs[target_rank]) {
                if (seq->context_len(engine_id, sender_rank) > 0) {
                    offset++;
                }
            }
        }
        
        res_rank_start_loc[target_rank] = offset;
    }
    meta.q_output_stride = res_rank_start_loc;

    meta.attention_compute_bs = 0;
    for(int c : sp_valid_request_counts) meta.attention_compute_bs += c;
    
    // global_context_lens
    const auto& my_master_seqs = sp_seqs[sp_rank];
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        for (int seq_id = 0; seq_id < max_num_seqs; ++seq_id) {
            if (seq_id < (int)my_master_seqs.size()) {
                Sequence* seq = my_master_seqs[seq_id];
                meta.global_context_lens_flat[sp_idx * max_num_seqs + seq_id] = 
                    seq->context_len(engine_id, sp_idx);
            }
        }
    }

    // 4. Calculate q_slice_get
    // Indices of sequences in the current sp_rank that have context_len > 0
    for (int seq_id = 0; seq_id < (int)sp_seqs[sp_rank].size(); ++seq_id) {
        // Access via flat array we just filled
        if (meta.context_lens_flat[sp_rank * max_num_seqs + seq_id] > 0) {
            meta.q_slice_get.push_back(seq_id);
        }
    }

    // 5. Calculate q_slice_fill, res_slice_get_to_buffer_output
    int current_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
            for (size_t i = 0; i < meta.q_slice_get.size(); ++i) {
                meta.q_slice_fill.push_back(current_pos + i);
            }
            current_pos += sp_valid_request_counts[sp_idx];
        } else {
            current_pos += sp_valid_request_counts[sp_idx];
        }
    }
    
    // q_copy_mask is just 1s
    meta.q_copy_mask.assign(meta.q_slice_get.size(), 1);

    // res_slice_get_to_buffer_output is same as q_slice_fill
    meta.res_slice_get_to_buffer_output = meta.q_slice_fill;

    // 6. Calculate res_slice_fill_to_buffer_output
    for (int seq_idx : meta.q_slice_get) {
        meta.res_slice_fill_to_buffer_output.push_back(sp_rank * max_num_seqs + seq_idx);
    }
    
    // res_to_buffer_output_mask is 1s
    meta.res_to_buffer_output_mask.assign(meta.res_slice_get_to_buffer_output.size(), 1);

    // 7. Calculate res_slice_get_to_buffer_input, res_slice_fill_to_buffer_input
    int current_attention_pos = 0;
    for (int sp_idx = 0; sp_idx < sp_size; ++sp_idx) {
        if (sp_idx == sp_rank) {
             current_attention_pos += sp_valid_request_counts[sp_idx];
             continue;
        }

        const auto& batch_seqs = sp_seqs[sp_idx];
        for (int seq_id = 0; seq_id < (int)batch_seqs.size(); ++seq_id) {
             // check context len
             if (meta.context_lens_flat[sp_idx * max_num_seqs + seq_id] > 0) {
                 meta.res_slice_get_to_buffer_input.push_back(current_attention_pos);
                 meta.res_slice_fill_to_buffer_input.push_back(sp_idx * max_num_seqs + seq_id);
                 current_attention_pos++;
             }
        }
    }
    
    // res_to_buffer_input_mask is 1s
    meta.res_to_buffer_input_mask.assign(meta.res_slice_get_to_buffer_input.size(), 1);

    // 8. Block tables (Packed format for decode)
    build_block_tables_packed(dp_seqs, engine_id, sp_rank, sp_size, 
                              meta.block_tables_flat, meta.max_num_blocks);

    return meta;
}

} // namespace nanodeploy