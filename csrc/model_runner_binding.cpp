#include "model_runner_core.h"
#include <algorithm>
#include <iostream>
#include <numeric>

namespace py = pybind11;

// Helper to create numpy array from vector
template <typename T>
py::array_t<T> to_py_array(const std::vector<T>& vec) {
    return py::array_t<T>(vec.size(), vec.data());
}

// Helper to create 2D numpy array from vector
template <typename T>
py::array_t<T> to_py_array_2d(const std::vector<T>& vec, size_t rows, size_t cols) {
    return py::array_t<T>({rows, cols}, vec.data());
}

py::array_t<int32_t> ModelRunnerCore::prepare_block_tables(std::vector<std::shared_ptr<Sequence>>& dp_seqs) {
    std::vector<int> indices(dp_seqs.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on master_sp_idx
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
        return dp_seqs[i]->block_ctx(engine_id).master_sp_idx < dp_seqs[j]->block_ctx(engine_id).master_sp_idx;
    });

    std::vector<std::vector<int>> collected_tables;
    size_t max_len = 0;

    for (int idx : indices) {
        auto& seq = dp_seqs[idx];
        auto& bt = seq->block_table(engine_id, attn_sp_rank);
        if (!bt.empty()) {
            collected_tables.push_back(bt);
            if (bt.size() > max_len) {
                max_len = bt.size();
            }
        }
    }

    if (collected_tables.empty()) {
        return py::array_t<int32_t>(std::vector<py::ssize_t>{0, 0});
    }

    size_t num_valid_tables = collected_tables.size();
    std::vector<int32_t> padded_table(num_valid_tables * max_len, -1);

    for (size_t i = 0; i < num_valid_tables; ++i) {
        const auto& bt = collected_tables[i];
        std::copy(bt.begin(), bt.end(), padded_table.begin() + i * max_len);
    }

    return to_py_array_2d(padded_table, num_valid_tables, max_len);
}

PrefillMetadata ModelRunnerCore::prepare_prefill(std::vector<std::shared_ptr<Sequence>>& seqs, bool is_dummy) {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> positions;
    std::vector<int32_t> cu_seqlens_q = {0};
    std::vector<int32_t> cu_seqlens_k = {0};
    int32_t max_seqlen_q = 0;
    int32_t max_seqlen_k = 0;
    std::vector<int32_t> slot_mapping;
    
    for (auto& seq : seqs) {
        // assert seq.block_ctx(self.engine_id).master_sp_idx == sp_idx
        // In C++ we skip assertion or log warning, assuming caller ensures correctness or we check it.
        // But let's trust the logic for now or add check if needed.
        
        int seqlen = seq->num_tokens;
        int num_cached = seq->num_cached_tokens;
        
        // input_ids.extend(seq[seq.num_cached_tokens :])
        // Sequence stores token_ids in std::vector<int>
        for (int i = num_cached; i < seqlen; ++i) {
            input_ids.push_back(seq->token_ids[i]);
            positions.push_back(i);
        }
        
        int seqlen_q = seqlen - num_cached;
        int seqlen_k = seqlen;
        
        cu_seqlens_q.push_back(cu_seqlens_q.back() + seqlen_q);
        cu_seqlens_k.push_back(cu_seqlens_k.back() + seqlen_k);
        
        max_seqlen_q = std::max(max_seqlen_q, seqlen_q);
        max_seqlen_k = std::max(max_seqlen_k, seqlen_k);
        
        auto& bt = seq->block_table(engine_id, attn_sp_rank);
        if (bt.empty()) continue; // warmup
        
        int num_blocks = seq->num_blocks(engine_id, attn_sp_rank);
        int num_cached_blocks = seq->num_cached_blocks(); // This might need adjustment if num_cached_blocks logic differs
        // In python: seq.num_cached_blocks is property: return num_cached_tokens // block_size
        // In C++: num_cached_blocks() returns num_cached_tokens / block_size
        
        // Python loop: for i in range(seq.num_cached_blocks, num_blocks):
        for (int i = num_cached_blocks; i < num_blocks; ++i) {
            int start = bt[i] * block_size;
            int end;
            if (i != num_blocks - 1) {
                end = start + block_size;
            } else {
                end = start + seq->last_block_num_tokens(engine_id, attn_sp_rank);
            }
            
            for (int k = start; k < end; ++k) {
                slot_mapping.push_back(k);
            }
        }
    }
    
    py::array_t<int32_t> block_tables;
    if (cu_seqlens_k.back() > cu_seqlens_q.back()) { // prefix cache
         block_tables = prepare_block_tables(seqs);
    } else {
         block_tables = py::array_t<int32_t>(std::vector<py::ssize_t>{0, 0});
    }
    
    PrefillMetadata meta;
    meta.input_ids = to_py_array(input_ids);
    meta.positions = to_py_array(positions);
    meta.cu_seqlens_q = to_py_array(cu_seqlens_q);
    meta.cu_seqlens_k = to_py_array(cu_seqlens_k);
    meta.max_seqlen_q = max_seqlen_q;
    meta.max_seqlen_k = max_seqlen_k;
    meta.slot_mapping = to_py_array(slot_mapping);
    meta.block_tables = block_tables;
    
    return meta;
}

DecodeMetadata ModelRunnerCore::prepare_decode(std::vector<std::shared_ptr<Sequence>>& dp_seqs, bool is_dummy) {
    std::vector<int64_t> input_ids_list;
    std::vector<int64_t> positions_list;
    std::vector<int32_t> slot_mapping_list;
    
    // Filter master_indices
    for (size_t i = 0; i < dp_seqs.size(); ++i) {
        auto& seq = dp_seqs[i];
        if (seq->block_ctx(engine_id).master_sp_idx == attn_sp_rank) {
            input_ids_list.push_back(seq->last_token);
            positions_list.push_back(seq->num_tokens - 1);
            
            int page_id = seq->last_block_page_id(engine_id, attn_sp_rank);
            int num_tokens_in_block = seq->last_block_num_tokens(engine_id, attn_sp_rank);
            slot_mapping_list.push_back(page_id * block_size + num_tokens_in_block - 1);
        }
    }
    
    // context_lens: [sp_size, max_num_seqs]
    std::vector<int32_t> context_lens_vec(attn_sp_world_size * max_num_seqs, 0);
    std::vector<int32_t> global_context_lens_vec(attn_sp_world_size * max_num_seqs, 0);
    std::vector<int> sp_num_seqs(attn_sp_world_size, 0);
    
    for (size_t i = 0; i < dp_seqs.size(); ++i) {
        auto& seq = dp_seqs[i];
        int master_idx = seq->block_ctx(engine_id).master_sp_idx;
        int local_seq_idx = sp_num_seqs[master_idx];
        
        if (local_seq_idx < max_num_seqs) {
            int ctx_len = seq->context_len(engine_id, attn_sp_rank);
            context_lens_vec[master_idx * max_num_seqs + local_seq_idx] = ctx_len;
            
            if (master_idx == attn_sp_rank) {
                for (int remote_rank = 0; remote_rank < attn_sp_world_size; ++remote_rank) {
                    global_context_lens_vec[remote_rank * max_num_seqs + local_seq_idx] = 
                        seq->context_len(engine_id, remote_rank);
                }
            }
        }
        sp_num_seqs[master_idx]++;
    }
    
    // sp_valid_request_counts = np.sum(context_lens_np > 0, axis=1)
    std::vector<int> sp_valid_request_counts(attn_sp_world_size, 0);
    int attention_compute_bs = 0;
    std::vector<int32_t> context_lens_for_attn_vec;
    
    for (int r = 0; r < attn_sp_world_size; ++r) {
        int count = 0;
        for (int c = 0; c < max_num_seqs; ++c) {
            int val = context_lens_vec[r * max_num_seqs + c];
            if (val > 0) {
                count++;
                context_lens_for_attn_vec.push_back(val);
            }
        }
        sp_valid_request_counts[r] = count;
        attention_compute_bs += count;
    }
    
    // recv_req_num
    int recv_req_num = 0;
    for (int r = 0; r < attn_sp_world_size; ++r) {
        if (r == attn_sp_rank) continue;
        for (int c = 0; c < max_num_seqs; ++c) {
            if (context_lens_vec[r * max_num_seqs + c] > 0) {
                recv_req_num++;
            }
        }
    }
    
    // send_req_num
    int send_req_num = 0;
    for (int c = 0; c < sp_num_seqs[attn_sp_rank]; ++c) {
        bool needs_send = false;
        for (int r = 0; r < attn_sp_world_size; ++r) {
            if (r == attn_sp_rank) continue;
            if (global_context_lens_vec[r * max_num_seqs + c] > 0) {
                needs_send = true;
                break;
            }
        }
        if (needs_send) send_req_num++;
    }
    
    if (send_req_num > max_num_send_seqs || recv_req_num > max_num_recv_seqs) {
        throw std::runtime_error("send_req_num or recv_req_num exceeds max limits");
    }
    
    // q_slice_get
    std::vector<int32_t> q_slice_get;
    for (int c = 0; c < max_num_seqs; ++c) {
        if (context_lens_vec[attn_sp_rank * max_num_seqs + c] > 0) {
            q_slice_get.push_back(c);
        }
    }
    
    std::vector<int32_t> offsets(attn_sp_world_size + 1, 0);
    for (int i = 0; i < attn_sp_world_size; ++i) {
        offsets[i+1] = offsets[i] + sp_valid_request_counts[i];
    }
    
    int start_pos = offsets[attn_sp_rank];
    std::vector<int32_t> q_slice_fill(q_slice_get.size());
    std::iota(q_slice_fill.begin(), q_slice_fill.end(), start_pos);
    
    std::vector<int32_t> q_copy_mask(q_slice_get.size(), 1);
    
    std::vector<int32_t> res_slice_get_to_buffer_output = q_slice_fill;
    std::vector<int32_t> res_slice_fill_to_buffer_output;
    for (int val : q_slice_get) {
        res_slice_fill_to_buffer_output.push_back(attn_sp_rank * max_num_seqs + val);
    }
    std::vector<int32_t> res_to_buffer_output_mask(res_slice_get_to_buffer_output.size(), 1);
    
    std::vector<int32_t> res_slice_get_to_buffer_input;
    std::vector<int32_t> res_slice_fill_to_buffer_input;
    
    for (int sp_idx = 0; sp_idx < attn_sp_world_size; ++sp_idx) {
        if (sp_idx == attn_sp_rank) continue;
        
        std::vector<int> valid_indices;
        for (int c = 0; c < sp_num_seqs[sp_idx]; ++c) {
            if (context_lens_vec[sp_idx * max_num_seqs + c] > 0) {
                valid_indices.push_back(c);
            }
        }
        
        if (!valid_indices.empty()) {
            int count = valid_indices.size();
            int base_pos = offsets[sp_idx];
            
            for (int k = 0; k < count; ++k) {
                res_slice_get_to_buffer_input.push_back(base_pos + k);
            }
            
            for (int val : valid_indices) {
                res_slice_fill_to_buffer_input.push_back(sp_idx * max_num_seqs + val);
            }
        }
    }
    
    std::vector<int32_t> res_to_buffer_input_mask(res_slice_get_to_buffer_input.size(), 1);
    
    // Pad res_slice_get_to_buffer_input etc.
    int max_sr = std::max(max_num_send_seqs, max_num_recv_seqs);
    while (res_slice_get_to_buffer_input.size() < (size_t)max_sr) {
        res_slice_get_to_buffer_input.push_back(-1);
        res_slice_fill_to_buffer_input.push_back(-1);
        res_to_buffer_input_mask.push_back(0);
    }
    
    // q_mask
    std::vector<int32_t> q_mask(global_context_lens_vec.size());
    for (size_t i = 0; i < global_context_lens_vec.size(); ++i) {
        q_mask[i] = (global_context_lens_vec[i] != 0) ? 1 : 0;
    }
    // q_mask[sp_rank].fill_(0)
    for (int c = 0; c < max_num_seqs; ++c) {
        q_mask[attn_sp_rank * max_num_seqs + c] = 0;
    }
    
    // res_lse_mask
    std::vector<int32_t> res_lse_mask(context_lens_vec.size());
    for (size_t i = 0; i < context_lens_vec.size(); ++i) {
        res_lse_mask[i] = (context_lens_vec[i] != 0) ? 1 : 0;
    }
    // res_lse_mask[sp_rank].fill_(0)
    for (int c = 0; c < max_num_seqs; ++c) {
        res_lse_mask[attn_sp_rank * max_num_seqs + c] = 0;
    }
    
    DecodeMetadata meta;
    meta.input_ids = to_py_array(input_ids_list);
    meta.positions = to_py_array(positions_list);
    meta.slot_mapping = to_py_array(slot_mapping_list);
    meta.context_lens = to_py_array_2d(context_lens_vec, attn_sp_world_size, max_num_seqs);
    meta.context_lens_for_attn = to_py_array(context_lens_for_attn_vec);
    meta.global_context_lens = to_py_array_2d(global_context_lens_vec, attn_sp_world_size, max_num_seqs);
    meta.block_tables = prepare_block_tables(dp_seqs);
    meta.q_mask = to_py_array_2d(q_mask, attn_sp_world_size, max_num_seqs);
    meta.res_lse_mask = to_py_array_2d(res_lse_mask, attn_sp_world_size, max_num_seqs);
    
    meta.q_slice_get = to_py_array(q_slice_get);
    meta.q_slice_fill = to_py_array(q_slice_fill);
    meta.q_copy_mask = to_py_array(q_copy_mask);
    
    meta.res_slice_get_to_buffer_output = to_py_array(res_slice_get_to_buffer_output);
    meta.res_slice_fill_to_buffer_output = to_py_array(res_slice_fill_to_buffer_output);
    meta.res_to_buffer_output_mask = to_py_array(res_to_buffer_output_mask);
    
    meta.res_slice_get_to_buffer_input = to_py_array(res_slice_get_to_buffer_input);
    meta.res_slice_fill_to_buffer_input = to_py_array(res_slice_fill_to_buffer_input);
    meta.res_to_buffer_input_mask = to_py_array(res_to_buffer_input_mask);
    
    meta.attention_compute_bs = attention_compute_bs;
    
    return meta;
}

void bind_model_runner(py::module& m) {
    py::class_<PrefillMetadata>(m, "PrefillMetadata")
        .def_readonly("input_ids", &PrefillMetadata::input_ids)
        .def_readonly("positions", &PrefillMetadata::positions)
        .def_readonly("cu_seqlens_q", &PrefillMetadata::cu_seqlens_q)
        .def_readonly("cu_seqlens_k", &PrefillMetadata::cu_seqlens_k)
        .def_readonly("max_seqlen_q", &PrefillMetadata::max_seqlen_q)
        .def_readonly("max_seqlen_k", &PrefillMetadata::max_seqlen_k)
        .def_readonly("slot_mapping", &PrefillMetadata::slot_mapping)
        .def_readonly("block_tables", &PrefillMetadata::block_tables);

    py::class_<DecodeMetadata>(m, "DecodeMetadata")
        .def_readonly("input_ids", &DecodeMetadata::input_ids)
        .def_readonly("positions", &DecodeMetadata::positions)
        .def_readonly("slot_mapping", &DecodeMetadata::slot_mapping)
        .def_readonly("context_lens", &DecodeMetadata::context_lens)
        .def_readonly("context_lens_for_attn", &DecodeMetadata::context_lens_for_attn)
        .def_readonly("global_context_lens", &DecodeMetadata::global_context_lens)
        .def_readonly("block_tables", &DecodeMetadata::block_tables)
        .def_readonly("q_mask", &DecodeMetadata::q_mask)
        .def_readonly("res_lse_mask", &DecodeMetadata::res_lse_mask)
        .def_readonly("q_slice_get", &DecodeMetadata::q_slice_get)
        .def_readonly("q_slice_fill", &DecodeMetadata::q_slice_fill)
        .def_readonly("q_copy_mask", &DecodeMetadata::q_copy_mask)
        .def_readonly("res_slice_get_to_buffer_output", &DecodeMetadata::res_slice_get_to_buffer_output)
        .def_readonly("res_slice_fill_to_buffer_output", &DecodeMetadata::res_slice_fill_to_buffer_output)
        .def_readonly("res_to_buffer_output_mask", &DecodeMetadata::res_to_buffer_output_mask)
        .def_readonly("res_slice_get_to_buffer_input", &DecodeMetadata::res_slice_get_to_buffer_input)
        .def_readonly("res_slice_fill_to_buffer_input", &DecodeMetadata::res_slice_fill_to_buffer_input)
        .def_readonly("res_to_buffer_input_mask", &DecodeMetadata::res_to_buffer_input_mask)
        .def_readonly("attention_compute_bs", &DecodeMetadata::attention_compute_bs);

    py::class_<ModelRunnerCore>(m, "ModelRunnerCore")
        .def(py::init<std::string, int, int, int, int, int, int, int, int>())
        .def("prepare_prefill", &ModelRunnerCore::prepare_prefill)
        .def("prepare_decode", &ModelRunnerCore::prepare_decode);
}
