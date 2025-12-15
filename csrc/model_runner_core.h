#pragma once

#include "sequence_core.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <string>
#include <optional>

namespace py = pybind11;

struct PrefillMetadata {
    py::array_t<int64_t> input_ids;
    py::array_t<int64_t> positions;
    py::array_t<int32_t> cu_seqlens_q;
    py::array_t<int32_t> cu_seqlens_k;
    int32_t max_seqlen_q;
    int32_t max_seqlen_k;
    py::array_t<int32_t> slot_mapping;
    py::array_t<int32_t> block_tables;
};

struct DecodeMetadata {
    py::array_t<int64_t> input_ids;
    py::array_t<int64_t> positions;
    py::array_t<int32_t> slot_mapping;
    py::array_t<int32_t> context_lens;
    py::array_t<int32_t> context_lens_for_attn;
    py::array_t<int32_t> global_context_lens;
    py::array_t<int32_t> block_tables;
    py::array_t<int32_t> q_mask;
    py::array_t<int32_t> res_lse_mask;
    
    py::array_t<int32_t> q_slice_get;
    py::array_t<int32_t> q_slice_fill;
    py::array_t<int32_t> q_copy_mask;
    
    py::array_t<int32_t> res_slice_get_to_buffer_output;
    py::array_t<int32_t> res_slice_fill_to_buffer_output;
    py::array_t<int32_t> res_to_buffer_output_mask;
    
    py::array_t<int32_t> res_slice_get_to_buffer_input;
    py::array_t<int32_t> res_slice_fill_to_buffer_input;
    py::array_t<int32_t> res_to_buffer_input_mask;

    int32_t attention_compute_bs;
};

class ModelRunnerCore {
public:
    std::string engine_id;
    int rank;
    int world_size;
    int max_num_seqs;
    int max_num_send_seqs;
    int max_num_recv_seqs;
    int block_size;
    int attn_sp_rank;
    int attn_sp_world_size;

    ModelRunnerCore(
        std::string engine_id,
        int rank,
        int world_size,
        int max_num_seqs,
        int max_num_send_seqs,
        int max_num_recv_seqs,
        int block_size,
        int attn_sp_rank,
        int attn_sp_world_size
    ) : engine_id(engine_id), rank(rank), world_size(world_size),
        max_num_seqs(max_num_seqs), max_num_send_seqs(max_num_send_seqs),
        max_num_recv_seqs(max_num_recv_seqs), block_size(block_size),
        attn_sp_rank(attn_sp_rank), attn_sp_world_size(attn_sp_world_size) {}

    PrefillMetadata prepare_prefill(std::vector<std::shared_ptr<Sequence>>& seqs, bool is_dummy);
    DecodeMetadata prepare_decode(std::vector<std::shared_ptr<Sequence>>& dp_seqs, bool is_dummy);
    
private:
    py::array_t<int32_t> prepare_block_tables(std::vector<std::shared_ptr<Sequence>>& dp_seqs);
};

void bind_model_runner(py::module& m);
