#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nanodeploy/csrc/engine/serialization.h"
#include "nanodeploy/csrc/worker/model_runner_utils.h"

namespace py = pybind11;
using namespace nanodeploy;

void bind_model_runner_utils(py::module_& m)
{
    // ========== Metadata structs ==========
    py::class_<PrefillMetadata>(m, "PrefillMetadata")
        .def_readonly("input_ids", &PrefillMetadata::input_ids)
        .def_readonly("positions", &PrefillMetadata::positions)
        .def_readonly("cu_seqlens_q", &PrefillMetadata::cu_seqlens_q)
        .def_readonly("cu_seqlens_k", &PrefillMetadata::cu_seqlens_k)
        .def_readonly("max_seqlen_q", &PrefillMetadata::max_seqlen_q)
        .def_readonly("max_seqlen_k", &PrefillMetadata::max_seqlen_k)
        .def_readonly("slot_mapping", &PrefillMetadata::slot_mapping)
        .def_readonly("block_tables_flat", &PrefillMetadata::block_tables_flat)
        .def_readonly("max_num_blocks", &PrefillMetadata::max_num_blocks)
        .def_readonly("use_block_tables", &PrefillMetadata::use_block_tables)
        .def_readonly("sampling_token_indices", &PrefillMetadata::sampling_token_indices)
        .def_readonly("sampling_seq_indices", &PrefillMetadata::sampling_seq_indices);

    py::class_<DecodeMetadata>(m, "DecodeMetadata")
        .def_readonly("input_ids", &DecodeMetadata::input_ids)
        .def_readonly("positions", &DecodeMetadata::positions)
        .def_readonly("slot_mapping", &DecodeMetadata::slot_mapping)
        .def_readonly("context_lens_flat", &DecodeMetadata::context_lens_flat)
        .def_readonly("global_context_lens_flat", &DecodeMetadata::global_context_lens_flat)
        .def_readonly("block_tables_flat", &DecodeMetadata::block_tables_flat)
        .def_readonly("max_num_blocks", &DecodeMetadata::max_num_blocks)
        .def_readonly("context_lens_for_attn", &DecodeMetadata::context_lens_for_attn)
        .def_readonly("q_slice_get", &DecodeMetadata::q_slice_get)
        .def_readonly("q_slice_fill", &DecodeMetadata::q_slice_fill)
        .def_readonly("q_copy_mask", &DecodeMetadata::q_copy_mask)
        .def_readonly("res_slice_get_to_buffer_output", &DecodeMetadata::res_slice_get_to_buffer_output)
        .def_readonly("res_slice_fill_to_buffer_output", &DecodeMetadata::res_slice_fill_to_buffer_output)
        .def_readonly("res_to_buffer_output_mask", &DecodeMetadata::res_to_buffer_output_mask)
        .def_readonly("res_slice_get_to_buffer_input", &DecodeMetadata::res_slice_get_to_buffer_input)
        .def_readonly("res_slice_fill_to_buffer_input", &DecodeMetadata::res_slice_fill_to_buffer_input)
        .def_readonly("res_to_buffer_input_mask", &DecodeMetadata::res_to_buffer_input_mask)
        .def_readonly("q_offsets", &DecodeMetadata::q_offsets);

    py::class_<BatchAuxData>(m, "BatchAuxData")
        .def_readonly("temperatures", &BatchAuxData::temperatures)
        .def_readonly("state_slots", &BatchAuxData::state_slots)
        .def_readonly("master_group_indices", &BatchAuxData::master_group_indices)
        .def_readonly("num_group_seqs", &BatchAuxData::num_group_seqs);

    py::class_<MigrateSequenceView>(m, "MigrateSequenceView")
        .def_readonly("seq_id", &MigrateSequenceView::seq_id)
        .def_readonly("migrate_engine_id", &MigrateSequenceView::migrate_engine_id)
        .def_readonly("migrate_num_kvcache_blocks", &MigrateSequenceView::migrate_num_kvcache_blocks)
        .def_readonly("migrate_group_size", &MigrateSequenceView::migrate_group_size)
        .def_readonly("migrate_dp_idx", &MigrateSequenceView::migrate_dp_idx)
        .def_readonly("migrate_block_location", &MigrateSequenceView::migrate_block_location)
        .def_readonly("migrate_state_slot", &MigrateSequenceView::migrate_state_slot)
        .def_readonly("active_block_location", &MigrateSequenceView::active_block_location)
        .def_readonly("active_state_slot", &MigrateSequenceView::active_state_slot);

    py::class_<VisionSlotView>(m, "VisionSlotView")
        .def_readonly("encoder_engine_id", &VisionSlotView::encoder_engine_id)
        .def_readonly("slot_idx", &VisionSlotView::slot_idx)
        .def_readonly("num_tokens", &VisionSlotView::num_tokens)
        .def_readonly("hidden_size", &VisionSlotView::hidden_size)
        .def_readonly("max_tokens_per_slot", &VisionSlotView::max_tokens_per_slot)
        .def_readonly("seq_index", &VisionSlotView::seq_index);

    // ========== Engine side: serialize → py::bytes ==========
    m.def(
        "serialize_run_batch",
        [](const std::vector<Sequence*>& seqs, bool is_prefill) -> py::bytes {
            auto buf = serialize_run_batch(seqs, is_prefill);
            return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        },
        py::arg("seqs"),
        py::arg("is_prefill"));

    m.def(
        "serialize_migrate_batch",
        [](const std::vector<Sequence*>& seqs) -> py::bytes {
            auto buf = serialize_migrate_batch(seqs);
            return py::bytes(reinterpret_cast<const char*>(buf.data()), buf.size());
        },
        py::arg("seqs"));

    // ========== Runner side: deserialize + prepare ==========
    m.def(
        "prepare_prefill_from_bytes",
        [](py::bytes data, int group_rank, int group_size, int block_size, int max_num_seqs, int num_gpu_blocks)
            -> PrefillMetadata {
            std::string_view sv = data;
            return prepare_prefill_from_bytes(reinterpret_cast<const uint8_t*>(sv.data()),
                                              sv.size(),
                                              group_rank,
                                              group_size,
                                              block_size,
                                              max_num_seqs,
                                              num_gpu_blocks);
        },
        py::arg("data"),
        py::arg("group_rank"),
        py::arg("group_size"),
        py::arg("block_size"),
        py::arg("max_num_seqs"),
        py::arg("num_gpu_blocks"));

    m.def(
        "prepare_decode_from_bytes",
        [](py::bytes data, int group_rank, int group_size, int block_size, int max_num_seqs, int num_gpu_blocks)
            -> DecodeMetadata {
            std::string_view sv = data;
            return prepare_decode_from_bytes(reinterpret_cast<const uint8_t*>(sv.data()),
                                             sv.size(),
                                             group_rank,
                                             group_size,
                                             block_size,
                                             max_num_seqs,
                                             num_gpu_blocks);
        },
        py::arg("data"),
        py::arg("group_rank"),
        py::arg("group_size"),
        py::arg("block_size"),
        py::arg("max_num_seqs"),
        py::arg("num_gpu_blocks"));

    m.def(
        "extract_aux_from_bytes",
        [](py::bytes data, int group_rank) -> BatchAuxData {
            std::string_view sv = data;
            return extract_aux_from_bytes(reinterpret_cast<const uint8_t*>(sv.data()), sv.size(), group_rank);
        },
        py::arg("data"),
        py::arg("group_rank"));

    m.def(
        "extract_vision_slots_from_bytes",
        [](py::bytes data) -> std::vector<VisionSlotView> {
            std::string_view sv = data;
            return extract_vision_slots_from_bytes(reinterpret_cast<const uint8_t*>(sv.data()), sv.size());
        },
        py::arg("data"));

    m.def(
        "parse_migrate_batch",
        [](py::bytes data) -> std::vector<MigrateSequenceView> {
            std::string_view sv = data;
            return parse_migrate_batch(reinterpret_cast<const uint8_t*>(sv.data()), sv.size());
        },
        py::arg("data"));
}
