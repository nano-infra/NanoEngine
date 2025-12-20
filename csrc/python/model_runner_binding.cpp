#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nanodeploy/engine/model_runner_utils.h"

namespace py = pybind11;
using namespace nanodeploy;

void bind_model_runner_utils(py::module_& m)
{
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
        .def_readonly("use_block_tables", &PrefillMetadata::use_block_tables);

    py::class_<DecodeMetadata>(m, "DecodeMetadata")
        .def_readonly("input_ids", &DecodeMetadata::input_ids)
        .def_readonly("positions", &DecodeMetadata::positions)
        .def_readonly("slot_mapping", &DecodeMetadata::slot_mapping)
        .def_readonly("context_lens_flat", &DecodeMetadata::context_lens_flat)
        .def_readonly("global_context_lens_flat", &DecodeMetadata::global_context_lens_flat)
        .def_readonly("block_tables_flat", &DecodeMetadata::block_tables_flat)
        .def_readonly("max_num_blocks", &DecodeMetadata::max_num_blocks);

    m.def("prepare_prefill_cpp",
          &prepare_prefill_cpp,
          py::arg("seqs"),
          py::arg("engine_id"),
          py::arg("sp_rank"),
          py::arg("sp_size"),
          py::arg("block_size"),
          py::arg("max_num_seqs")
    );

    m.def("prepare_decode_cpp",
          &prepare_decode_cpp,
          py::arg("dp_seqs"),
          py::arg("engine_id"),
          py::arg("sp_rank"),
          py::arg("sp_size"),
          py::arg("block_size"),
          py::arg("max_num_seqs")
    );
}