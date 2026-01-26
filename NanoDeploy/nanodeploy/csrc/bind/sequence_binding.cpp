#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "nanodeploy/csrc/metrics/sequence_metric.h"

#include "nanodeploy/csrc/sequence/sequence.h"
#include "nanodeploy/csrc/sequence/serialization.h"

#include "opaque_types.h"

namespace py = pybind11;
using namespace nanodeploy;

namespace {

BlockContext::BlockIdList block_id_list_from_iterable(const py::iterable& it)
{
    BlockContext::BlockIdList out;
    for (auto item : it) {
        out.push_back(item.cast<int>());
    }
    return out;
}

}  // namespace

void bind_sequence(py::module_& m)
{
    // Bind wrapper containers used for mutable proxy views.
    // These are intentionally distinct from std::vector<int> used by token_ids.
    auto block_id_list = py::bind_vector<BlockContext::BlockIdList>(m, "BlockIdList");
    block_id_list.def(py::init<>()).def(py::init([](py::iterable it) { return block_id_list_from_iterable(it); }));
    py::implicitly_convertible<py::list, BlockContext::BlockIdList>();

    py::bind_vector<BlockContext::BlockLocationList>(m, "BlockLocationList").def(py::init<>());

    // Directly accepts address and size
    m.def("serialize",
          &serialize_sequences,
          py::arg("data_ptr"),
          py::arg("buffer_size"),
          py::arg("seqs"),
          py::arg("is_prefill"));

    m.def("deserialize", &deserialize_sequences, py::arg("data_ptr"), py::arg("data_len"));

    // Wrapper class for sp_block_table to provide defaultdict(list) behavior.
    py::class_<BlockContext::SpBlockTable>(m, "DefaultListDict")
        .def(py::init<>())
        .def(
            "__getitem__",
            [](BlockContext::SpBlockTable& self, int key) -> BlockContext::BlockIdList& {
                // Mimic defaultdict(list): create empty list for missing keys.
                return self[key];
            },
            py::return_value_policy::reference_internal)
        .def("__setitem__", [](BlockContext::SpBlockTable& self, int key, py::iterable value) {
            self[key] = block_id_list_from_iterable(value);
        });

    py::enum_<SequenceStatus>(m, "SequenceStatus")
        .value("WAITING", SequenceStatus::WAITING)
        .value("RUNNING", SequenceStatus::RUNNING)
        .value("FINISHED", SequenceStatus::FINISHED)
        .value("TO_BE_MIGRATED", SequenceStatus::TO_BE_MIGRATED)
        .export_values();

    py::enum_<BlockContextSlot>(m, "BlockContextSlot")
        .value("ACTIVE", BlockContextSlot::ACTIVE)
        .value("MIGRATE", BlockContextSlot::MIGRATE)
        .value("SWAP", BlockContextSlot::SWAP)
        .export_values();

    // Wrapper class for num_dispatched_tokens to provide defaultdict(int) behavior
    py::class_<std::unordered_map<int, int>>(m, "DefaultIntDict")
        .def(py::init<>())
        .def(
            "__getitem__",
            [](std::unordered_map<int, int>& self, int key) -> int& {
                // This mimics defaultdict(int) - returns 0 for missing keys and creates entry
                return self[key];
            },
            py::return_value_policy::reference_internal)
        .def("__setitem__", [](std::unordered_map<int, int>& self, int key, int value) { self[key] = value; })
        .def("__contains__",
             [](const std::unordered_map<int, int>& self, int key) { return self.find(key) != self.end(); })
        .def("keys",
             [](const std::unordered_map<int, int>& self) {
                 py::list keys;
                 for (const auto& pair : self) {
                     keys.append(pair.first);
                 }
                 return keys;
             })
        .def("items",
             [](const std::unordered_map<int, int>& self) {
                 py::list items;
                 for (const auto& pair : self) {
                     items.append(py::make_tuple(pair.first, pair.second));
                 }
                 return items;
             })
        .def("clear", [](std::unordered_map<int, int>& self) { self.clear(); })
        .def(py::pickle(
            [](const std::unordered_map<int, int>& self) {
                py::dict d;
                for (const auto& pair : self) {
                    d[py::cast(pair.first)] = py::cast(pair.second);
                }
                return d;
            },
            [](py::dict d) {
                std::unordered_map<int, int> self;
                for (auto item : d) {
                    self[item.first.cast<int>()] = item.second.cast<int>();
                }
                return self;
            }));

    py::class_<BlockContext>(m, "BlockContext")
        .def(py::init<>())
        .def_readwrite("engine_id", &BlockContext::engine_id_)
        .def_readwrite("dp_idx", &BlockContext::dp_idx_)
        .def_readwrite("master_sp_idx", &BlockContext::master_sp_idx_)
        .def_readwrite("attention_sp", &BlockContext::attention_sp_)
        .def_readwrite("attention_dp", &BlockContext::attention_dp_)
        .def_property(
            "block_location",
            [](BlockContext& self) -> BlockContext::BlockLocationList& { return self.block_location; },
            [](BlockContext& self, const BlockContext::BlockLocationList& value) { self.block_location = value; },
            py::return_value_policy::reference_internal)
        .def_property(
            "sp_block_table",
            [](BlockContext& self) -> BlockContext::SpBlockTable& { return self.sp_block_table; },
            [](BlockContext& self, const BlockContext::SpBlockTable& value) { self.sp_block_table = value; },
            py::return_value_policy::reference_internal)
        .def_readwrite("num_dispatched_tokens", &BlockContext::num_dispatched_tokens)
        .def("reset", &BlockContext::reset, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def(py::pickle([](const BlockContext& p) { return p.getstate(); },
                        [](const std::tuple<std::string,
                                            int,
                                            int,
                                            int,
                                            int,
                                            std::vector<std::pair<int, int>>,
                                            std::vector<std::vector<int>>,
                                            std::vector<int>>& t) { return BlockContext::setstate(t); }));

    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init<>())
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("max_tokens", &SamplingParams::max_tokens)
        .def_readwrite("ignore_eos", &SamplingParams::ignore_eos);

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, const SamplingParams&>(),
             py::arg("token_ids"),
             py::arg("sampling_params") = SamplingParams())
        .def("active", &Sequence::active, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def("migrate", &Sequence::migrate)
        .def("context_len", &Sequence::context_len, py::arg("engine_id"), py::arg("sp_idx") = std::nullopt)
        .def("append_token",
             &Sequence::append_token,
             py::arg("token_id"),
             py::arg("slot"),
             py::arg("sp_idx") = std::nullopt)
        .def("block_ctx",
             static_cast<BlockContext& (Sequence::*)(BlockContextSlot)>(&Sequence::block_ctx),
             py::arg("slot") = BlockContextSlot::ACTIVE,
             py::return_value_policy::reference_internal)
        .def("block_table",
             &Sequence::block_table,
             py::arg("slot"),
             py::arg("sp_idx") = 0,
             py::return_value_policy::reference_internal)
        .def("dp_idx", &Sequence::dp_idx, py::arg("slot"))
        .def("num_blocks", &Sequence::num_blocks, py::arg("slot"), py::arg("sp_idx"))
        .def("last_block_page_id", &Sequence::last_block_page_id, py::arg("slot"), py::arg("sp_idx"))
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens, py::arg("slot"), py::arg("sp_idx"))
        .def("block", &Sequence::block, py::arg("i"), py::arg("slot"), py::arg("sp_idx"))

        .def_readwrite("seq_id", &Sequence::seq_id)
        .def_readwrite("status", &Sequence::status)
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_checkpointed_tokens", &Sequence::num_checkpointed_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("metric", &Sequence::metric)
        .def_readwrite("sampling_params", &Sequence::sampling_params)

        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def_property_readonly("is_to_be_migrated", &Sequence::is_to_be_migrated)
        .def_property_readonly("num_completed_tokens", &Sequence::num_completed_tokens)
        .def_property_readonly("num_generated_tokens_since_checkpoint",
                               &Sequence::num_generated_tokens_since_checkpoint)
        .def_property_readonly("prompt_token_ids", &Sequence::prompt_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("num_cached_blocks", &Sequence::num_cached_blocks)

        .def("__len__", [](const Sequence& s) { return s.num_tokens; })
        .def("__getitem__",
             [](const Sequence& s, py::object key) -> py::object {
                 if (py::isinstance<py::slice>(key)) {
                     py::slice slice_obj = key.cast<py::slice>();
                     size_t    start, stop, step, slicelength;
                     if (!slice_obj.compute(s.token_ids.size(), &start, &stop, &step, &slicelength)) {
                         throw py::error_already_set();
                     }
                     std::vector<int> result;
                     result.reserve(slicelength);
                     for (size_t i = 0; i < slicelength; ++i) {
                         result.push_back(s.token_ids[start]);
                         start += step;
                     }
                     return py::cast(result);
                 }
                 else {
                     int idx = key.cast<int>();
                     if (idx < 0)
                         idx += s.token_ids.size();
                     if (idx < 0 || idx >= static_cast<int>(s.token_ids.size()))
                         throw py::index_error();
                     return py::cast(s.token_ids[idx]);
                 }
             })

        .def(py::pickle(
            [](const Sequence& p) {  // __getstate__
                // Always serialize full token_ids to ensure correct state recovery during migration
                return std::make_tuple(p.num_tokens,
                                       p.num_checkpointed_tokens,
                                       p.num_cached_tokens,
                                       p.slots_,
                                       p.sampling_params.temperature,
                                       p.token_ids,
                                       p.status,
                                       p.seq_id,
                                       p.num_prompt_tokens,
                                       p.sampling_params.max_tokens,
                                       p.sampling_params.ignore_eos);
            },
            [](const std::tuple<int,
                                int,
                                int,
                                std::array<BlockContext, (size_t)BlockContextSlot::_COUNT>,
                                double,
                                std::vector<int>,
                                SequenceStatus,
                                uint64_t,
                                int,
                                int,
                                bool>& t) {  // __setstate__
                // Extract token_ids (always present now)
                std::vector<int> token_ids = std::get<5>(t);

                // Restore SamplingParams
                SamplingParams sp;
                sp.temperature = std::get<4>(t);
                sp.max_tokens  = std::get<9>(t);
                sp.ignore_eos  = std::get<10>(t);

                // Reconstruct Sequence with full token history
                auto seq = std::make_shared<Sequence>(token_ids, sp);

                // Restore other fields
                seq->num_tokens              = std::get<0>(t);
                seq->num_checkpointed_tokens = std::get<1>(t);
                seq->num_cached_tokens       = std::get<2>(t);
                seq->slots_                  = std::move(std::get<3>(t));
                seq->status                  = std::get<6>(t);
                seq->seq_id                  = std::get<7>(t);
                seq->num_prompt_tokens       = std::get<8>(t);

                // Ensure last_token is consistent if token_ids is not empty
                if (!seq->token_ids.empty()) {
                    seq->last_token = seq->token_ids.back();
                }

                return seq;
            }))

        .def_readonly_static("block_size", &Sequence::block_size);
}
