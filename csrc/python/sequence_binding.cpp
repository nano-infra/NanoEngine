#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "nanodeploy/metrics/sequence_metric.h"

#include "nanodeploy/sequence/sequence.h"
#include "nanodeploy/sequence/serialization.h"

#include "opaque_types.h"

namespace py = pybind11;
using namespace nanodeploy;

namespace {

inline constexpr uint64_t kSequencePickleMagic   = 0x4E44534551504B4CULL;  // "NDSEQPKL"
inline constexpr uint32_t kSequencePickleVersion = 1;

using SequencePickleState =
    std::tuple<uint64_t,
               uint32_t,
               uint64_t,
               SequenceStatus,
               int,
               std::vector<int>,
               int,
               int,
               int,
               int,
               int,
               std::array<BlockContext, (size_t)BlockContextSlot::_COUNT>,
               std::shared_ptr<SequenceMetric>,
               double,
               int,
               bool>;

BlockContext::BlockIdList block_id_list_from_iterable(const py::iterable& it)
{
    BlockContext::BlockIdList out;
    for (auto item : it) {
        out.push_back(item.cast<int>());
    }
    return out;
}

void validate_sequence_pickle_state(const SequencePickleState& state)
{
    if (std::get<0>(state) != kSequencePickleMagic || std::get<1>(state) != kSequencePickleVersion) {
        throw py::value_error("unsupported Sequence pickle schema");
    }
    if (!is_valid_sequence_status(std::get<3>(state))) {
        throw py::value_error("invalid SequenceStatus in pickle state");
    }
    if (std::get<4>(state) < -1) {
        throw py::value_error("invalid assigned_dp in pickle state");
    }

    const int num_tokens              = std::get<7>(state);
    const int num_prompt_tokens       = std::get<8>(state);
    const int num_checkpointed_tokens = std::get<9>(state);
    const int num_cached_tokens       = std::get<10>(state);
    if (num_tokens < 0 || num_prompt_tokens < 0 || num_prompt_tokens > num_tokens
        || num_checkpointed_tokens < 0 || num_checkpointed_tokens > num_tokens || num_cached_tokens < 0
        || num_cached_tokens > num_tokens || std::get<14>(state) < 0) {
        throw py::value_error("invalid token counters in Sequence pickle state");
    }
    const auto& token_ids = std::get<5>(state);
    if (token_ids.size() != static_cast<size_t>(num_tokens)) {
        throw py::value_error("Sequence pickle state does not contain the complete token history");
    }
    if (!token_ids.empty() && token_ids.back() != std::get<6>(state)) {
        throw py::value_error("Sequence pickle last_token does not match the token history");
    }
    const auto& metric = std::get<12>(state);
    if (metric && metric->seq_id != std::get<2>(state)) {
        throw py::value_error("Sequence metric identity does not match pickle state");
    }
    for (const auto& context : std::get<11>(state)) {
        try {
            validate_serializable_block_context(context);
        }
        catch (const std::exception& error) {
            throw py::value_error(std::string("invalid BlockContext in Sequence pickle state: ") + error.what());
        }
    }
    try {
        validate_serializable_sequence_context_ownership(
            std::get<4>(state), std::get<3>(state), std::get<11>(state)[(size_t)BlockContextSlot::ACTIVE]);
    }
    catch (const std::exception& error) {
        throw py::value_error(std::string("invalid Sequence pickle ownership: ") + error.what());
    }
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
          py::arg("is_prefill"),
          py::arg("sp_rank") = -1,
          py::arg("sp_size") = -1);

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
        .value("PAUSED_OFFLOAD", SequenceStatus::PAUSED_OFFLOAD)
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
        .def_readwrite("pending_token_present", &BlockContext::pending_token_present_)
        .def_readwrite("pending_token_target_sp", &BlockContext::pending_token_target_sp_)
        .def_property(
            "num_dispatched_tokens",
            [](const BlockContext& self) { return self.num_dispatched_tokens; },
            [](BlockContext& self, const std::vector<int>& value) { self.num_dispatched_tokens = value; })
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
        .def("reset", &BlockContext::reset, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def(py::pickle([](const BlockContext& p) {
                            validate_serializable_block_context(p);
                            return p.getstate();
                        },
                        [](const std::tuple<std::string,
                                            int,
                                            int,
                                            int,
                                            int,
                                            bool,
                                            int,
                                            std::vector<std::pair<int, int>>,
                                            std::vector<std::vector<int>>,
                                            std::vector<int>>& t) {
                            auto context = BlockContext::setstate(t);
                            validate_serializable_block_context(context);
                            return context;
                        }));

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, double, int, bool>(),
             py::arg("token_ids"),
             py::arg("temperature") = 1.0,
             py::arg("max_tokens")  = 256,
             py::arg("ignore_eos")  = false)
        .def("active", &Sequence::active, py::arg("engine_id"), py::arg("attention_sp"), py::arg("attention_dp"))
        .def("migrate", &Sequence::migrate)
        .def("context_len", &Sequence::context_len, py::arg("engine_id"), py::arg("sp_idx") = std::nullopt)
        .def("append_token",
             &Sequence::append_token,
             py::arg("token_id"),
             py::arg("slot"),
             py::arg("sp_idx") = std::nullopt)
        .def("mark_last_token_pending",
             &Sequence::mark_last_token_pending,
             py::arg("slot")   = BlockContextSlot::ACTIVE,
             py::arg("sp_idx") = std::nullopt)
        .def("committed_context_len", &Sequence::committed_context_len, py::arg("slot"), py::arg("sp_idx"))
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

        .def_property(
            "seq_id",
            [](const Sequence& self) { return self.seq_id; },
            [](Sequence& self, uint64_t seq_id) {
                if (self.assigned_dp != -1) {
                    throw py::value_error("cannot modify seq_id after assigned_dp is set");
                }
                self.seq_id = seq_id;
            })
        .def_readwrite("status", &Sequence::status)
        .def_property(
            "assigned_dp",
            [](const Sequence& self) { return self.assigned_dp; },
            [](Sequence& self, int assigned_dp) {
                if (assigned_dp < -1) {
                    throw py::value_error("assigned_dp must be -1 or a non-negative DP index");
                }
                if (self.assigned_dp != -1 && assigned_dp != self.assigned_dp) {
                    throw py::value_error("cannot modify assigned_dp after it is set");
                }
                self.assigned_dp = assigned_dp;
            })
        .def_readwrite("token_ids", &Sequence::token_ids)
        .def_readwrite("last_token", &Sequence::last_token)
        .def_readwrite("num_tokens", &Sequence::num_tokens)
        .def_readwrite("num_prompt_tokens", &Sequence::num_prompt_tokens)
        .def_readwrite("num_checkpointed_tokens", &Sequence::num_checkpointed_tokens)
        .def_readwrite("num_cached_tokens", &Sequence::num_cached_tokens)
        .def_readwrite("metric", &Sequence::metric)
        .def_readwrite("temperature", &Sequence::temperature)
        .def_readwrite("max_tokens", &Sequence::max_tokens)
        .def_readwrite("ignore_eos", &Sequence::ignore_eos)

        .def_property_readonly("is_finished", &Sequence::is_finished)
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
            [](const Sequence& p) -> py::tuple {  // __getstate__
                SequencePickleState state{kSequencePickleMagic,
                                          kSequencePickleVersion,
                                          p.seq_id,
                                          p.status,
                                          p.assigned_dp,
                                          p.token_ids,
                                          p.last_token,
                                          p.num_tokens,
                                          p.num_prompt_tokens,
                                          p.num_checkpointed_tokens,
                                          p.num_cached_tokens,
                                          p.slots_,
                                          p.metric,
                                          p.temperature,
                                          p.max_tokens,
                                          p.ignore_eos};
                validate_sequence_pickle_state(state);
                return py::cast(state).cast<py::tuple>();
            },
            [](py::tuple raw_state) {  // __setstate__
                if (raw_state.size() != std::tuple_size_v<SequencePickleState>) {
                    throw py::value_error("unsupported Sequence pickle schema");
                }
                SequencePickleState state;
                try {
                    state = raw_state.cast<SequencePickleState>();
                }
                catch (const py::cast_error&) {
                    throw py::value_error("invalid field types in Sequence pickle schema");
                }
                validate_sequence_pickle_state(state);

                auto seq = std::make_shared<Sequence>(
                    std::get<5>(state), std::get<13>(state), std::get<14>(state), std::get<15>(state));
                seq->restore_seq_id(std::get<2>(state));
                seq->status                  = std::get<3>(state);
                seq->assigned_dp             = std::get<4>(state);
                seq->last_token              = std::get<6>(state);
                seq->num_tokens              = std::get<7>(state);
                seq->num_prompt_tokens       = std::get<8>(state);
                seq->num_checkpointed_tokens = std::get<9>(state);
                seq->num_cached_tokens       = std::get<10>(state);
                seq->slots_                  = std::get<11>(state);
                seq->metric                  = std::get<12>(state);
                return seq;
            }))

        .def_readonly_static("block_size", &Sequence::block_size);
}
