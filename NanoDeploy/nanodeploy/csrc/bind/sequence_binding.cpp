#include <utility>

#include <flatbuffers/flatbuffers.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "nanodeploy/csrc/metrics/sequence_metric.h"
#include "nanodeploy/csrc/sequence/sequence.h"
#include "nanodeploy/csrc/sequence/serialization.h"
#include "sequence_generated.h"

namespace py = pybind11;
using namespace nanodeploy;

void bind_sequence(py::module_& m)
{
    // Bind FlatBuffers BlockLocation struct (read-only)
    py::class_<fbs::BlockLocation>(m, "BlockLocation")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_property_readonly("first", [](const fbs::BlockLocation& bl) { return bl.first(); })
        .def_property_readonly("second", [](const fbs::BlockLocation& bl) { return bl.second(); })
        .def("__repr__", [](const fbs::BlockLocation& bl) {
            return "BlockLocation(first=" + std::to_string(bl.first()) + ", second=" + std::to_string(bl.second())
                   + ")";
        });

    // Directly accepts address and size
    m.def("serialize",
          &serialize_sequences,
          py::arg("data_ptr"),
          py::arg("buffer_size"),
          py::arg("seqs"),
          py::arg("is_prefill"));

    m.def("deserialize", &deserialize_sequences, py::arg("data_ptr"), py::arg("data_len"));

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

    py::class_<BlockContext>(m, "BlockContext")
        .def(py::init<>())
        .def_readwrite("engine_id", &BlockContext::engine_id)
        .def_readwrite("dp_idx", &BlockContext::dp_idx)
        .def_readwrite("master_group_id", &BlockContext::master_group_id)
        .def_readwrite("group_size", &BlockContext::group_size)
        .def_readwrite("attention_dp", &BlockContext::attention_dp)
        .def_readwrite("num_kvcache_blocks", &BlockContext::num_kvcache_blocks)
        .def_readwrite("state_slot", &BlockContext::state_slot)
        .def_property(
            "block_location",
            [](BlockContext& self) -> std::vector<fbs::BlockLocation>& { return self.block_location; },
            [](BlockContext& self, const std::vector<fbs::BlockLocation>& value) { self.block_location = value; },
            py::return_value_policy::reference_internal)
        .def_property(
            "group_block_table",
            [](BlockContext& self) -> std::vector<std::unique_ptr<fbs::IntListT>>& { return self.group_block_table; },
            [](BlockContext& self, const std::vector<std::unique_ptr<fbs::IntListT>>& value) {
                self.group_block_table.clear();
                for (const auto& item : value) {
                    if (item) {
                        auto new_item    = std::make_unique<fbs::IntListT>();
                        new_item->values = item->values;
                        self.group_block_table.push_back(std::move(new_item));
                    }
                    else {
                        self.group_block_table.push_back(std::make_unique<fbs::IntListT>());
                    }
                }
            },
            py::return_value_policy::reference_internal)
        .def_readwrite("num_dispatched_tokens", &BlockContext::num_dispatched_tokens)
        .def(
            "reset",
            [](BlockContext&      self,
               const std::string& engine_id,
               int                group_size,
               int                attention_dp,
               int                num_kvcache_blocks) {
                reset_block_context(self, engine_id, group_size, attention_dp, num_kvcache_blocks);
            },
            py::arg("engine_id"),
            py::arg("group_size"),
            py::arg("attention_dp"),
            py::arg("num_kvcache_blocks"))
        .def(py::pickle(
            [](const BlockContext& ctx) -> py::bytes {
                flatbuffers::FlatBufferBuilder builder(256);
                auto                           offset = fbs::BlockContext::Pack(builder, &ctx);
                builder.Finish(offset);
                return py::bytes(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
            },
            [](py::bytes bytes) -> std::unique_ptr<BlockContext> {
                py::buffer_info info(py::buffer(bytes).request());
                const uint8_t*  buffer = static_cast<const uint8_t*>(info.ptr);

                flatbuffers::Verifier verifier(buffer, info.size);
                if (!verifier.VerifyBuffer<fbs::BlockContext>()) {
                    throw std::runtime_error("Invalid flatbuffer data in BlockContext pickle");
                }

                auto fb_ctx = flatbuffers::GetRoot<fbs::BlockContext>(buffer);
                return std::unique_ptr<BlockContext>(fb_ctx->UnPack());
            }));

    py::class_<SamplingParams>(m, "SamplingParams")
        .def(py::init<>())
        .def(py::init([](double temperature, int32_t max_tokens, bool ignore_eos) {
                 SamplingParams sp;
                 sp.temperature = temperature;
                 sp.max_tokens  = max_tokens;
                 sp.ignore_eos  = ignore_eos;
                 return sp;
             }),
             py::arg("temperature") = 1.0,
             py::arg("max_tokens")  = 256,
             py::arg("ignore_eos")  = false)
        .def_readwrite("temperature", &SamplingParams::temperature)
        .def_readwrite("max_tokens", &SamplingParams::max_tokens)
        .def_readwrite("ignore_eos", &SamplingParams::ignore_eos);

    py::class_<Sequence, std::shared_ptr<Sequence>>(m, "Sequence")
        .def(py::init<const std::vector<int>&, const SamplingParams&>(),
             py::arg("token_ids"),
             py::arg("sampling_params") = SamplingParams())
        .def("active",
             &Sequence::active,
             py::arg("engine_id"),
             py::arg("group_size"),
             py::arg("attention_dp"),
             py::arg("num_kvcache_blocks"))
        .def("migrate", &Sequence::migrate)
        .def("context_len", &Sequence::context_len, py::arg("engine_id"), py::arg("group_id") = std::nullopt)
        .def("append_token",
             &Sequence::append_token,
             py::arg("token_id"),
             py::arg("slot"),
             py::arg("group_id") = std::nullopt)
        .def("block_ctx",
             static_cast<BlockContext& (Sequence::*)(BlockContextSlot)>(&Sequence::block_ctx),
             py::arg("slot") = BlockContextSlot::ACTIVE,
             py::return_value_policy::reference_internal)
        .def("block_table",
             &Sequence::block_table,
             py::arg("slot"),
             py::arg("group_id") = 0,
             py::return_value_policy::reference_internal)
        .def("dp_idx", &Sequence::dp_idx, py::arg("slot"))
        .def("state_slot", &Sequence::state_slot, py::arg("slot") = BlockContextSlot::ACTIVE)
        .def("set_state_slot", &Sequence::set_state_slot, py::arg("slot"), py::arg("state_slot"))
        .def("num_blocks", &Sequence::num_blocks, py::arg("slot"), py::arg("group_id"))
        .def("last_block_page_id", &Sequence::last_block_page_id, py::arg("slot"), py::arg("group_id"))
        .def("last_block_num_tokens", &Sequence::last_block_num_tokens, py::arg("slot"), py::arg("group_id"))
        .def("block", &Sequence::block, py::arg("i"), py::arg("slot"), py::arg("group_id"))

        .def_property("seq_id", &Sequence::seq_id, &Sequence::set_seq_id)
        .def_property("status", &Sequence::status, &Sequence::set_status)
        .def_property(
            "token_ids",
            [](const Sequence& s) -> std::vector<int> {
                // Return by value (copy) to avoid aliasing issues during serialization
                return s.token_ids();
            },
            [](Sequence& s, const std::vector<int>& v) { s.token_ids() = v; })
        .def_property("last_token", &Sequence::last_token, &Sequence::set_last_token)
        .def_property("num_tokens", &Sequence::num_tokens, &Sequence::set_num_tokens)
        .def_property("num_prompt_tokens", &Sequence::num_prompt_tokens, &Sequence::set_num_prompt_tokens)
        .def_property(
            "num_checkpointed_tokens", &Sequence::num_checkpointed_tokens, &Sequence::set_num_checkpointed_tokens)
        .def_property("num_cached_tokens", &Sequence::num_cached_tokens, &Sequence::set_num_cached_tokens)
        .def_readwrite("metric", &Sequence::metric)
        .def_property("sampling_params", &Sequence::sampling_params, &Sequence::set_sampling_params)

        .def_property_readonly("is_finished", &Sequence::is_finished)
        .def_property_readonly("is_to_be_migrated", &Sequence::is_to_be_migrated)
        .def_property_readonly("num_completed_tokens", &Sequence::num_completed_tokens)
        .def_property_readonly("num_generated_tokens_since_checkpoint",
                               &Sequence::num_generated_tokens_since_checkpoint)
        .def_property_readonly("prompt_token_ids", &Sequence::prompt_token_ids)
        .def_property_readonly("completion_token_ids", &Sequence::completion_token_ids)
        .def_property_readonly("num_cached_blocks", &Sequence::num_cached_blocks)

        .def("__len__", [](const Sequence& s) { return s.num_tokens(); })
        .def("__getitem__",
             [](const Sequence& s, py::object key) -> py::object {
                 const auto& token_ids = s.token_ids();
                 if (py::isinstance<py::slice>(key)) {
                     py::slice slice_obj = key.cast<py::slice>();
                     size_t    start, stop, step, slicelength;
                     if (!slice_obj.compute(token_ids.size(), &start, &stop, &step, &slicelength)) {
                         throw py::error_already_set();
                     }
                     std::vector<int> result;
                     result.reserve(slicelength);
                     for (size_t i = 0; i < slicelength; ++i) {
                         result.push_back(token_ids[start]);
                         start += step;
                     }
                     return py::cast(result);
                 }
                 else {
                     int idx = key.cast<int>();
                     if (idx < 0)
                         idx += token_ids.size();
                     if (idx < 0 || idx >= static_cast<int>(token_ids.size()))
                         throw py::index_error();
                     return py::cast(token_ids[idx]);
                 }
             })

        // Vision slot management (EP separated mode)
        .def("add_vision_slot",
             &Sequence::add_vision_slot,
             py::arg("encoder_engine_id"),
             py::arg("slot_idx"),
             py::arg("num_tokens"),
             py::arg("hidden_size"),
             py::arg("max_tokens_per_slot"))
        .def("clear_vision_slots", &Sequence::clear_vision_slots)
        .def_property_readonly("vision_slots",
                               [](const Sequence& s) {
                                   py::list result;
                                   for (const auto& vs : s.vision_slots()) {
                                       if (!vs)
                                           continue;
                                       py::dict d;
                                       d["encoder_engine_id"]   = vs->encoder_engine_id;
                                       d["slot_idx"]            = vs->slot_idx;
                                       d["num_tokens"]          = vs->num_tokens;
                                       d["hidden_size"]         = vs->hidden_size;
                                       d["max_tokens_per_slot"] = vs->max_tokens_per_slot;
                                       result.append(d);
                                   }
                                   return result;
                               })

        .def(py::pickle(
            [](const Sequence& seq) -> py::bytes {  // __getstate__
                try {
                    // Safety check: ensure data_ is valid before serialization
                    if (!seq.data_) {
                        throw std::runtime_error("Cannot pickle Sequence: data_ pointer is null");
                    }

                    // Full validation matching serialize_sequences() in serialization.cpp
                    for (size_t i = 0; i < seq.data_->slots.size(); ++i) {
                        auto& slot = seq.data_->slots[i];
                        if (slot) {
                            // Ensure engine_id is valid
                            // (no-op if already set, but guards against uninitialized memory)

                            // Ensure num_dispatched_tokens matches group_size
                            if (slot->num_dispatched_tokens.size() != static_cast<size_t>(slot->group_size)) {
                                std::cerr << "  slot[" << i << "]: fixing num_dispatched_tokens size "
                                          << slot->num_dispatched_tokens.size() << " -> " << slot->group_size
                                          << std::endl;
                                slot->num_dispatched_tokens.resize(slot->group_size, 0);
                            }

                            // Ensure group_block_table matches group_size and has no null pointers
                            if (slot->group_block_table.size() != static_cast<size_t>(slot->group_size)) {
                                std::cerr << "  slot[" << i << "]: fixing group_block_table size "
                                          << slot->group_block_table.size() << " -> " << slot->group_size << std::endl;
                                slot->group_block_table.resize(slot->group_size);
                            }
                            for (size_t j = 0; j < slot->group_block_table.size(); ++j) {
                                if (!slot->group_block_table[j]) {
                                    std::cerr << "  slot[" << i << "]: fixing null group_block_table[" << j << "]"
                                              << std::endl;
                                    slot->group_block_table[j] = std::make_unique<fbs::IntListT>();
                                }
                            }
                        }
                        else {
                            // Initialize null slot with safe defaults (matches serialize_sequences)
                            slot                     = std::make_unique<fbs::BlockContextT>();
                            slot->engine_id          = "";
                            slot->dp_idx             = 0;
                            slot->master_group_id    = 0;
                            slot->group_size         = 0;
                            slot->attention_dp       = 0;
                            slot->num_kvcache_blocks = 0;
                        }
                    }

                    flatbuffers::FlatBufferBuilder builder(64 * 1024);  // 64KB initial
                    auto                           offset = fbs::Sequence::Pack(builder, seq.data_.get());
                    builder.Finish(offset);

                    return py::bytes(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
                }
                catch (const std::exception& e) {
                    std::cerr << "=== PICKLE EXCEPTION seq_id=" << seq.seq_id() << ": " << e.what()
                              << " ===" << std::endl;
                    throw std::runtime_error(std::string("Sequence pickle failed for seq_id=")
                                             + std::to_string(seq.seq_id()) + ": " + e.what());
                }
            },
            [](py::bytes bytes) -> std::shared_ptr<Sequence> {  // __setstate__
                // Simplified: Use UnPack() directly
                try {
                    py::buffer_info info(py::buffer(bytes).request());
                    const uint8_t*  buffer = static_cast<const uint8_t*>(info.ptr);

                    if (!buffer || info.size == 0) {
                        throw std::runtime_error("Invalid buffer: null or empty");
                    }

                    flatbuffers::Verifier verifier(buffer, info.size);
                    if (!verifier.VerifyBuffer<fbs::Sequence>()) {
                        throw std::runtime_error("Invalid flatbuffer data in Sequence pickle");
                    }

                    auto fb_seq   = flatbuffers::GetRoot<fbs::Sequence>(buffer);
                    auto unpacked = fb_seq->UnPack();
                    if (!unpacked) {
                        throw std::runtime_error("Failed to unpack Sequence from flatbuffer");
                    }

                    return Sequence::from_data(std::unique_ptr<fbs::SequenceT>(unpacked));
                }
                catch (const std::exception& e) {
                    throw std::runtime_error(std::string("Sequence deserialization failed: ") + e.what());
                }
            }))

        .def_readwrite_static("block_size", &Sequence::block_size)
        .def_static("set_block_size", &Sequence::set_block_size, py::arg("block_size"));
}
