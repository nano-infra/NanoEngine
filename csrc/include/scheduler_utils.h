#pragma once
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sp_state_manager.h"

namespace nanodeploy {

namespace py = pybind11;

void postprocess_sequences(
    std::vector<SPStateManager*> worker_states,
    py::list dp_seqs,
    py::list dp_token_ids,
    const std::string& engine_id,
    int eos_id,
    bool is_prefill,
    py::dict to_be_migrated,
    bool update_metrics
);

} // namespace nanodeploy