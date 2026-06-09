#pragma once

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "nanodeploy/csrc/sequence/sequence.h"

namespace nanodeploy {

// Serialize a batch of sequences into lean RunBatchInput FlatBuffers.
// For prefill: includes token_ids[num_cached_tokens:]
// For decode: includes last_token, omits token_ids
// Returns a DetachedBuffer owning the serialized data.
flatbuffers::DetachedBuffer serialize_run_batch(const std::vector<Sequence*>& seqs, bool is_prefill);

// Serialize a batch of sequences for migration.
// Extracts only ACTIVE + MIGRATE block context fields.
// Returns a DetachedBuffer owning the serialized data.
flatbuffers::DetachedBuffer serialize_migrate_batch(const std::vector<Sequence*>& seqs);

}  // namespace nanodeploy
