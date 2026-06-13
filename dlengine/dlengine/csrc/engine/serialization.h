#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <flatbuffers/flatbuffers.h>

#include "dlengine/csrc/sequence/sequence.h"

namespace dlengine {

// Serialize a batch of sequences into lean RunBatchInput FlatBuffers.
// For prefill: includes token_ids[num_cached_tokens:]
// For decode: includes last_token, omits token_ids
// Returns a DetachedBuffer owning the serialized data.
flatbuffers::DetachedBuffer serialize_run_batch(const std::vector<Sequence*>& seqs, bool is_prefill);

// Serialize a batch of sequences for migration.
// Extracts only ACTIVE + MIGRATE block context fields.
// Returns a DetachedBuffer owning the serialized data.
flatbuffers::DetachedBuffer serialize_migrate_batch(const std::vector<Sequence*>& seqs);

// ========== DLSlime run_batch reply (RunBatchOutput) ==========
//
// Wire format (identical to the Python implementation it replaces, see
// dlengine/engine/dlslime_protocol.py): an 8-byte little-endian uint64
// holding the server-side handler duration in nanoseconds, followed by a
// RunBatchOutput FlatBuffer (interface.fbs). Per-seq token_ids vectors are
// kept so multi-token replies (MTP / future CP) need no schema change.

inline constexpr size_t kRunResultHeaderSize = 8;

struct RunResultView {
    std::vector<std::vector<int32_t>> token_ids;
    std::vector<std::vector<float>>   logprobs;  // parallel to token_ids when has_logprobs
    bool                              has_logprobs = false;
};

// Encode one step's worker result into reply bytes (header + FlatBuffer).
// ``logprobs`` may be nullptr (no logprobs requested); when shorter than
// token_ids the missing tail entries are simply omitted, mirroring the
// Python encoder's bounds guard.
std::string encode_run_result(const std::vector<std::vector<int32_t>>& token_ids,
                              const std::vector<std::vector<float>>*   logprobs,
                              uint64_t                                 server_handler_ns);

// Decode reply bytes produced by encode_run_result (the 8-byte header is
// skipped; it stays accessible via unpack_reply_header on the Python side).
RunResultView decode_run_result(const uint8_t* data, size_t len);

}  // namespace dlengine
