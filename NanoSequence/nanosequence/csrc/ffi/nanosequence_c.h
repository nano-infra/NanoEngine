#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pointer types
typedef void* NanosequenceSequence;
typedef void* NanosequenceBlockContext;
typedef void* NanosequenceSamplingParams;

// Enums
typedef enum {
    NANOSEQUENCE_STATUS_WAITING        = 0,
    NANOSEQUENCE_STATUS_RUNNING        = 1,
    NANOSEQUENCE_STATUS_FINISHED       = 2,
    NANOSEQUENCE_STATUS_TO_BE_MIGRATED = 3,
} NanosequenceStatus;

typedef enum {
    NANOSEQUENCE_SLOT_ACTIVE  = 0,
    NANOSEQUENCE_SLOT_MIGRATE = 1,
    NANOSEQUENCE_SLOT_SWAP    = 2,
} NanosequenceBlockContextSlot;

// SamplingParams API
NanosequenceSamplingParams nanosequence_sampling_params_new(double temperature, int max_tokens, bool ignore_eos);
void                       nanosequence_sampling_params_free(NanosequenceSamplingParams params);

// Sequence API
NanosequenceSequence
     nanosequence_sequence_new(const int32_t* token_ids, size_t token_count, NanosequenceSamplingParams params);
void nanosequence_sequence_free(NanosequenceSequence seq);

// Sequence properties
uint64_t           nanosequence_sequence_get_seq_id(NanosequenceSequence seq);
void               nanosequence_sequence_set_seq_id(NanosequenceSequence seq, uint64_t seq_id);
NanosequenceStatus nanosequence_sequence_get_status(NanosequenceSequence seq);
void               nanosequence_sequence_set_status(NanosequenceSequence seq, NanosequenceStatus status);
int                nanosequence_sequence_get_num_tokens(NanosequenceSequence seq);
int                nanosequence_sequence_get_num_prompt_tokens(NanosequenceSequence seq);
int                nanosequence_sequence_get_last_token(NanosequenceSequence seq);

// Sequence token operations
void   nanosequence_sequence_get_token_ids(NanosequenceSequence seq, int32_t* out_buffer, size_t buffer_size);
size_t nanosequence_sequence_get_token_count(NanosequenceSequence seq);
void   nanosequence_sequence_append_token(NanosequenceSequence         seq,
                                          int32_t                      token_id,
                                          NanosequenceBlockContextSlot slot,
                                          int                          sp_idx);

// Sequence block context operations
int32_t nanosequence_sequence_active(
    NanosequenceSequence seq, const char* engine_id, int attention_sp, int attention_dp, int num_kvcache_blocks);
int32_t                  nanosequence_sequence_migrate(NanosequenceSequence seq);
NanosequenceBlockContext nanosequence_sequence_get_block_ctx(NanosequenceSequence         seq,
                                                             NanosequenceBlockContextSlot slot);

// BlockContext API
void        nanosequence_block_context_free(NanosequenceBlockContext ctx);
const char* nanosequence_block_context_get_engine_id(NanosequenceBlockContext ctx);
int         nanosequence_block_context_get_dp_idx(NanosequenceBlockContext ctx);
int         nanosequence_block_context_get_master_sp_idx(NanosequenceBlockContext ctx);
int         nanosequence_block_context_get_attention_sp(NanosequenceBlockContext ctx);
int         nanosequence_block_context_get_attention_dp(NanosequenceBlockContext ctx);
int         nanosequence_block_context_get_num_kvcache_blocks(NanosequenceBlockContext ctx);

// Serialization API
size_t nanosequence_serialize_sequences(uintptr_t                   data_ptr,
                                        size_t                      buffer_size,
                                        const NanosequenceSequence* sequences,
                                        size_t                      sequence_count,
                                        bool                        is_prefill);

// Deserialization API - returns array of sequences
// Caller must free the returned sequences using nanosequence_sequence_free
NanosequenceSequence*
nanosequence_deserialize_sequences(uintptr_t data_ptr, size_t data_len, size_t* out_sequence_count);

// Free array of sequences returned by deserialize
void nanosequence_sequences_free(NanosequenceSequence* sequences, size_t count);

#ifdef __cplusplus
}
#endif
