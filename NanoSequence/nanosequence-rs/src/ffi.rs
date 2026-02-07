#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use libc::{c_char, c_int, c_uint, c_void, size_t, uintptr_t};

// Opaque pointer types
pub type NanosequenceSequence = *mut c_void;
pub type NanosequenceBlockContext = *mut c_void;
pub type NanosequenceSamplingParams = *mut c_void;

// Enums
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanosequenceStatus {
    Waiting = 0,
    Running = 1,
    Finished = 2,
    ToBeMigrated = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanosequenceBlockContextSlot {
    Active = 0,
    Migrate = 1,
    Swap = 2,
}

extern "C" {
    // SamplingParams API
    pub fn nanosequence_sampling_params_new(
        temperature: f64,
        max_tokens: c_int,
        ignore_eos: bool,
    ) -> NanosequenceSamplingParams;
    pub fn nanosequence_sampling_params_free(params: NanosequenceSamplingParams);

    // Sequence API
    pub fn nanosequence_sequence_new(
        token_ids: *const i32,
        token_count: size_t,
        params: NanosequenceSamplingParams,
    ) -> NanosequenceSequence;
    pub fn nanosequence_sequence_free(seq: NanosequenceSequence);

    // Sequence properties
    pub fn nanosequence_sequence_get_seq_id(seq: NanosequenceSequence) -> u64;
    pub fn nanosequence_sequence_set_seq_id(seq: NanosequenceSequence, seq_id: u64);
    pub fn nanosequence_sequence_get_status(seq: NanosequenceSequence) -> NanosequenceStatus;
    pub fn nanosequence_sequence_set_status(seq: NanosequenceSequence, status: NanosequenceStatus);
    pub fn nanosequence_sequence_get_num_tokens(seq: NanosequenceSequence) -> c_int;
    pub fn nanosequence_sequence_get_num_prompt_tokens(seq: NanosequenceSequence) -> c_int;
    pub fn nanosequence_sequence_get_last_token(seq: NanosequenceSequence) -> c_int;

    // Sequence token operations
    pub fn nanosequence_sequence_get_token_ids(
        seq: NanosequenceSequence,
        out_buffer: *mut i32,
        buffer_size: size_t,
    );
    pub fn nanosequence_sequence_get_token_count(seq: NanosequenceSequence) -> size_t;
    pub fn nanosequence_sequence_append_token(
        seq: NanosequenceSequence,
        token_id: i32,
        slot: NanosequenceBlockContextSlot,
        sp_idx: c_int,
    );

    // Sequence block context operations
    pub fn nanosequence_sequence_active(
        seq: NanosequenceSequence,
        engine_id: *const c_char,
        attention_sp: c_int,
        attention_dp: c_int,
        num_kvcache_blocks: c_int,
    ) -> i32;
    pub fn nanosequence_sequence_migrate(seq: NanosequenceSequence) -> i32;
    pub fn nanosequence_sequence_get_block_ctx(
        seq: NanosequenceSequence,
        slot: NanosequenceBlockContextSlot,
    ) -> NanosequenceBlockContext;

    // BlockContext API
    pub fn nanosequence_block_context_free(ctx: NanosequenceBlockContext);
    pub fn nanosequence_block_context_get_engine_id(
        ctx: NanosequenceBlockContext,
    ) -> *const c_char;
    pub fn nanosequence_block_context_get_dp_idx(ctx: NanosequenceBlockContext) -> c_int;
    pub fn nanosequence_block_context_get_master_sp_idx(
        ctx: NanosequenceBlockContext,
    ) -> c_int;
    pub fn nanosequence_block_context_get_attention_sp(
        ctx: NanosequenceBlockContext,
    ) -> c_int;
    pub fn nanosequence_block_context_get_attention_dp(
        ctx: NanosequenceBlockContext,
    ) -> c_int;
    pub fn nanosequence_block_context_get_num_kvcache_blocks(
        ctx: NanosequenceBlockContext,
    ) -> c_int;

    // Serialization API
    pub fn nanosequence_serialize_sequences(
        data_ptr: uintptr_t,
        buffer_size: size_t,
        sequences: *const NanosequenceSequence,
        sequence_count: size_t,
        is_prefill: bool,
    ) -> size_t;

    // Deserialization API
    pub fn nanosequence_deserialize_sequences(
        data_ptr: uintptr_t,
        data_len: size_t,
        out_sequence_count: *mut size_t,
    ) -> *mut NanosequenceSequence;

    pub fn nanosequence_sequences_free(sequences: *mut NanosequenceSequence, count: size_t);
}
