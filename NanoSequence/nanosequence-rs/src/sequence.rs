use crate::ffi::*;
use std::ffi::{CStr, CString};
use std::ptr;

/// Sequence status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Finished,
    ToBeMigrated,
}

impl From<NanosequenceStatus> for SequenceStatus {
    fn from(status: NanosequenceStatus) -> Self {
        match status {
            NanosequenceStatus::Waiting => SequenceStatus::Waiting,
            NanosequenceStatus::Running => SequenceStatus::Running,
            NanosequenceStatus::Finished => SequenceStatus::Finished,
            NanosequenceStatus::ToBeMigrated => SequenceStatus::ToBeMigrated,
        }
    }
}

impl From<SequenceStatus> for NanosequenceStatus {
    fn from(status: SequenceStatus) -> Self {
        match status {
            SequenceStatus::Waiting => NanosequenceStatus::Waiting,
            SequenceStatus::Running => NanosequenceStatus::Running,
            SequenceStatus::Finished => NanosequenceStatus::Finished,
            SequenceStatus::ToBeMigrated => NanosequenceStatus::ToBeMigrated,
        }
    }
}

/// Block context slot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockContextSlot {
    Active,
    Migrate,
    Swap,
}

impl From<NanosequenceBlockContextSlot> for BlockContextSlot {
    fn from(slot: NanosequenceBlockContextSlot) -> Self {
        match slot {
            NanosequenceBlockContextSlot::Active => BlockContextSlot::Active,
            NanosequenceBlockContextSlot::Migrate => BlockContextSlot::Migrate,
            NanosequenceBlockContextSlot::Swap => BlockContextSlot::Swap,
        }
    }
}

impl From<BlockContextSlot> for NanosequenceBlockContextSlot {
    fn from(slot: BlockContextSlot) -> Self {
        match slot {
            BlockContextSlot::Active => NanosequenceBlockContextSlot::Active,
            BlockContextSlot::Migrate => NanosequenceBlockContextSlot::Migrate,
            BlockContextSlot::Swap => NanosequenceBlockContextSlot::Swap,
        }
    }
}

/// Sampling parameters for sequence generation
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub max_tokens: i32,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            max_tokens: 256,
            ignore_eos: false,
        }
    }
}

/// Block context information
#[derive(Debug, Clone)]
pub struct BlockContext {
    pub engine_id: String,
    pub dp_idx: i32,
    pub master_sp_idx: i32,
    pub attention_sp: i32,
    pub attention_dp: i32,
    pub num_kvcache_blocks: i32,
}

impl BlockContext {
    fn from_raw(ctx: NanosequenceBlockContext) -> Option<Self> {
        if ctx.is_null() {
            return None;
        }

        unsafe {
            let engine_id_cstr = nanosequence_block_context_get_engine_id(ctx);
            let engine_id = if engine_id_cstr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(engine_id_cstr).to_string_lossy().into_owned()
            };

            Some(BlockContext {
                engine_id,
                dp_idx: nanosequence_block_context_get_dp_idx(ctx),
                master_sp_idx: nanosequence_block_context_get_master_sp_idx(ctx),
                attention_sp: nanosequence_block_context_get_attention_sp(ctx),
                attention_dp: nanosequence_block_context_get_attention_dp(ctx),
                num_kvcache_blocks: nanosequence_block_context_get_num_kvcache_blocks(ctx),
            })
        }
    }
}

/// A sequence of tokens with associated metadata
pub struct Sequence {
    inner: NanosequenceSequence,
}

impl Sequence {
    /// Create a new sequence with the given token IDs and sampling parameters
    pub fn new(token_ids: &[i32], params: Option<SamplingParams>) -> Result<Self, String> {
        let params_ptr = if let Some(ref p) = params {
            unsafe { nanosequence_sampling_params_new(p.temperature, p.max_tokens, p.ignore_eos) }
        } else {
            let default = SamplingParams::default();
            unsafe {
                nanosequence_sampling_params_new(
                    default.temperature,
                    default.max_tokens,
                    default.ignore_eos,
                )
            }
        };

        if params_ptr.is_null() {
            return Err("Failed to create sampling parameters".to_string());
        }

        let seq = unsafe {
            nanosequence_sequence_new(
                token_ids.as_ptr(),
                token_ids.len(),
                params_ptr,
            )
        };

        if params.is_some() {
            unsafe {
                nanosequence_sampling_params_free(params_ptr);
            }
        }

        if seq.is_null() {
            return Err("Failed to create sequence".to_string());
        }

        Ok(Sequence { inner: seq })
    }

    /// Get the sequence ID
    pub fn seq_id(&self) -> u64 {
        unsafe { nanosequence_sequence_get_seq_id(self.inner) }
    }

    /// Set the sequence ID
    pub fn set_seq_id(&mut self, seq_id: u64) {
        unsafe {
            nanosequence_sequence_set_seq_id(self.inner, seq_id);
        }
    }

    /// Get the sequence status
    pub fn status(&self) -> SequenceStatus {
        unsafe { nanosequence_sequence_get_status(self.inner).into() }
    }

    /// Set the sequence status
    pub fn set_status(&mut self, status: SequenceStatus) {
        unsafe {
            nanosequence_sequence_set_status(self.inner, status.into());
        }
    }

    /// Get the number of tokens
    pub fn num_tokens(&self) -> i32 {
        unsafe { nanosequence_sequence_get_num_tokens(self.inner) }
    }

    /// Get the number of prompt tokens
    pub fn num_prompt_tokens(&self) -> i32 {
        unsafe { nanosequence_sequence_get_num_prompt_tokens(self.inner) }
    }

    /// Get the last token
    pub fn last_token(&self) -> i32 {
        unsafe { nanosequence_sequence_get_last_token(self.inner) }
    }

    /// Get all token IDs
    pub fn token_ids(&self) -> Vec<i32> {
        let count = unsafe { nanosequence_sequence_get_token_count(self.inner) };
        let mut tokens = vec![0i32; count];
        unsafe {
            nanosequence_sequence_get_token_ids(self.inner, tokens.as_mut_ptr(), count);
        }
        tokens
    }

    /// Append a token to the sequence
    pub fn append_token(&mut self, token_id: i32, slot: BlockContextSlot, sp_idx: Option<i32>) {
        unsafe {
            nanosequence_sequence_append_token(
                self.inner,
                token_id,
                slot.into(),
                sp_idx.unwrap_or(-1),
            );
        }
    }

    /// Activate the sequence with the given engine configuration
    pub fn active(
        &mut self,
        engine_id: &str,
        attention_sp: i32,
        attention_dp: i32,
        num_kvcache_blocks: i32,
    ) -> Result<(), String> {
        let engine_id_cstr = CString::new(engine_id)
            .map_err(|e| format!("Invalid engine_id: {}", e))?;

        let result = unsafe {
            nanosequence_sequence_active(
                self.inner,
                engine_id_cstr.as_ptr(),
                attention_sp,
                attention_dp,
                num_kvcache_blocks,
            )
        };

        if result != 0 {
            Err(format!("Failed to activate sequence: {}", result))
        } else {
            Ok(())
        }
    }

    /// Migrate the sequence
    pub fn migrate(&mut self) -> Result<(), String> {
        let result = unsafe { nanosequence_sequence_migrate(self.inner) };
        if result != 0 {
            Err(format!("Failed to migrate sequence: {}", result))
        } else {
            Ok(())
        }
    }

    /// Get the block context for the given slot
    pub fn block_ctx(&self, slot: BlockContextSlot) -> Option<BlockContext> {
        let ctx = unsafe { nanosequence_sequence_get_block_ctx(self.inner, slot.into()) };
        BlockContext::from_raw(ctx)
    }

    /// Get the raw pointer (for use with serialization)
    pub fn as_ptr(&self) -> NanosequenceSequence {
        self.inner
    }
}

impl Drop for Sequence {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                nanosequence_sequence_free(self.inner);
            }
        }
    }
}

unsafe impl Send for Sequence {}
unsafe impl Sync for Sequence {}
