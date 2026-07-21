use crate::sampling::SamplingParams;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

static NEXT_SEQ_ID: AtomicU64 = AtomicU64::new(0);
static BLOCK_SIZE: AtomicI32 = AtomicI32::new(256);

pub(crate) fn set_sequence_block_size(block_size: i32) {
    BLOCK_SIZE.store(block_size.max(1), Ordering::Relaxed);
}

#[derive(Clone)]
pub(crate) struct VisionSlot {
    pub(crate) encoder_engine_id: String,
    pub(crate) slot_idx: i32,
    pub(crate) num_tokens: i32,
    pub(crate) hidden_size: i32,
    pub(crate) max_tokens_per_slot: i32,
}

pub(crate) struct Sequence {
    pub(crate) seq_id: u64,
    pub(crate) status: i32,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) last_token: i32,
    pub(crate) num_tokens: i32,
    pub(crate) num_prompt_tokens: i32,
    pub(crate) num_checkpointed_tokens: i32,
    pub(crate) num_cached_tokens: i32,
    pub(crate) prefill_start_offset: i32,
    pub(crate) affinity_key: u64,
    pub(crate) sampling_params: SamplingParams,
    pub(crate) completion_logprobs: Vec<f32>,
    pub(crate) vision_slots: Vec<VisionSlot>,
    pub(crate) active_block_table: Vec<i32>,
    pub(crate) active_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) host_block_table: Vec<i32>,
    pub(crate) host_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) last_scheduled_step: u64,
    pub(crate) last_swapped_out_step: u64,
    pub(crate) active_dispatched_tokens: Vec<i32>,
    pub(crate) migrate_block_table: Vec<i32>,
    pub(crate) migrate_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) migrate_engine_id: String,
    pub(crate) migrate_num_kvcache_blocks: i32,
    pub(crate) migrate_group_size: i32,
    pub(crate) migrate_dp_idx: i32,
    pub(crate) active_dp_idx: i32,
    pub(crate) active_group_id: i32,
    pub(crate) migrate_group_id: i32,
    pub(crate) active_state_slot: i32,
    pub(crate) migrate_state_slot: i32,
    pub(crate) active_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) migrate_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) active_hisparse_slot: i32,
    pub(crate) migrate_hisparse_slot: i32,
}

impl Sequence {
    pub(crate) fn new(token_ids: Vec<i32>, sampling_params: Option<SamplingParams>) -> Self {
        let last_token = token_ids.last().copied().unwrap_or(-1);
        let num_tokens = token_ids.len() as i32;
        Self {
            seq_id: NEXT_SEQ_ID.fetch_add(1, Ordering::Relaxed),
            status: 0,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_checkpointed_tokens: num_tokens,
            num_cached_tokens: 0,
            prefill_start_offset: 0,
            affinity_key: 0,
            sampling_params: sampling_params
                .unwrap_or_else(|| SamplingParams::new(1.0, 256, false, false, None, None)),
            completion_logprobs: Vec::new(),
            vision_slots: Vec::new(),
            active_block_table: Vec::new(),
            active_block_tables: HashMap::new(),
            host_block_table: Vec::new(),
            host_block_tables: HashMap::new(),
            last_scheduled_step: 0,
            last_swapped_out_step: 0,
            active_dispatched_tokens: Vec::new(),
            migrate_block_table: Vec::new(),
            migrate_block_tables: HashMap::new(),
            migrate_engine_id: String::new(),
            migrate_num_kvcache_blocks: 0,
            migrate_group_size: 1,
            migrate_dp_idx: 0,
            active_dp_idx: 0,
            active_group_id: 0,
            migrate_group_id: 0,
            active_state_slot: -1,
            migrate_state_slot: -1,
            active_compressed_block_tables: HashMap::new(),
            migrate_compressed_block_tables: HashMap::new(),
            active_hisparse_slot: -1,
            migrate_hisparse_slot: -1,
        }
    }
}
