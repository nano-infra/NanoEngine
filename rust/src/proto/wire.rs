use crate::sampling::SamplingParams;
use crate::sequence::{Sequence, VisionSlot};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireBatch {
    pub(super) is_prefill: bool,
    pub(super) input_ids: Vec<i64>,
    pub(super) positions: Vec<i64>,
    pub(super) seq_lens: Vec<i32>,
    pub(super) block_tables: Vec<Vec<i32>>,
    pub(super) temperatures: Vec<f32>,
    pub(super) state_slots: Vec<i64>,
    pub(super) compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>>,
    pub(super) hisparse_slots: Vec<i64>,
    pub(super) seq_ids: Vec<u64>,
    pub(super) sample_mask: Vec<bool>,
    pub(super) any_return_completion_logprobs: bool,
    pub(super) is_dummy: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WireSamplingParams {
    pub(crate) temperature: f64,
    pub(crate) max_tokens: i32,
    pub(crate) ignore_eos: bool,
    pub(crate) return_completion_logprobs: bool,
}

impl From<&SamplingParams> for WireSamplingParams {
    fn from(value: &SamplingParams) -> Self {
        Self {
            temperature: value.temperature,
            max_tokens: value.max_tokens,
            ignore_eos: value.ignore_eos,
            return_completion_logprobs: value.return_completion_logprobs,
        }
    }
}

impl WireSamplingParams {
    pub(crate) fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams::new(
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
            self.return_completion_logprobs,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WireRequestIn {
    pub(crate) seq_id: u64,
    pub(crate) prompt_token_ids: Vec<i32>,
    pub(crate) sampling_params: WireSamplingParams,
    pub(crate) affinity_key: u64,
    pub(crate) vision_slots: Vec<WireVisionSlot>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WireVisionSlot {
    pub(crate) encoder_engine_id: String,
    pub(crate) slot_idx: i32,
    pub(crate) num_tokens: i32,
    pub(crate) hidden_size: i32,
    pub(crate) max_tokens_per_slot: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct WireRequestMigrate {
    pub(crate) seq_id: u64,
    pub(crate) status: i32,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) last_token: i32,
    pub(crate) num_tokens: i32,
    pub(crate) num_prompt_tokens: i32,
    pub(crate) num_checkpointed_tokens: i32,
    pub(crate) num_cached_tokens: i32,
    pub(crate) affinity_key: u64,
    pub(crate) sampling_params: WireSamplingParams,
    pub(crate) completion_logprobs: Vec<f32>,
    pub(crate) active_block_table: Vec<i32>,
    pub(crate) active_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) active_dispatched_tokens: Vec<i32>,
    pub(crate) active_dp_idx: i32,
    pub(crate) active_group_id: i32,
    pub(crate) active_state_slot: i32,
    pub(crate) active_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) active_hisparse_slot: i32,
    pub(crate) migrate_block_table: Vec<i32>,
    pub(crate) migrate_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) migrate_engine_id: String,
    pub(crate) migrate_num_kvcache_blocks: i32,
    pub(crate) migrate_group_size: i32,
    pub(crate) migrate_dp_idx: i32,
    pub(crate) migrate_group_id: i32,
    pub(crate) migrate_state_slot: i32,
    pub(crate) migrate_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(crate) migrate_hisparse_slot: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireMigrateSequence {
    pub(crate) seq_id: u64,
    pub(crate) migrate_engine_id: String,
    pub(crate) migrate_num_kvcache_blocks: i32,
    pub(crate) migrate_group_size: i32,
    pub(crate) migrate_dp_idx: i32,
    pub(crate) migrate_block_location: Vec<(i32, i32)>,
    pub(crate) migrate_state_slot: i32,
    pub(crate) migrate_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
    pub(crate) active_block_location: Vec<(i32, i32)>,
    pub(crate) active_state_slot: i32,
    pub(crate) active_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireRunResult {
    pub(crate) token_ids: Vec<Vec<i64>>,
    pub(super) logprobs: Option<Vec<Vec<f32>>>,
    pub(super) server_handler_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WirePacket {
    pub(crate) action: i32,
    pub(crate) payload: Vec<u8>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireStepOut {
    pub(crate) seq_id: u64,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) status: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireFreeSequences {
    pub(crate) seq_ids: Vec<u64>,
    pub(crate) source_engine_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireFreeVisionSlots {
    pub(crate) encoder_engine_id: String,
    pub(crate) slot_indices: Vec<i32>,
    pub(crate) source_engine_id: String,
}

impl WireBatch {
    pub(super) fn empty(is_prefill: bool) -> Self {
        Self {
            is_prefill,
            input_ids: Vec::new(),
            positions: Vec::new(),
            seq_lens: Vec::new(),
            block_tables: Vec::new(),
            temperatures: Vec::new(),
            state_slots: Vec::new(),
            compressed_block_tables: std::collections::HashMap::new(),
            hisparse_slots: Vec::new(),
            seq_ids: Vec::new(),
            sample_mask: Vec::new(),
            any_return_completion_logprobs: false,
            is_dummy: false,
        }
    }

    pub(super) fn dummy(is_prefill: bool) -> Self {
        Self {
            is_prefill,
            input_ids: vec![0],
            positions: vec![0],
            seq_lens: vec![1],
            block_tables: vec![Vec::new()],
            temperatures: vec![0.0],
            state_slots: vec![-1],
            compressed_block_tables: std::collections::HashMap::new(),
            hisparse_slots: vec![-1],
            seq_ids: vec![0],
            sample_mask: vec![true],
            any_return_completion_logprobs: false,
            is_dummy: true,
        }
    }

    pub(super) fn num_group_seqs(&self) -> usize {
        self.seq_lens.len()
    }

    /// Split a prefill batch into ordered, single-request token fragments.
    ///
    /// Keeping one request per microbatch makes independent requests pipeline
    /// naturally, while splitting a long request at `max_tokens` fills the
    /// same forward-only pipeline. Only the last fragment of an original row
    /// remains sampleable; earlier fragments update cache/state only.
    pub(super) fn prefill_microbatches(&self, max_tokens: usize) -> Vec<(Self, usize, bool)> {
        if !self.is_prefill || self.is_dummy || max_tokens == 0 || self.seq_lens.is_empty() {
            return vec![(self.clone(), 0, true)];
        }

        let mut result = Vec::new();
        let mut token_offset = 0usize;
        for (seq_idx, seq_len) in self.seq_lens.iter().copied().enumerate() {
            let seq_len = seq_len.max(0) as usize;
            if seq_len == 0 {
                token_offset += seq_len;
                continue;
            }
            let seq_end = token_offset.saturating_add(seq_len);
            let mut fragment_start = token_offset;
            while fragment_start < seq_end {
                let fragment_end = fragment_start.saturating_add(max_tokens).min(seq_end);
                let is_last_fragment = fragment_end == seq_end;
                let mut compressed_block_tables = HashMap::new();
                for (ratio, rows) in &self.compressed_block_tables {
                    compressed_block_tables
                        .insert(*ratio, vec![rows.get(seq_idx).cloned().unwrap_or_default()]);
                }
                result.push((
                    Self {
                        is_prefill: true,
                        input_ids: self.input_ids[fragment_start..fragment_end].to_vec(),
                        positions: self.positions[fragment_start..fragment_end].to_vec(),
                        seq_lens: vec![(fragment_end - fragment_start) as i32],
                        block_tables: vec![self
                            .block_tables
                            .get(seq_idx)
                            .cloned()
                            .unwrap_or_default()],
                        temperatures: vec![self.temperatures.get(seq_idx).copied().unwrap_or(0.0)],
                        state_slots: vec![self.state_slots.get(seq_idx).copied().unwrap_or(-1)],
                        compressed_block_tables,
                        hisparse_slots: vec![self
                            .hisparse_slots
                            .get(seq_idx)
                            .copied()
                            .unwrap_or(-1)],
                        seq_ids: vec![self.seq_ids.get(seq_idx).copied().unwrap_or(0)],
                        sample_mask: vec![
                            is_last_fragment
                                && self.sample_mask.get(seq_idx).copied().unwrap_or(true),
                        ],
                        any_return_completion_logprobs: self
                            .any_return_completion_logprobs,
                        is_dummy: false,
                    },
                    seq_idx,
                    is_last_fragment,
                ));
                fragment_start = fragment_end;
            }
            token_offset = seq_end;
        }
        if result.is_empty() {
            result.push((self.clone(), 0, true));
        }
        result
    }
}

pub(crate) fn encode_binary<T: Serialize>(value: &T, what: &str) -> PyResult<Vec<u8>> {
    bincode::serialize(value)
        .map_err(|e| PyValueError::new_err(format!("failed to encode {what}: {e}")))
}

pub(crate) fn decode_binary<T: DeserializeOwned>(data: &[u8], what: &str) -> PyResult<T> {
    bincode::deserialize(data)
        .map_err(|e| PyValueError::new_err(format!("failed to decode {what}: {e}")))
}

pub(crate) fn sequence_refs_runner_in_bytes(
    seqs: Vec<&Sequence>,
    is_prefill: bool,
) -> PyResult<Vec<u8>> {
    if seqs.is_empty() {
        return encode_binary(&WireBatch::empty(is_prefill), "run batch");
    }

    let mut input_ids = Vec::new();
    let mut positions = Vec::new();
    let mut seq_lens = Vec::new();
    let mut block_tables = Vec::new();
    let mut temperatures = Vec::new();
    let mut state_slots = Vec::new();
    let mut compressed_block_tables: HashMap<i32, Vec<Vec<i32>>> = HashMap::new();
    let mut hisparse_slots = Vec::new();
    let mut seq_ids = Vec::new();
    let mut sample_mask = Vec::new();
    let mut any_return_completion_logprobs = false;

    for item in seqs {
        let tokens: Vec<i64> = item.token_ids.iter().copied().map(i64::from).collect();
        let prompt_len = (item.num_prompt_tokens.max(0) as usize).min(tokens.len());
        let block_table = item.active_block_table.clone();
        let compressed_tables = item.active_compressed_block_tables.clone();
        let temperature = item.sampling_params.temperature as f32;
        any_return_completion_logprobs |=
            item.sampling_params.return_completion_logprobs;

        let should_sample;
        if is_prefill {
            let start = (item.prefill_start_offset.max(0) as usize).min(tokens.len());
            let chunk_end = item.num_tokens.max(0) as usize;
            let end = chunk_end.min(prompt_len).min(tokens.len()).max(start);
            should_sample = end >= prompt_len;
            let slice = &tokens[start..end];
            seq_lens.push(slice.len() as i32);
            for (offset, token) in slice.iter().enumerate() {
                input_ids.push(*token);
                positions.push((start + offset) as i64);
            }
        } else {
            should_sample = true;
            let token = i64::from(item.last_token);
            input_ids.push(token);
            positions.push(tokens.len().saturating_sub(1) as i64);
            seq_lens.push(1);
        }
        block_tables.push(block_table);
        temperatures.push(temperature);
        state_slots.push(i64::from(item.active_state_slot));
        hisparse_slots.push(i64::from(item.active_hisparse_slot));
        seq_ids.push(item.seq_id);
        sample_mask.push(should_sample);
        for (ratio, blocks) in compressed_tables {
            let rows = compressed_block_tables.entry(ratio).or_default();
            while rows.len() + 1 < seq_lens.len() {
                rows.push(Vec::new());
            }
            rows.push(blocks);
        }
        for rows in compressed_block_tables.values_mut() {
            while rows.len() < seq_lens.len() {
                rows.push(Vec::new());
            }
        }
    }

    encode_binary(
        &WireBatch {
            is_prefill,
            input_ids,
            positions,
            seq_lens,
            block_tables,
            temperatures,
            state_slots,
            compressed_block_tables,
            hisparse_slots,
            seq_ids,
            sample_mask,
            any_return_completion_logprobs,
            is_dummy: false,
        },
        "run batch",
    )
}

pub(crate) fn sequence_refs_migrate_batch_bytes(seqs: Vec<&Sequence>) -> PyResult<Vec<u8>> {
    let mut wire = Vec::with_capacity(seqs.len());
    for seq in seqs {
        wire.push(sequence_to_migrate_wire_ref(seq));
    }
    encode_binary(&wire, "migrate batch")
}

pub(super) fn decode_wire(data: &[u8]) -> PyResult<WireBatch> {
    decode_binary(data, "run batch")
}

pub(crate) fn bytes_arg(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    data.extract::<Vec<u8>>()
}

pub(crate) fn request_to_sequence(request: WireRequestIn) -> Sequence {
    let mut seq = Sequence::new(
        request.prompt_token_ids,
        Some(request.sampling_params.to_sampling_params()),
    );
    seq.seq_id = request.seq_id;
    seq.affinity_key = request.affinity_key;
    seq.vision_slots = request
        .vision_slots
        .into_iter()
        .map(|slot| VisionSlot {
            encoder_engine_id: slot.encoder_engine_id,
            slot_idx: slot.slot_idx,
            num_tokens: slot.num_tokens,
            hidden_size: slot.hidden_size,
            max_tokens_per_slot: slot.max_tokens_per_slot,
        })
        .collect();
    seq
}

pub(crate) fn sequence_to_migration_request_ref(seq: &Sequence) -> WireRequestMigrate {
    WireRequestMigrate {
        seq_id: seq.seq_id,
        status: seq.status,
        token_ids: seq.token_ids.clone(),
        last_token: seq.last_token,
        num_tokens: seq.num_tokens,
        num_prompt_tokens: seq.num_prompt_tokens,
        num_checkpointed_tokens: seq.num_checkpointed_tokens,
        num_cached_tokens: seq.num_cached_tokens,
        affinity_key: seq.affinity_key,
        sampling_params: WireSamplingParams::from(&seq.sampling_params),
        completion_logprobs: seq.completion_logprobs.clone(),
        active_block_table: seq.active_block_table.clone(),
        active_block_tables: seq.active_block_tables.clone(),
        active_dispatched_tokens: seq.active_dispatched_tokens.clone(),
        active_dp_idx: seq.active_dp_idx,
        active_group_id: seq.active_group_id,
        active_state_slot: seq.active_state_slot,
        active_compressed_block_tables: seq.active_compressed_block_tables.clone(),
        active_hisparse_slot: seq.active_hisparse_slot,
        migrate_block_table: seq.migrate_block_table.clone(),
        migrate_block_tables: seq.migrate_block_tables.clone(),
        migrate_engine_id: seq.migrate_engine_id.clone(),
        migrate_num_kvcache_blocks: seq.migrate_num_kvcache_blocks,
        migrate_group_size: seq.migrate_group_size,
        migrate_dp_idx: seq.migrate_dp_idx,
        migrate_group_id: seq.migrate_group_id,
        migrate_state_slot: seq.migrate_state_slot,
        migrate_compressed_block_tables: seq.migrate_compressed_block_tables.clone(),
        migrate_hisparse_slot: seq.migrate_hisparse_slot,
    }
}

pub(crate) fn migrate_request_to_sequence(request: WireRequestMigrate) -> Sequence {
    let mut seq = Sequence::new(
        request.token_ids,
        Some(request.sampling_params.to_sampling_params()),
    );
    seq.seq_id = request.seq_id;
    seq.status = request.status;
    seq.last_token = request.last_token;
    seq.num_tokens = request.num_tokens;
    seq.num_prompt_tokens = request.num_prompt_tokens;
    seq.num_checkpointed_tokens = request.num_checkpointed_tokens;
    seq.num_cached_tokens = request.num_cached_tokens;
    seq.affinity_key = request.affinity_key;
    seq.completion_logprobs = request.completion_logprobs;
    seq.active_block_table = request.active_block_table;
    seq.active_block_tables = request.active_block_tables;
    seq.active_dispatched_tokens = request.active_dispatched_tokens;
    seq.active_dp_idx = request.active_dp_idx;
    seq.active_group_id = request.active_group_id;
    seq.active_state_slot = request.active_state_slot;
    seq.active_compressed_block_tables = request.active_compressed_block_tables;
    seq.active_hisparse_slot = request.active_hisparse_slot;
    seq.migrate_block_table = request.migrate_block_table;
    seq.migrate_block_tables = request.migrate_block_tables;
    seq.migrate_engine_id = request.migrate_engine_id;
    seq.migrate_num_kvcache_blocks = request.migrate_num_kvcache_blocks;
    seq.migrate_group_size = request.migrate_group_size.max(1);
    seq.migrate_dp_idx = request.migrate_dp_idx;
    seq.migrate_group_id = request.migrate_group_id;
    seq.migrate_state_slot = request.migrate_state_slot;
    seq.migrate_compressed_block_tables = request.migrate_compressed_block_tables;
    seq.migrate_hisparse_slot = request.migrate_hisparse_slot;
    seq
}

fn block_locations(group_id: i32, blocks: &[i32]) -> Vec<(i32, i32)> {
    blocks
        .iter()
        .copied()
        .map(|block| (group_id, block))
        .collect()
}

pub(super) fn sequence_to_migrate_wire_ref(seq: &Sequence) -> WireMigrateSequence {
    let migrate_blocks = seq
        .migrate_block_tables
        .get(&seq.migrate_group_id)
        .unwrap_or(&seq.migrate_block_table);
    let active_blocks = seq
        .active_block_tables
        .get(&seq.active_group_id)
        .unwrap_or(&seq.active_block_table);
    WireMigrateSequence {
        seq_id: seq.seq_id,
        migrate_engine_id: seq.migrate_engine_id.clone(),
        migrate_num_kvcache_blocks: seq.migrate_num_kvcache_blocks,
        migrate_group_size: seq.migrate_group_size,
        migrate_dp_idx: seq.migrate_dp_idx,
        migrate_block_location: block_locations(seq.migrate_group_id, migrate_blocks),
        migrate_state_slot: seq.migrate_state_slot,
        migrate_compressed_block_tables: seq.migrate_compressed_block_tables.clone(),
        active_block_location: block_locations(seq.active_group_id, active_blocks),
        active_state_slot: seq.active_state_slot,
        active_compressed_block_tables: seq.active_compressed_block_tables.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_microbatches_split_long_and_independent_requests_in_order() {
        let mut compressed = HashMap::new();
        compressed.insert(4, vec![vec![10, 11], vec![20]]);
        let batch = WireBatch {
            is_prefill: true,
            input_ids: (0..8).collect(),
            positions: vec![0, 1, 2, 3, 4, 10, 11, 12],
            seq_lens: vec![5, 3],
            block_tables: vec![vec![1, 2], vec![3]],
            temperatures: vec![0.1, 0.2],
            state_slots: vec![7, 8],
            compressed_block_tables: compressed,
            hisparse_slots: vec![9, 10],
            seq_ids: vec![100, 200],
            sample_mask: vec![true, true],
            any_return_completion_logprobs: true,
            is_dummy: false,
        };

        let fragments = batch.prefill_microbatches(2);
        assert_eq!(fragments.len(), 5);
        assert_eq!(
            fragments
                .iter()
                .map(|(_, seq_idx, is_last)| (*seq_idx, *is_last))
                .collect::<Vec<_>>(),
            vec![(0, false), (0, false), (0, true), (1, false), (1, true)]
        );
        assert_eq!(fragments[0].0.input_ids, vec![0, 1]);
        assert_eq!(fragments[2].0.input_ids, vec![4]);
        assert_eq!(fragments[3].0.input_ids, vec![5, 6]);
        assert_eq!(fragments[4].0.positions, vec![12]);
        assert_eq!(fragments[0].0.sample_mask, vec![false]);
        assert_eq!(fragments[2].0.sample_mask, vec![true]);
        assert_eq!(fragments[3].0.seq_ids, vec![200]);
        assert_eq!(fragments[3].0.block_tables, vec![vec![3]]);
        assert_eq!(
            fragments[3].0.compressed_block_tables.get(&4),
            Some(&vec![vec![20]])
        );

        let intermediate_payload =
            encode_binary(&fragments[0].0, "intermediate microbatch").unwrap();
        let intermediate_meta =
            crate::proto::prepare::runner_in_prefill(&intermediate_payload, 0, 1, 64, 8, 16)
                .unwrap();
        assert!(intermediate_meta.sampling_token_indices.is_empty());

        let final_payload = encode_binary(&fragments[2].0, "final microbatch").unwrap();
        let final_meta =
            crate::proto::prepare::runner_in_prefill(&final_payload, 0, 1, 64, 8, 16).unwrap();
        assert_eq!(final_meta.sampling_token_indices, vec![0]);
    }
}
