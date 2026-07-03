use crate::sequence::{SamplingParams, Sequence};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
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
    pub(super) is_dummy: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireSamplingParams {
    pub(super) temperature: f64,
    pub(super) max_tokens: i32,
    pub(super) ignore_eos: bool,
    pub(super) return_completion_logprobs: bool,
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
    fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams::new(
            self.temperature,
            self.max_tokens,
            self.ignore_eos,
            self.return_completion_logprobs,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireSequence {
    seq_id: u64,
    status: i32,
    token_ids: Vec<i32>,
    last_token: i32,
    num_prompt_tokens: i32,
    num_cached_tokens: i32,
    affinity_key: u64,
    sampling_params: WireSamplingParams,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireAddRequest {
    pub(super) seq_id: u64,
    pub(super) prompt_token_ids: Vec<i32>,
    pub(super) sampling_params: WireSamplingParams,
    pub(super) affinity_key: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireMigrationRequest {
    pub(super) seq_id: u64,
    pub(super) status: i32,
    pub(super) token_ids: Vec<i32>,
    pub(super) last_token: i32,
    pub(super) num_tokens: i32,
    pub(super) num_prompt_tokens: i32,
    pub(super) num_checkpointed_tokens: i32,
    pub(super) num_cached_tokens: i32,
    pub(super) affinity_key: u64,
    pub(super) sampling_params: WireSamplingParams,
    pub(super) completion_logprobs: Vec<f32>,
    pub(super) active_block_table: Vec<i32>,
    pub(super) active_block_tables: HashMap<i32, Vec<i32>>,
    pub(super) active_dispatched_tokens: Vec<i32>,
    pub(super) active_dp_idx: i32,
    pub(super) active_group_id: i32,
    pub(super) active_state_slot: i32,
    pub(super) active_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(super) active_hisparse_slot: i32,
    pub(super) migrate_block_table: Vec<i32>,
    pub(super) migrate_block_tables: HashMap<i32, Vec<i32>>,
    pub(super) migrate_engine_id: String,
    pub(super) migrate_num_kvcache_blocks: i32,
    pub(super) migrate_group_size: i32,
    pub(super) migrate_dp_idx: i32,
    pub(super) migrate_group_id: i32,
    pub(super) migrate_state_slot: i32,
    pub(super) migrate_compressed_block_tables: HashMap<i32, Vec<i32>>,
    pub(super) migrate_hisparse_slot: i32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireMigrateSequence {
    pub(super) seq_id: u64,
    pub(super) migrate_engine_id: String,
    pub(super) migrate_num_kvcache_blocks: i32,
    pub(super) migrate_group_size: i32,
    pub(super) migrate_dp_idx: i32,
    pub(super) migrate_block_location: Vec<(i32, i32)>,
    pub(super) migrate_state_slot: i32,
    pub(super) migrate_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
    pub(super) active_block_location: Vec<(i32, i32)>,
    pub(super) active_state_slot: i32,
    pub(super) active_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct WireRunResult {
    pub(super) token_ids: Vec<Vec<i64>>,
    pub(super) logprobs: Option<Vec<Vec<f32>>>,
    pub(super) server_handler_ns: u64,
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
            is_dummy: true,
        }
    }

    pub(super) fn num_group_seqs(&self) -> usize {
        self.seq_lens.len()
    }
}

pub(super) fn encode_binary<T: Serialize>(value: &T, what: &str) -> PyResult<Vec<u8>> {
    bincode::serialize(value)
        .map_err(|e| PyValueError::new_err(format!("failed to encode {what}: {e}")))
}

pub(super) fn decode_binary<T: DeserializeOwned>(data: &[u8], what: &str) -> PyResult<T> {
    bincode::deserialize(data)
        .map_err(|e| PyValueError::new_err(format!("failed to decode {what}: {e}")))
}

pub(super) fn encode_wire(py: Python<'_>, batch: &WireBatch) -> PyResult<PyObject> {
    let bytes = encode_binary(batch, "run batch")?;
    Ok(PyBytes::new(py, &bytes).into())
}

pub(super) fn decode_wire(data: &[u8]) -> PyResult<WireBatch> {
    decode_binary(data, "run batch")
}

pub(super) fn bytes_arg(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    data.extract::<Vec<u8>>()
}

pub(super) fn sequence_to_wire(py: Python<'_>, seq: &Py<Sequence>) -> WireSequence {
    let seq = seq.borrow(py);
    WireSequence {
        seq_id: seq.seq_id,
        status: seq.status,
        token_ids: seq.token_ids.clone(),
        last_token: seq.last_token,
        num_prompt_tokens: seq.num_prompt_tokens,
        num_cached_tokens: seq.num_cached_tokens,
        affinity_key: seq.affinity_key,
        sampling_params: WireSamplingParams::from(&seq.sampling_params),
    }
}

pub(super) fn wire_to_sequence(py: Python<'_>, wire: WireSequence) -> PyResult<Py<Sequence>> {
    let sampling_params = wire.sampling_params.to_sampling_params();
    let mut seq = Sequence::new(wire.token_ids, Some(sampling_params));
    seq.seq_id = wire.seq_id;
    seq.status = wire.status;
    seq.last_token = wire.last_token;
    seq.num_prompt_tokens = wire.num_prompt_tokens;
    seq.num_cached_tokens = wire.num_cached_tokens;
    seq.affinity_key = wire.affinity_key;
    Py::new(py, seq)
}

pub(super) fn add_request_to_sequence(
    py: Python<'_>,
    request: WireAddRequest,
) -> PyResult<Py<Sequence>> {
    let mut seq = Sequence::new(
        request.prompt_token_ids,
        Some(request.sampling_params.to_sampling_params()),
    );
    seq.seq_id = request.seq_id;
    seq.affinity_key = request.affinity_key;
    Py::new(py, seq)
}

pub(super) fn sequence_to_migration_request(
    py: Python<'_>,
    seq: &Py<Sequence>,
) -> WireMigrationRequest {
    let seq = seq.borrow(py);
    WireMigrationRequest {
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

pub(super) fn migration_request_to_sequence(
    py: Python<'_>,
    request: WireMigrationRequest,
) -> PyResult<Py<Sequence>> {
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
    Py::new(py, seq)
}

fn block_locations(group_id: i32, blocks: &[i32]) -> Vec<(i32, i32)> {
    blocks
        .iter()
        .copied()
        .map(|block| (group_id, block))
        .collect()
}

pub(super) fn sequence_to_migrate_wire(py: Python<'_>, seq: &Py<Sequence>) -> WireMigrateSequence {
    let seq = seq.borrow(py);
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

pub(super) fn run_result_tuple(py: Python<'_>, result: WireRunResult) -> PyResult<PyObject> {
    let token_ids = result.token_ids.into_pyobject(py)?.unbind().into_any();
    let logprobs = if let Some(logprobs) = result.logprobs {
        logprobs.into_pyobject(py)?.unbind().into_any()
    } else {
        py.None()
    };
    Ok(PyTuple::new(py, [token_ids, logprobs])?.into())
}
