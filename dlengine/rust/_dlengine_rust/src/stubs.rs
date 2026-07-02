use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireBatch {
    is_prefill: bool,
    input_ids: Vec<i64>,
    positions: Vec<i64>,
    seq_lens: Vec<i32>,
    block_tables: Vec<Vec<i32>>,
    temperatures: Vec<f32>,
    state_slots: Vec<i64>,
    compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>>,
    hisparse_slots: Vec<i64>,
    is_dummy: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireSamplingParams {
    temperature: f64,
    max_tokens: i32,
    ignore_eos: bool,
    return_completion_logprobs: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireSequence {
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
struct WireMigrateSequence {
    seq_id: u64,
    migrate_engine_id: String,
    migrate_num_kvcache_blocks: i32,
    migrate_group_size: i32,
    migrate_dp_idx: i32,
    migrate_block_location: Vec<(i32, i32)>,
    migrate_state_slot: i32,
    migrate_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
    active_block_location: Vec<(i32, i32)>,
    active_state_slot: i32,
    active_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct WireRunResult {
    token_ids: Vec<Vec<i64>>,
    logprobs: Option<Vec<Vec<f32>>>,
    server_handler_ns: u64,
}

impl WireBatch {
    fn empty(is_prefill: bool) -> Self {
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

    fn dummy(is_prefill: bool) -> Self {
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

    fn num_group_seqs(&self) -> usize {
        self.seq_lens.len()
    }
}

fn encode_binary<T: Serialize>(value: &T, what: &str) -> PyResult<Vec<u8>> {
    bincode::serialize(value)
        .map_err(|e| PyValueError::new_err(format!("failed to encode {what}: {e}")))
}

fn decode_binary<T: DeserializeOwned>(data: &[u8], what: &str) -> PyResult<T> {
    bincode::deserialize(data)
        .map_err(|e| PyValueError::new_err(format!("failed to decode {what}: {e}")))
}

fn encode_wire(py: Python<'_>, batch: &WireBatch) -> PyResult<PyObject> {
    let bytes = encode_binary(batch, "run batch")?;
    Ok(PyBytes::new(py, &bytes).into())
}

fn decode_wire(data: &[u8]) -> PyResult<WireBatch> {
    decode_binary(data, "run batch")
}

fn bytes_arg(data: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = data.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    data.extract::<Vec<u8>>()
}

fn sequence_to_wire(seq: &Bound<'_, PyAny>) -> PyResult<WireSequence> {
    let sampling_params = seq.getattr("sampling_params")?;
    Ok(WireSequence {
        seq_id: seq.getattr("seq_id")?.extract()?,
        status: seq.getattr("status")?.extract().unwrap_or(0),
        token_ids: seq.getattr("token_ids")?.extract()?,
        last_token: seq.getattr("last_token")?.extract().unwrap_or(-1),
        num_prompt_tokens: seq
            .getattr("num_prompt_tokens")
            .and_then(|v| v.extract::<i32>())
            .unwrap_or(0),
        num_cached_tokens: seq
            .getattr("num_cached_tokens")
            .and_then(|v| v.extract::<i32>())
            .unwrap_or(0),
        affinity_key: seq
            .getattr("affinity_key")
            .and_then(|v| v.extract::<u64>())
            .unwrap_or(0),
        sampling_params: WireSamplingParams {
            temperature: sampling_params
                .getattr("temperature")
                .and_then(|v| v.extract::<f64>())
                .unwrap_or(1.0),
            max_tokens: sampling_params
                .getattr("max_tokens")
                .and_then(|v| v.extract::<i32>())
                .unwrap_or(256),
            ignore_eos: sampling_params
                .getattr("ignore_eos")
                .and_then(|v| v.extract::<bool>())
                .unwrap_or(false),
            return_completion_logprobs: sampling_params
                .getattr("return_completion_logprobs")
                .and_then(|v| v.extract::<bool>())
                .unwrap_or(false),
        },
    })
}

fn wire_to_sequence(py: Python<'_>, wire: WireSequence) -> PyResult<PyObject> {
    let module = py.import("dlengine._dlengine_rust")?;
    let sampling_params_cls = module.getattr("SamplingParams")?;
    let sequence_cls = module.getattr("Sequence")?;
    let sampling_params = sampling_params_cls.call1((
        wire.sampling_params.temperature,
        wire.sampling_params.max_tokens,
        wire.sampling_params.ignore_eos,
        wire.sampling_params.return_completion_logprobs,
    ))?;
    let seq = sequence_cls.call1((wire.token_ids, sampling_params))?;
    seq.setattr("seq_id", wire.seq_id)?;
    seq.setattr("status", wire.status)?;
    seq.setattr("last_token", wire.last_token)?;
    seq.setattr("num_prompt_tokens", wire.num_prompt_tokens)?;
    seq.setattr("num_cached_tokens", wire.num_cached_tokens)?;
    seq.setattr("affinity_key", wire.affinity_key)?;
    Ok(seq.into())
}

fn extract_i32_pair_list(obj: &Bound<'_, PyAny>, name: &str) -> Vec<(i32, i32)> {
    obj.getattr(name)
        .ok()
        .and_then(|value| value.extract::<Vec<(i32, i32)>>().ok())
        .unwrap_or_default()
}

fn extract_compressed_tables(
    obj: &Bound<'_, PyAny>,
    name: &str,
) -> std::collections::HashMap<i32, Vec<i32>> {
    obj.getattr(name)
        .ok()
        .and_then(|value| {
            value
                .extract::<std::collections::HashMap<i32, Vec<i32>>>()
                .ok()
        })
        .unwrap_or_default()
}

fn sequence_to_migrate_wire(seq: &Bound<'_, PyAny>) -> WireMigrateSequence {
    let migrate_engine_id = seq
        .call_method0("migrate_engine_id")
        .ok()
        .and_then(|value| value.extract::<String>().ok())
        .or_else(|| {
            seq.getattr("migrate_engine_id")
                .ok()
                .and_then(|value| value.extract::<String>().ok())
        })
        .unwrap_or_default();

    WireMigrateSequence {
        seq_id: seq
            .getattr("seq_id")
            .and_then(|value| value.extract::<u64>())
            .unwrap_or(0),
        migrate_engine_id,
        migrate_num_kvcache_blocks: seq
            .getattr("migrate_num_kvcache_blocks")
            .and_then(|value| value.extract::<i32>())
            .unwrap_or(0),
        migrate_group_size: seq
            .getattr("migrate_group_size")
            .and_then(|value| value.extract::<i32>())
            .unwrap_or(1),
        migrate_dp_idx: seq
            .getattr("migrate_dp_idx")
            .and_then(|value| value.extract::<i32>())
            .unwrap_or(0),
        migrate_block_location: extract_i32_pair_list(seq, "migrate_block_location"),
        migrate_state_slot: seq
            .getattr("migrate_state_slot")
            .and_then(|value| value.extract::<i32>())
            .unwrap_or(-1),
        migrate_compressed_block_tables: extract_compressed_tables(
            seq,
            "migrate_compressed_block_tables",
        ),
        active_block_location: extract_i32_pair_list(seq, "active_block_location"),
        active_state_slot: seq
            .getattr("active_state_slot")
            .and_then(|value| value.extract::<i32>())
            .unwrap_or(-1),
        active_compressed_block_tables: extract_compressed_tables(
            seq,
            "active_compressed_block_tables",
        ),
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone, Debug)]
pub struct BatchAuxData {
    #[pyo3(get, set)]
    pub num_group_seqs: usize,
    #[pyo3(get, set)]
    pub temperatures: Vec<f32>,
    #[pyo3(get, set)]
    pub state_slots: Vec<i64>,
    #[pyo3(get, set)]
    pub compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>>,
    #[pyo3(get, set)]
    pub hisparse_slots: Vec<i64>,
    #[pyo3(get, set)]
    pub any_return_completion_logprobs: bool,
}

#[pymethods]
impl BatchAuxData {
    #[new]
    #[pyo3(signature = (
        num_group_seqs = 0,
        temperatures = Vec::new(),
        state_slots = Vec::new(),
        compressed_block_tables = std::collections::HashMap::new(),
        hisparse_slots = Vec::new(),
        any_return_completion_logprobs = false
    ))]
    fn new(
        num_group_seqs: usize,
        temperatures: Vec<f32>,
        state_slots: Vec<i64>,
        compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>>,
        hisparse_slots: Vec<i64>,
        any_return_completion_logprobs: bool,
    ) -> Self {
        Self {
            num_group_seqs,
            temperatures,
            state_slots,
            compressed_block_tables,
            hisparse_slots,
            any_return_completion_logprobs,
        }
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone, Debug)]
pub struct PrefillMeta {
    #[pyo3(get, set)]
    pub input_ids: Vec<i64>,
    #[pyo3(get, set)]
    pub positions: Vec<i64>,
    #[pyo3(get, set)]
    pub cu_seqlens_q: Vec<i32>,
    #[pyo3(get, set)]
    pub cu_seqlens_k: Vec<i32>,
    #[pyo3(get, set)]
    pub slot_mapping: Vec<i32>,
    #[pyo3(get, set)]
    pub use_block_tables: bool,
    #[pyo3(get, set)]
    pub block_tables_flat: Vec<i32>,
    #[pyo3(get, set)]
    pub max_num_blocks: usize,
    #[pyo3(get, set)]
    pub max_seqlen_q: usize,
    #[pyo3(get, set)]
    pub max_seqlen_k: usize,
    #[pyo3(get, set)]
    pub sampling_token_indices: Vec<i64>,
    #[pyo3(get, set)]
    pub sampling_seq_indices: Vec<i64>,
}

#[pymethods]
impl PrefillMeta {
    #[new]
    fn new() -> Self {
        Self {
            input_ids: Vec::new(),
            positions: Vec::new(),
            cu_seqlens_q: vec![0],
            cu_seqlens_k: vec![0],
            slot_mapping: Vec::new(),
            use_block_tables: false,
            block_tables_flat: Vec::new(),
            max_num_blocks: 0,
            max_seqlen_q: 0,
            max_seqlen_k: 0,
            sampling_token_indices: Vec::new(),
            sampling_seq_indices: Vec::new(),
        }
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone, Debug)]
pub struct DecodeMeta {
    #[pyo3(get, set)]
    pub input_ids: Vec<i64>,
    #[pyo3(get, set)]
    pub positions: Vec<i64>,
    #[pyo3(get, set)]
    pub slot_mapping: Vec<i32>,
    #[pyo3(get, set)]
    pub context_lens_flat: Vec<i32>,
    #[pyo3(get, set)]
    pub block_tables_flat: Vec<i32>,
    #[pyo3(get, set)]
    pub max_num_blocks: usize,
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone, Debug)]
pub struct MigrateSequenceView {
    #[pyo3(get, set)]
    pub seq_id: u64,
    #[pyo3(get, set)]
    pub migrate_engine_id: String,
    #[pyo3(get, set)]
    pub migrate_num_kvcache_blocks: i32,
    #[pyo3(get, set)]
    pub migrate_group_size: i32,
    #[pyo3(get, set)]
    pub migrate_dp_idx: i32,
    #[pyo3(get, set)]
    pub migrate_block_location: Vec<(i32, i32)>,
    #[pyo3(get, set)]
    pub migrate_state_slot: i32,
    #[pyo3(get, set)]
    pub migrate_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
    #[pyo3(get, set)]
    pub active_block_location: Vec<(i32, i32)>,
    #[pyo3(get, set)]
    pub active_state_slot: i32,
    #[pyo3(get, set)]
    pub active_compressed_block_tables: std::collections::HashMap<i32, Vec<i32>>,
}

#[pymethods]
impl MigrateSequenceView {
    #[new]
    fn new() -> Self {
        Self {
            seq_id: 0,
            migrate_engine_id: String::new(),
            migrate_num_kvcache_blocks: 0,
            migrate_group_size: 1,
            migrate_dp_idx: 0,
            migrate_block_location: Vec::new(),
            migrate_state_slot: -1,
            migrate_compressed_block_tables: std::collections::HashMap::new(),
            active_block_location: Vec::new(),
            active_state_slot: -1,
            active_compressed_block_tables: std::collections::HashMap::new(),
        }
    }
}

impl From<WireMigrateSequence> for MigrateSequenceView {
    fn from(wire: WireMigrateSequence) -> Self {
        Self {
            seq_id: wire.seq_id,
            migrate_engine_id: wire.migrate_engine_id,
            migrate_num_kvcache_blocks: wire.migrate_num_kvcache_blocks,
            migrate_group_size: wire.migrate_group_size,
            migrate_dp_idx: wire.migrate_dp_idx,
            migrate_block_location: wire.migrate_block_location,
            migrate_state_slot: wire.migrate_state_slot,
            migrate_compressed_block_tables: wire.migrate_compressed_block_tables,
            active_block_location: wire.active_block_location,
            active_state_slot: wire.active_state_slot,
            active_compressed_block_tables: wire.active_compressed_block_tables,
        }
    }
}

#[pymethods]
impl DecodeMeta {
    #[new]
    fn new() -> Self {
        Self {
            input_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            context_lens_flat: Vec::new(),
            block_tables_flat: Vec::new(),
            max_num_blocks: 0,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (seqs, is_prefill))]
fn serialize_run_batch(
    py: Python<'_>,
    seqs: &Bound<'_, PyAny>,
    is_prefill: bool,
) -> PyResult<PyObject> {
    let seq: Vec<PyObject> = seqs.extract()?;
    if seq.is_empty() {
        return encode_wire(py, &WireBatch::empty(is_prefill));
    }

    let mut input_ids = Vec::new();
    let mut positions = Vec::new();
    let mut seq_lens = Vec::new();
    let mut block_tables = Vec::new();
    let mut temperatures = Vec::new();
    let mut state_slots = Vec::new();
    let mut compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>> =
        std::collections::HashMap::new();
    let mut hisparse_slots = Vec::new();

    for item in seq {
        let item = item.bind(py);
        let tokens: Vec<i64> = item.getattr("token_ids")?.extract()?;
        let prompt_len = item
            .getattr("num_prompt_tokens")
            .and_then(|v| v.extract::<usize>())
            .unwrap_or(tokens.len());
        let sampling_params = item.getattr("sampling_params").ok();
        let block_table = item
            .getattr("active_block_table")
            .and_then(|value| value.extract::<Vec<i32>>())
            .unwrap_or_default();
        let compressed_tables = item
            .getattr("active_compressed_block_tables")
            .and_then(|value| value.extract::<std::collections::HashMap<i32, Vec<i32>>>())
            .unwrap_or_default();
        let temperature = sampling_params
            .as_ref()
            .and_then(|sp| sp.getattr("temperature").ok())
            .and_then(|v| v.extract::<f32>().ok())
            .unwrap_or(0.0);

        if is_prefill {
            let start = item
                .getattr("num_cached_tokens")
                .and_then(|v| v.extract::<usize>())
                .unwrap_or(0)
                .min(tokens.len());
            let chunk_end = item
                .getattr("num_tokens")
                .and_then(|v| v.extract::<usize>())
                .unwrap_or(prompt_len);
            let end = chunk_end.min(prompt_len).min(tokens.len()).max(start);
            let slice = &tokens[start..end];
            seq_lens.push(slice.len() as i32);
            for (offset, token) in slice.iter().enumerate() {
                input_ids.push(*token);
                positions.push((start + offset) as i64);
            }
        } else {
            let token = item.getattr("last_token")?.extract::<i64>()?;
            input_ids.push(token);
            positions.push(tokens.len().saturating_sub(1) as i64);
            seq_lens.push(1);
        }
        block_tables.push(block_table);
        temperatures.push(temperature);
        state_slots.push(
            item.call_method1("state_slot", (0,))
                .and_then(|v| v.extract::<i64>())
                .unwrap_or(-1),
        );
        hisparse_slots.push(
            item.getattr("active_hisparse_slot")
                .and_then(|v| v.extract::<i64>())
                .unwrap_or(-1),
        );
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

    encode_wire(
        py,
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
            is_dummy: false,
        },
    )
}

#[pyfunction]
#[pyo3(signature = (_engine_id, _num_kvcache_blocks, is_prefill))]
fn serialize_dummy_run_batch(
    py: Python<'_>,
    _engine_id: &str,
    _num_kvcache_blocks: i64,
    is_prefill: bool,
) -> PyResult<PyObject> {
    encode_wire(py, &WireBatch::dummy(is_prefill))
}

#[pyfunction]
#[pyo3(signature = (data, _sp_rank = 0))]
fn extract_aux_from_bytes(data: &Bound<'_, PyAny>, _sp_rank: usize) -> PyResult<BatchAuxData> {
    let data = bytes_arg(data)?;
    let batch = decode_wire(&data)?;
    Ok(BatchAuxData {
        num_group_seqs: batch.num_group_seqs(),
        temperatures: batch.temperatures,
        state_slots: batch.state_slots,
        compressed_block_tables: batch.compressed_block_tables,
        hisparse_slots: batch.hisparse_slots,
        any_return_completion_logprobs: false,
    })
}

#[pyfunction]
#[pyo3(signature = (data, _sp_rank, _sp_size, _block_size, _max_num_seqs, _num_kvcache_blocks))]
fn prepare_prefill_from_bytes(
    data: &Bound<'_, PyAny>,
    _sp_rank: usize,
    _sp_size: usize,
    _block_size: usize,
    _max_num_seqs: usize,
    _num_kvcache_blocks: usize,
) -> PyResult<PrefillMeta> {
    let data = bytes_arg(data)?;
    let batch = decode_wire(&data)?;
    let mut cu = Vec::with_capacity(batch.seq_lens.len() + 1);
    cu.push(0);
    let mut total = 0i32;
    let mut max_len = 0usize;
    let max_blocks = batch
        .block_tables
        .iter()
        .map(|blocks| blocks.len())
        .max()
        .unwrap_or(0);
    let mut block_tables_flat = Vec::new();
    if !batch.is_dummy && max_blocks > 0 {
        block_tables_flat.resize(_sp_size * _max_num_seqs.max(1) * max_blocks, 0);
        for sp in 0.._sp_size {
            for (seq_idx, blocks) in batch.block_tables.iter().enumerate() {
                if seq_idx >= _max_num_seqs {
                    break;
                }
                let base = (sp * _max_num_seqs.max(1) + seq_idx) * max_blocks;
                for (block_idx, block) in blocks.iter().copied().enumerate() {
                    block_tables_flat[base + block_idx] = block;
                }
            }
        }
    }
    let mut sampling_token_indices = Vec::new();
    let mut sampling_seq_indices = Vec::new();
    for (idx, len) in batch.seq_lens.iter().copied().enumerate() {
        let len_usize = len.max(0) as usize;
        total += len.max(0);
        cu.push(total);
        max_len = max_len.max(len_usize);
        if len > 0 {
            sampling_token_indices.push((total - 1) as i64);
            sampling_seq_indices.push(idx as i64);
        }
    }
    let mut token_seq_indices = Vec::with_capacity(total.max(0) as usize);
    for (seq_idx, len) in batch.seq_lens.iter().copied().enumerate() {
        for _ in 0..len.max(0) {
            token_seq_indices.push(seq_idx);
        }
    }
    Ok(PrefillMeta {
        input_ids: batch.input_ids,
        positions: batch.positions.clone(),
        cu_seqlens_q: cu.clone(),
        cu_seqlens_k: cu,
        slot_mapping: if batch.is_dummy {
            vec![-1; total.max(0) as usize]
        } else {
            batch
                .positions
                .iter()
                .enumerate()
                .map(|(idx, pos)| {
                    let seq_idx = token_seq_indices.get(idx).copied().unwrap_or(0);
                    let block_table = batch.block_tables.get(seq_idx).cloned().unwrap_or_default();
                    let block_size = _block_size.max(1) as i64;
                    let logical_block = (*pos / block_size).max(0) as usize;
                    let offset = (*pos % block_size) as i32;
                    block_table
                        .get(logical_block)
                        .copied()
                        .map(|block| block * _block_size.max(1) as i32 + offset)
                        .unwrap_or(idx as i32)
                })
                .collect()
        },
        use_block_tables: !block_tables_flat.is_empty(),
        block_tables_flat,
        max_num_blocks: max_blocks,
        max_seqlen_q: max_len,
        max_seqlen_k: max_len,
        sampling_token_indices,
        sampling_seq_indices,
    })
}

#[pyfunction]
#[pyo3(signature = (data, _sp_rank, sp_size, _block_size, max_num_seqs, _num_kvcache_blocks))]
fn prepare_decode_from_bytes(
    data: &Bound<'_, PyAny>,
    _sp_rank: usize,
    sp_size: usize,
    _block_size: usize,
    max_num_seqs: usize,
    _num_kvcache_blocks: usize,
) -> PyResult<DecodeMeta> {
    let data = bytes_arg(data)?;
    let batch = decode_wire(&data)?;
    let num = batch.num_group_seqs();
    let max_num_blocks = batch
        .block_tables
        .iter()
        .map(|blocks| blocks.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let mut block_tables_flat = vec![0; sp_size * num.max(1) * max_num_blocks];
    if !batch.is_dummy {
        for sp in 0..sp_size {
            for seq_idx in 0..num.max(1) {
                let base = (sp * num.max(1) + seq_idx) * max_num_blocks;
                if let Some(blocks) = batch.block_tables.get(seq_idx) {
                    for (block_idx, block) in blocks.iter().copied().enumerate() {
                        block_tables_flat[base + block_idx] = block;
                    }
                }
            }
        }
    }
    let mut context_lens_flat = vec![0; sp_size * max_num_seqs];
    for i in 0..num.min(max_num_seqs) {
        context_lens_flat[i] = batch.positions.get(i).copied().unwrap_or(0) as i32 + 1;
    }
    Ok(DecodeMeta {
        input_ids: batch.input_ids,
        positions: batch.positions.clone(),
        slot_mapping: if batch.is_dummy {
            vec![-1; num]
        } else {
            batch
                .positions
                .iter()
                .enumerate()
                .map(|(idx, pos)| {
                    let block_table = batch.block_tables.get(idx).cloned().unwrap_or_default();
                    let block_size = _block_size.max(1) as i64;
                    let logical_block = (*pos / block_size).max(0) as usize;
                    let offset = (*pos % block_size) as i32;
                    block_table
                        .get(logical_block)
                        .copied()
                        .map(|block| block * _block_size.max(1) as i32 + offset)
                        .unwrap_or(idx as i32)
                })
                .collect()
        },
        context_lens_flat,
        block_tables_flat: if batch.is_dummy {
            Vec::new()
        } else {
            block_tables_flat
        },
        max_num_blocks: if batch.is_dummy { 0 } else { max_num_blocks },
    })
}

#[pyfunction]
fn extract_vision_slots_from_bytes(_data: &Bound<'_, PyAny>) -> PyResult<Vec<PyObject>> {
    Ok(Vec::new())
}

#[pyfunction]
#[pyo3(signature = (token_ids, logprobs = None, server_handler_ns = 0))]
fn encode_run_result(
    py: Python<'_>,
    token_ids: &Bound<'_, PyAny>,
    logprobs: Option<&Bound<'_, PyAny>>,
    server_handler_ns: u64,
) -> PyResult<PyObject> {
    let payload = WireRunResult {
        server_handler_ns,
        token_ids: token_ids.extract::<Vec<Vec<i64>>>()?,
        logprobs: match logprobs {
            Some(value) if !value.is_none() => Some(value.extract::<Vec<Vec<f32>>>()?),
            _ => None,
        },
    };
    let bytes = encode_binary(&payload, "run result")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
fn decode_run_result(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let data = bytes_arg(data)?;
    let result: WireRunResult = decode_binary(&data, "run result")?;
    let token_ids = result.token_ids.into_pyobject(py)?.unbind().into_any();
    let logprobs = if let Some(logprobs) = result.logprobs {
        logprobs.into_pyobject(py)?.unbind().into_any()
    } else {
        py.None()
    };
    Ok(PyTuple::new(py, [token_ids, logprobs])?.into())
}

#[pyfunction]
#[pyo3(signature = (data_ptr, buffer_size, seqs, _is_prefill))]
fn serialize(
    py: Python<'_>,
    data_ptr: usize,
    buffer_size: usize,
    seqs: &Bound<'_, PyAny>,
    _is_prefill: bool,
) -> PyResult<usize> {
    if data_ptr == 0 {
        return Err(PyValueError::new_err("serialize data_ptr is null"));
    }
    let seqs: Vec<PyObject> = seqs.extract()?;
    let mut wire = Vec::with_capacity(seqs.len());
    for seq in seqs {
        wire.push(sequence_to_wire(seq.bind(py))?);
    }
    let bytes = encode_binary(&wire, "sequences")?;
    if bytes.len() > buffer_size {
        return Err(PyValueError::new_err(format!(
            "serialize buffer too small: need {} bytes, got {}",
            bytes.len(),
            buffer_size
        )));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), data_ptr as *mut u8, bytes.len());
    }
    Ok(bytes.len())
}

#[pyfunction]
#[pyo3(signature = (data_ptr, data_len))]
fn deserialize(py: Python<'_>, data_ptr: usize, data_len: usize) -> PyResult<Vec<PyObject>> {
    if data_ptr == 0 {
        return Err(PyValueError::new_err("deserialize data_ptr is null"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) };
    let wire: Vec<WireSequence> = decode_binary(bytes, "sequences")?;
    wire.into_iter()
        .map(|seq| wire_to_sequence(py, seq))
        .collect()
}

#[pyfunction]
#[pyo3(signature = (seqs))]
fn serialize_migrate_batch(py: Python<'_>, seqs: &Bound<'_, PyAny>) -> PyResult<PyObject> {
    let seqs: Vec<PyObject> = seqs.extract()?;
    let mut wire = Vec::with_capacity(seqs.len());
    for seq in seqs {
        wire.push(sequence_to_migrate_wire(seq.bind(py)));
    }
    let bytes = encode_binary(&wire, "migrate batch")?;
    Ok(PyBytes::new(py, &bytes).into())
}

#[pyfunction]
#[pyo3(signature = (data))]
fn parse_migrate_batch(data: &Bound<'_, PyAny>) -> PyResult<Vec<MigrateSequenceView>> {
    let data = bytes_arg(data)?;
    let wire: Vec<WireMigrateSequence> = decode_binary(&data, "migrate batch")?;
    Ok(wire.into_iter().map(Into::into).collect())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BatchAuxData>()?;
    m.add_class::<PrefillMeta>()?;
    m.add_class::<DecodeMeta>()?;
    m.add_class::<MigrateSequenceView>()?;
    m.add_function(wrap_pyfunction!(serialize, m)?)?;
    m.add_function(wrap_pyfunction!(deserialize, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_migrate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(parse_migrate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_aux_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(extract_vision_slots_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(serialize_dummy_run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_prefill_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(prepare_decode_from_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(encode_run_result, m)?)?;
    m.add_function(wrap_pyfunction!(decode_run_result, m)?)?;
    Ok(())
}
