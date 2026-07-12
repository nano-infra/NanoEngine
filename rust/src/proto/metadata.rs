use super::wire::{
    decode_binary, encode_binary, WireBatch, WireFreeSequences, WireFreeVisionSlots,
    WireMigrateSequence, WirePacket, WireRequestIn, WireRequestMigrate, WireRunResult,
    WireSamplingParams, WireStepOut, WireVisionSlot,
};
use crate::sampling::SamplingParams;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyTuple, PyType};

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct RequestIn {
    #[pyo3(get, set)]
    pub seq_id: u64,
    #[pyo3(get, set)]
    pub prompt_token_ids: Vec<i32>,
    sampling_params: WireSamplingParams,
    #[pyo3(get, set)]
    pub affinity_key: u64,
    #[pyo3(get, set)]
    pub vision_slots: Vec<(String, i32, i32, i32, i32)>,
}

#[pymethods]
impl RequestIn {
    #[new]
    #[pyo3(signature = (seq_id, prompt_token_ids, sampling_params, affinity_key = 0, vision_slots = None))]
    fn new(
        py: Python<'_>,
        seq_id: u64,
        prompt_token_ids: Vec<i32>,
        sampling_params: Py<SamplingParams>,
        affinity_key: u64,
        vision_slots: Option<Vec<(String, i32, i32, i32, i32)>>,
    ) -> Self {
        let sampling_params = sampling_params.borrow(py);
        Self {
            seq_id,
            prompt_token_ids,
            sampling_params: WireSamplingParams::from(&*sampling_params),
            affinity_key,
            vision_slots: vision_slots.unwrap_or_default(),
        }
    }

    #[getter]
    fn sampling_params(&self) -> SamplingParams {
        self.sampling_params.to_sampling_params()
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_binary(&vec![WireRequestIn::from(self)], "add request")?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let data = super::wire::bytes_arg(data)?;
        let requests: Vec<WireRequestIn> = decode_binary(&data, "add request")?;
        let Some(request) = requests.into_iter().next() else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "expected one RequestIn, got 0",
            ));
        };
        Ok(request.into())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<RequestIn>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py)?;
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

impl From<WireRequestIn> for RequestIn {
    fn from(wire: WireRequestIn) -> Self {
        Self {
            seq_id: wire.seq_id,
            prompt_token_ids: wire.prompt_token_ids,
            sampling_params: wire.sampling_params,
            affinity_key: wire.affinity_key,
            vision_slots: wire
                .vision_slots
                .into_iter()
                .map(|slot| {
                    (
                        slot.encoder_engine_id,
                        slot.slot_idx,
                        slot.num_tokens,
                        slot.hidden_size,
                        slot.max_tokens_per_slot,
                    )
                })
                .collect(),
        }
    }
}

impl From<&RequestIn> for WireRequestIn {
    fn from(value: &RequestIn) -> Self {
        Self {
            seq_id: value.seq_id,
            prompt_token_ids: value.prompt_token_ids.clone(),
            sampling_params: value.sampling_params.clone(),
            affinity_key: value.affinity_key,
            vision_slots: value
                .vision_slots
                .iter()
                .map(
                    |(
                        encoder_engine_id,
                        slot_idx,
                        num_tokens,
                        hidden_size,
                        max_tokens_per_slot,
                    )| {
                        WireVisionSlot {
                            encoder_engine_id: encoder_engine_id.clone(),
                            slot_idx: *slot_idx,
                            num_tokens: *num_tokens,
                            hidden_size: *hidden_size,
                            max_tokens_per_slot: *max_tokens_per_slot,
                        }
                    },
                )
                .collect(),
        }
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct RequestMigrate {
    #[pyo3(get, set)]
    pub payload: Vec<u8>,
}

#[pymethods]
impl RequestMigrate {
    #[new]
    fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            payload: super::wire::bytes_arg(payload)?,
        })
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }

    #[getter]
    fn metadata(&self) -> PyResult<(u64, i32)> {
        let request: WireRequestMigrate = decode_binary(&self.payload, "migration request")?;
        let first_token = request
            .token_ids
            .get(request.num_prompt_tokens.max(0) as usize..)
            .and_then(|tokens| tokens.last())
            .copied()
            .unwrap_or(request.last_token);
        Ok((request.seq_id, first_token))
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<RequestMigrate>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py);
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct MigrationIn {
    payload: Vec<u8>,
    #[pyo3(get)]
    pub sequences: Vec<MigrateSequenceView>,
}

#[pymethods]
impl MigrationIn {
    #[new]
    fn new(payload: Vec<u8>) -> PyResult<Self> {
        Self::from_payload(payload)
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::from_payload(super::wire::bytes_arg(payload)?)
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }

    fn __len__(&self) -> usize {
        self.sequences.len()
    }

    fn __getitem__(&self, index: isize) -> PyResult<MigrateSequenceView> {
        let len = self.sequences.len() as isize;
        let index = if index < 0 { len + index } else { index };
        if index < 0 || index >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "MigrationIn index out of range",
            ));
        }
        Ok(self.sequences[index as usize].clone())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<MigrationIn>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py);
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

impl MigrationIn {
    fn from_payload(payload: Vec<u8>) -> PyResult<Self> {
        let wire: Vec<WireMigrateSequence> = decode_binary(&payload, "migrate batch")?;
        Ok(Self {
            payload,
            sequences: wire.into_iter().map(Into::into).collect(),
        })
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct RunnerIn {
    payload: Vec<u8>,
}

#[pymethods]
impl RunnerIn {
    #[new]
    fn new(payload: Vec<u8>) -> Self {
        Self { payload }
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            payload: super::wire::bytes_arg(payload)?,
        })
    }

    #[classmethod]
    #[pyo3(signature = (_engine_id, _num_kvcache_blocks, is_prefill))]
    fn dummy(
        _cls: &Bound<'_, PyType>,
        _engine_id: &str,
        _num_kvcache_blocks: i64,
        is_prefill: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            payload: encode_binary(&WireBatch::dummy(is_prefill), "run batch")?,
        })
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }

    #[getter]
    fn is_prefill(&self) -> PyResult<bool> {
        let batch: WireBatch = decode_binary(&self.payload, "run batch")?;
        Ok(batch.is_prefill)
    }

    #[pyo3(signature = (sp_rank = 0))]
    fn aux(&self, sp_rank: usize) -> PyResult<BatchAuxData> {
        super::prepare::runner_in_aux(&self.payload, sp_rank)
    }

    fn vision_slots(&self) -> PyResult<Vec<PyObject>> {
        super::prepare::runner_in_vision_slots(&self.payload)
    }

    #[pyo3(signature = (sp_rank, sp_size, block_size, max_num_seqs, num_kvcache_blocks))]
    fn prefill(
        &self,
        sp_rank: usize,
        sp_size: usize,
        block_size: usize,
        max_num_seqs: usize,
        num_kvcache_blocks: usize,
    ) -> PyResult<PrefillMeta> {
        super::prepare::runner_in_prefill(
            &self.payload,
            sp_rank,
            sp_size,
            block_size,
            max_num_seqs,
            num_kvcache_blocks,
        )
    }

    #[pyo3(signature = (sp_rank, sp_size, block_size, max_num_seqs, num_kvcache_blocks))]
    fn decode(
        &self,
        sp_rank: usize,
        sp_size: usize,
        block_size: usize,
        max_num_seqs: usize,
        num_kvcache_blocks: usize,
    ) -> PyResult<DecodeMeta> {
        super::prepare::runner_in_decode(
            &self.payload,
            sp_rank,
            sp_size,
            block_size,
            max_num_seqs,
            num_kvcache_blocks,
        )
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<RunnerIn>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py);
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct RunnerOut {
    #[pyo3(get, set)]
    pub token_ids: Vec<Vec<i64>>,
    #[pyo3(get, set)]
    pub logprobs: Option<Vec<Vec<f32>>>,
    #[pyo3(get, set)]
    pub server_handler_ns: u64,
}

#[pymethods]
impl RunnerOut {
    #[new]
    #[pyo3(signature = (token_ids = Vec::new(), logprobs = None, server_handler_ns = 0))]
    fn new(
        token_ids: Vec<Vec<i64>>,
        logprobs: Option<Vec<Vec<f32>>>,
        server_handler_ns: u64,
    ) -> Self {
        Self {
            token_ids,
            logprobs,
            server_handler_ns,
        }
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let payload = super::wire::bytes_arg(payload)?;
        let wire: WireRunResult = decode_binary(&payload, "run result")?;
        Ok(wire.into())
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_binary(&WireRunResult::from(self), "run result")?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[getter]
    fn result(&self, py: Python<'_>) -> PyResult<PyObject> {
        let token_ids = self
            .token_ids
            .clone()
            .into_pyobject(py)?
            .unbind()
            .into_any();
        let logprobs = if let Some(logprobs) = &self.logprobs {
            logprobs.clone().into_pyobject(py)?.unbind().into_any()
        } else {
            py.None()
        };
        Ok(PyTuple::new(py, [token_ids, logprobs])?.into())
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<RunnerOut>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py)?;
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

impl From<WireRunResult> for RunnerOut {
    fn from(wire: WireRunResult) -> Self {
        Self {
            token_ids: wire.token_ids,
            logprobs: wire.logprobs,
            server_handler_ns: wire.server_handler_ns,
        }
    }
}

impl From<&RunnerOut> for WireRunResult {
    fn from(value: &RunnerOut) -> Self {
        Self {
            token_ids: value.token_ids.clone(),
            logprobs: value.logprobs.clone(),
            server_handler_ns: value.server_handler_ns,
        }
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct Packet {
    #[pyo3(get, set)]
    pub action: i32,
    #[pyo3(get, set)]
    pub payload: Vec<u8>,
}

#[pymethods]
impl Packet {
    #[new]
    #[pyo3(signature = (action = 0, payload = Vec::new()))]
    fn new(action: i32, payload: Vec<u8>) -> Self {
        Self { action, payload }
    }
}

impl From<WirePacket> for Packet {
    fn from(wire: WirePacket) -> Self {
        Self {
            action: wire.action,
            payload: wire.payload,
        }
    }
}

impl From<&Packet> for WirePacket {
    fn from(value: &Packet) -> Self {
        Self {
            action: value.action,
            payload: value.payload.clone(),
        }
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct StepOut {
    #[pyo3(get, set)]
    pub seq_id: u64,
    #[pyo3(get, set)]
    pub token_ids: Vec<i32>,
    #[pyo3(get, set)]
    pub status: i32,
}

#[pymethods]
impl StepOut {
    #[new]
    #[pyo3(signature = (seq_id = 0, token_ids = Vec::new(), status = 1))]
    fn new(seq_id: u64, token_ids: Vec<i32>, status: i32) -> Self {
        Self {
            seq_id,
            token_ids,
            status,
        }
    }

    #[getter]
    fn token_id(&self) -> i32 {
        *self.token_ids.last().unwrap_or(&0)
    }

    #[classmethod]
    fn from_bytes(_cls: &Bound<'_, PyType>, payload: &Bound<'_, PyAny>) -> PyResult<Self> {
        let payload = super::wire::bytes_arg(payload)?;
        let wire: WireStepOut = decode_binary(&payload, "stepout")?;
        Ok(wire.into())
    }

    fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = encode_binary(&WireStepOut::from(self), "stepout")?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cls = py.get_type::<StepOut>();
        let callable = cls.getattr("from_bytes")?;
        let state = self.to_bytes(py)?;
        let args = PyTuple::new(py, [state])?;
        Ok((callable, args).into_pyobject(py)?.unbind().into())
    }
}

impl From<WireStepOut> for StepOut {
    fn from(wire: WireStepOut) -> Self {
        Self {
            seq_id: wire.seq_id,
            token_ids: wire.token_ids,
            status: wire.status,
        }
    }
}

impl From<&StepOut> for WireStepOut {
    fn from(value: &StepOut) -> Self {
        Self {
            seq_id: value.seq_id,
            token_ids: value.token_ids.clone(),
            status: value.status,
        }
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct FreeSequences {
    #[pyo3(get, set)]
    pub seq_ids: Vec<u64>,
    #[pyo3(get, set)]
    pub source_engine_id: String,
}

#[pymethods]
impl FreeSequences {
    #[new]
    #[pyo3(signature = (seq_ids = Vec::new(), source_engine_id = String::new()))]
    fn new(seq_ids: Vec<u64>, source_engine_id: String) -> Self {
        Self {
            seq_ids,
            source_engine_id,
        }
    }
}

impl From<WireFreeSequences> for FreeSequences {
    fn from(wire: WireFreeSequences) -> Self {
        Self {
            seq_ids: wire.seq_ids,
            source_engine_id: wire.source_engine_id,
        }
    }
}

impl From<&FreeSequences> for WireFreeSequences {
    fn from(value: &FreeSequences) -> Self {
        Self {
            seq_ids: value.seq_ids.clone(),
            source_engine_id: value.source_engine_id.clone(),
        }
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Debug)]
pub struct FreeVisionSlots {
    #[pyo3(get, set)]
    pub encoder_engine_id: String,
    #[pyo3(get, set)]
    pub slot_indices: Vec<i32>,
    #[pyo3(get, set)]
    pub source_engine_id: String,
}

#[pymethods]
impl FreeVisionSlots {
    #[new]
    #[pyo3(signature = (encoder_engine_id = String::new(), slot_indices = Vec::new(), source_engine_id = String::new()))]
    fn new(encoder_engine_id: String, slot_indices: Vec<i32>, source_engine_id: String) -> Self {
        Self {
            encoder_engine_id,
            slot_indices,
            source_engine_id,
        }
    }
}

impl From<WireFreeVisionSlots> for FreeVisionSlots {
    fn from(wire: WireFreeVisionSlots) -> Self {
        Self {
            encoder_engine_id: wire.encoder_engine_id,
            slot_indices: wire.slot_indices,
            source_engine_id: wire.source_engine_id,
        }
    }
}

impl From<&FreeVisionSlots> for WireFreeVisionSlots {
    fn from(value: &FreeVisionSlots) -> Self {
        Self {
            encoder_engine_id: value.encoder_engine_id.clone(),
            slot_indices: value.slot_indices.clone(),
            source_engine_id: value.source_engine_id.clone(),
        }
    }
}

#[pyclass(module = "dlengine._engine")]
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
    pub seq_ids: Vec<u64>,
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
        seq_ids = Vec::new(),
        any_return_completion_logprobs = false
    ))]
    fn new(
        num_group_seqs: usize,
        temperatures: Vec<f32>,
        state_slots: Vec<i64>,
        compressed_block_tables: std::collections::HashMap<i32, Vec<Vec<i32>>>,
        hisparse_slots: Vec<i64>,
        seq_ids: Vec<u64>,
        any_return_completion_logprobs: bool,
    ) -> Self {
        Self {
            num_group_seqs,
            temperatures,
            state_slots,
            compressed_block_tables,
            hisparse_slots,
            seq_ids,
            any_return_completion_logprobs,
        }
    }
}

#[pyclass(module = "dlengine._engine")]
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

#[pyclass(module = "dlengine._engine")]
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

#[pyclass(module = "dlengine._engine")]
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
