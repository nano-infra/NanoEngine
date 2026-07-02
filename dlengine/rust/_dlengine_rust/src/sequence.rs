use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

static NEXT_SEQ_ID: AtomicU64 = AtomicU64::new(0);
static BLOCK_SIZE: AtomicI32 = AtomicI32::new(256);

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone)]
pub struct SamplingParams {
    #[pyo3(get, set)]
    pub temperature: f64,
    #[pyo3(get, set)]
    pub max_tokens: i32,
    #[pyo3(get, set)]
    pub ignore_eos: bool,
    #[pyo3(get, set)]
    pub return_completion_logprobs: bool,
}

#[pymethods]
impl SamplingParams {
    #[new]
    #[pyo3(signature = (temperature = 1.0, max_tokens = 256, ignore_eos = false, return_completion_logprobs = false))]
    fn new(
        temperature: f64,
        max_tokens: i32,
        ignore_eos: bool,
        return_completion_logprobs: bool,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos,
            return_completion_logprobs,
        }
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
pub struct SequenceStatus;

#[pymethods]
impl SequenceStatus {
    #[classattr]
    const WAITING: i32 = 0;
    #[classattr]
    const RUNNING: i32 = 1;
    #[classattr]
    const FINISHED: i32 = 2;
    #[classattr]
    const TO_BE_MIGRATED: i32 = 3;
    #[classattr]
    const PREFILLING: i32 = 4;
}

#[derive(Clone)]
struct VisionSlot {
    encoder_engine_id: String,
    slot_idx: i32,
    num_tokens: i32,
    hidden_size: i32,
    max_tokens_per_slot: i32,
}

#[pyclass(module = "dlengine._dlengine_rust", unsendable)]
pub struct Sequence {
    seq_id: u64,
    status: i32,
    token_ids: Vec<i32>,
    last_token: i32,
    num_tokens: i32,
    num_prompt_tokens: i32,
    #[allow(dead_code)]
    num_checkpointed_tokens: i32,
    #[allow(dead_code)]
    num_cached_tokens: i32,
    affinity_key: u64,
    sampling_params: SamplingParams,
    completion_logprobs: Vec<f32>,
    vision_slots: Vec<VisionSlot>,
    metric: Option<Py<PyAny>>,
    active_block_table: Vec<i32>,
    active_block_tables: HashMap<i32, Vec<i32>>,
    active_dispatched_tokens: Vec<i32>,
    migrate_block_table: Vec<i32>,
    migrate_block_tables: HashMap<i32, Vec<i32>>,
    migrate_engine_id: String,
    migrate_num_kvcache_blocks: i32,
    migrate_group_size: i32,
    migrate_dp_idx: i32,
    active_dp_idx: i32,
    active_group_id: i32,
    migrate_group_id: i32,
    active_state_slot: i32,
    migrate_state_slot: i32,
    active_compressed_block_tables: HashMap<i32, Vec<i32>>,
    migrate_compressed_block_tables: HashMap<i32, Vec<i32>>,
    active_hisparse_slot: i32,
    migrate_hisparse_slot: i32,
}

#[pymethods]
impl Sequence {
    #[new]
    #[pyo3(signature = (token_ids, sampling_params = None))]
    fn new(token_ids: Vec<i32>, sampling_params: Option<SamplingParams>) -> Self {
        let last_token = token_ids.last().copied().unwrap_or(-1);
        let num_tokens = token_ids.len() as i32;
        Self {
            seq_id: NEXT_SEQ_ID.fetch_add(1, Ordering::Relaxed),
            status: SequenceStatus::WAITING,
            token_ids,
            last_token,
            num_tokens,
            num_prompt_tokens: num_tokens,
            num_checkpointed_tokens: num_tokens,
            num_cached_tokens: 0,
            affinity_key: 0,
            sampling_params: sampling_params
                .unwrap_or_else(|| SamplingParams::new(1.0, 256, false, false)),
            completion_logprobs: Vec::new(),
            vision_slots: Vec::new(),
            metric: None,
            active_block_table: Vec::new(),
            active_block_tables: HashMap::new(),
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

    #[getter]
    fn seq_id(&self) -> u64 {
        self.seq_id
    }

    #[setter]
    fn set_seq_id(&mut self, seq_id: u64) {
        self.seq_id = seq_id;
    }

    #[getter]
    fn last_token(&self) -> i32 {
        self.last_token
    }

    #[setter]
    fn set_last_token(&mut self, last_token: i32) {
        self.last_token = last_token;
    }

    #[getter]
    fn num_tokens(&self) -> i32 {
        self.num_tokens
    }

    #[setter]
    fn set_num_tokens(&mut self, num_tokens: i32) {
        self.num_tokens = num_tokens.max(0);
    }

    #[getter]
    fn token_ids(&self) -> Vec<i32> {
        self.token_ids.clone()
    }

    #[setter]
    fn set_token_ids(&mut self, token_ids: Vec<i32>) {
        self.last_token = token_ids.last().copied().unwrap_or(-1);
        self.num_tokens = token_ids.len() as i32;
        self.token_ids = token_ids;
    }

    #[getter]
    fn num_checkpointed_tokens(&self) -> i32 {
        self.num_checkpointed_tokens
    }

    #[setter]
    fn set_num_checkpointed_tokens(&mut self, num_checkpointed_tokens: i32) {
        self.num_checkpointed_tokens = num_checkpointed_tokens.max(0);
    }

    #[getter]
    fn num_prompt_tokens(&self) -> i32 {
        self.num_prompt_tokens
    }

    #[setter]
    fn set_num_prompt_tokens(&mut self, num_prompt_tokens: i32) {
        self.num_prompt_tokens = num_prompt_tokens;
    }

    #[getter]
    fn num_cached_tokens(&self) -> i32 {
        self.num_cached_tokens
    }

    #[setter]
    fn set_num_cached_tokens(&mut self, num_cached_tokens: i32) {
        self.num_cached_tokens = num_cached_tokens;
    }

    #[getter]
    fn status(&self) -> i32 {
        self.status
    }

    #[setter]
    fn set_status(&mut self, status: i32) {
        self.status = status;
    }

    #[getter]
    fn affinity_key(&self) -> u64 {
        self.affinity_key
    }

    #[setter]
    fn set_affinity_key(&mut self, affinity_key: u64) {
        self.affinity_key = affinity_key;
    }

    #[getter]
    fn sampling_params(&self) -> SamplingParams {
        self.sampling_params.clone()
    }

    #[getter]
    fn metric(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.metric.as_ref().map(|m| m.clone_ref(py))
    }

    #[setter]
    fn set_metric(&mut self, metric: Option<Py<PyAny>>) {
        self.metric = metric;
    }

    #[getter]
    fn is_finished(&self) -> bool {
        self.status == SequenceStatus::FINISHED
    }

    #[getter]
    fn is_to_be_migrated(&self) -> bool {
        self.status == SequenceStatus::TO_BE_MIGRATED
    }

    #[getter]
    fn prompt_token_ids(&self) -> Vec<i32> {
        self.token_ids
            .iter()
            .take(self.num_prompt_tokens.max(0) as usize)
            .copied()
            .collect()
    }

    #[getter]
    fn completion_token_ids(&self) -> Vec<i32> {
        self.token_ids
            .iter()
            .skip(self.num_prompt_tokens.max(0) as usize)
            .copied()
            .collect()
    }

    #[getter]
    fn completion_logprobs(&self) -> Vec<f32> {
        self.completion_logprobs.clone()
    }

    fn migrate_engine_id(&self) -> String {
        self.migrate_engine_id.clone()
    }

    fn set_migrate_engine_id(&mut self, value: String) {
        self.migrate_engine_id = value;
    }

    #[getter]
    fn migrate_num_kvcache_blocks(&self) -> i32 {
        self.migrate_num_kvcache_blocks
    }

    #[setter]
    fn set_migrate_num_kvcache_blocks(&mut self, value: i32) {
        self.migrate_num_kvcache_blocks = value;
    }

    #[getter]
    fn migrate_group_size(&self) -> i32 {
        self.migrate_group_size
    }

    #[setter]
    fn set_migrate_group_size(&mut self, value: i32) {
        self.migrate_group_size = value.max(1);
    }

    #[getter]
    fn migrate_dp_idx(&self) -> i32 {
        self.migrate_dp_idx
    }

    #[setter]
    fn set_migrate_dp_idx(&mut self, value: i32) {
        self.migrate_dp_idx = value;
    }

    #[getter]
    fn active_dp_idx(&self) -> i32 {
        self.active_dp_idx
    }

    #[setter]
    fn set_active_dp_idx(&mut self, value: i32) {
        self.active_dp_idx = value;
    }

    #[getter]
    fn active_group_id(&self) -> i32 {
        self.active_group_id
    }

    #[setter]
    fn set_active_group_id(&mut self, value: i32) {
        self.active_group_id = value.max(0);
    }

    #[getter]
    fn migrate_group_id(&self) -> i32 {
        self.migrate_group_id
    }

    #[setter]
    fn set_migrate_group_id(&mut self, value: i32) {
        self.migrate_group_id = value.max(0);
    }

    #[getter]
    fn migrate_block_location(&self) -> Vec<(i32, i32)> {
        self.migrate_block_tables
            .get(&self.migrate_group_id)
            .unwrap_or(&self.migrate_block_table)
            .iter()
            .copied()
            .map(|block| (self.migrate_group_id, block))
            .collect()
    }

    #[getter]
    fn active_block_location(&self) -> Vec<(i32, i32)> {
        self.active_block_tables
            .get(&self.active_group_id)
            .unwrap_or(&self.active_block_table)
            .iter()
            .copied()
            .map(|block| (self.active_group_id, block))
            .collect()
    }

    #[getter]
    fn migrate_state_slot(&self) -> i32 {
        self.migrate_state_slot
    }

    #[setter]
    fn set_migrate_state_slot(&mut self, value: i32) {
        self.migrate_state_slot = value;
    }

    #[getter]
    fn active_state_slot(&self) -> i32 {
        self.active_state_slot
    }

    #[setter]
    fn set_active_state_slot(&mut self, value: i32) {
        self.active_state_slot = value;
    }

    #[getter]
    fn migrate_compressed_block_tables(&self) -> HashMap<i32, Vec<i32>> {
        self.migrate_compressed_block_tables.clone()
    }

    #[getter]
    fn active_compressed_block_tables(&self) -> HashMap<i32, Vec<i32>> {
        self.active_compressed_block_tables.clone()
    }

    fn set_active_compressed_block_table(&mut self, ratio: i32, blocks: Vec<i32>) {
        if ratio > 0 {
            self.active_compressed_block_tables.insert(ratio, blocks);
        }
    }

    fn set_migrate_compressed_block_table(&mut self, ratio: i32, blocks: Vec<i32>) {
        if ratio > 0 {
            self.migrate_compressed_block_tables.insert(ratio, blocks);
        }
    }

    fn clear_active_compressed_block_tables(&mut self) {
        self.active_compressed_block_tables.clear();
    }

    fn clear_migrate_compressed_block_tables(&mut self) {
        self.migrate_compressed_block_tables.clear();
    }

    #[getter]
    fn active_hisparse_slot(&self) -> i32 {
        self.active_hisparse_slot
    }

    #[setter]
    fn set_active_hisparse_slot(&mut self, value: i32) {
        self.active_hisparse_slot = value;
    }

    #[getter]
    fn migrate_hisparse_slot(&self) -> i32 {
        self.migrate_hisparse_slot
    }

    #[setter]
    fn set_migrate_hisparse_slot(&mut self, value: i32) {
        self.migrate_hisparse_slot = value;
    }

    #[pyo3(signature = (token_id, _slot = 0, _group_id = None, logprob = None))]
    fn append_token(
        &mut self,
        token_id: i32,
        _slot: i32,
        _group_id: Option<i32>,
        logprob: Option<f32>,
    ) {
        self.token_ids.push(token_id);
        self.last_token = token_id;
        if let Some(logprob) = logprob {
            self.completion_logprobs.push(logprob);
        }
        self.num_tokens = self.token_ids.len() as i32;
    }

    #[pyo3(signature = (_slot = 0))]
    fn state_slot(&self, _slot: i32) -> i32 {
        self.active_state_slot
    }

    #[pyo3(signature = (_slot = 0, _group_id = 0))]
    fn block_table(&self, _slot: i32, _group_id: i32) -> Vec<i32> {
        self.active_block_tables
            .get(&_group_id)
            .cloned()
            .unwrap_or_else(|| self.active_block_table.clone())
    }

    fn set_active_block_table(&mut self, blocks: Vec<i32>) {
        self.set_active_group_block_table(self.active_group_id, blocks);
    }

    fn set_active_group_block_table(&mut self, group_id: i32, blocks: Vec<i32>) {
        let group_id = group_id.max(0);
        self.active_group_id = group_id;
        self.active_block_table = blocks.clone();
        self.active_block_tables.insert(group_id, blocks);
    }

    fn set_migrate_block_table(&mut self, blocks: Vec<i32>) {
        self.set_migrate_group_block_table(self.migrate_group_id, blocks);
    }

    fn set_migrate_group_block_table(&mut self, group_id: i32, blocks: Vec<i32>) {
        let group_id = group_id.max(0);
        self.migrate_group_id = group_id;
        self.migrate_block_table = blocks.clone();
        self.migrate_block_tables.insert(group_id, blocks);
    }

    #[getter]
    fn active_block_table(&self) -> Vec<i32> {
        self.active_block_table.clone()
    }

    #[getter]
    fn active_block_tables(&self) -> HashMap<i32, Vec<i32>> {
        self.active_block_tables.clone()
    }

    #[getter]
    fn migrate_block_tables(&self) -> HashMap<i32, Vec<i32>> {
        self.migrate_block_tables.clone()
    }

    fn set_active_dispatched_tokens(&mut self, tokens: Vec<i32>) {
        if tokens.is_empty() {
            return;
        }
        let max_group = tokens
            .iter()
            .enumerate()
            .max_by_key(|(_, tokens)| *tokens)
            .map(|(idx, _)| idx as i32)
            .unwrap_or(0);
        self.active_group_id = max_group;
        self.migrate_group_id = max_group;
        self.active_dispatched_tokens = tokens;
    }

    #[getter]
    fn active_dispatched_tokens(&self) -> Vec<i32> {
        self.active_dispatched_tokens.clone()
    }

    fn clear_vision_slots(&mut self) {
        self.vision_slots.clear();
    }

    fn add_vision_slot(
        &mut self,
        encoder_engine_id: String,
        slot_idx: i32,
        num_tokens: i32,
        hidden_size: i32,
        max_tokens_per_slot: i32,
    ) {
        self.vision_slots.push(VisionSlot {
            encoder_engine_id,
            slot_idx,
            num_tokens,
            hidden_size,
            max_tokens_per_slot,
        });
    }

    #[getter]
    fn vision_slots(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let mut out = Vec::with_capacity(self.vision_slots.len());
        for slot in &self.vision_slots {
            let d = PyDict::new(py);
            d.set_item("encoder_engine_id", &slot.encoder_engine_id)?;
            d.set_item("slot_idx", slot.slot_idx)?;
            d.set_item("num_tokens", slot.num_tokens)?;
            d.set_item("hidden_size", slot.hidden_size)?;
            d.set_item("max_tokens_per_slot", slot.max_tokens_per_slot)?;
            out.push(d.unbind());
        }
        Ok(out)
    }

    #[classattr]
    fn block_size() -> i32 {
        BLOCK_SIZE.load(Ordering::Relaxed)
    }

    #[staticmethod]
    fn set_block_size(block_size: i32) {
        BLOCK_SIZE.store(block_size.max(1), Ordering::Relaxed);
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SamplingParams>()?;
    m.add_class::<SequenceStatus>()?;
    m.add_class::<Sequence>()?;
    Ok(())
}
