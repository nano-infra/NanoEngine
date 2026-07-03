use super::wire::WireMigrateSequence;
use pyo3::prelude::*;

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
