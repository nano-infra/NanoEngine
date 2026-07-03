use crate::config::CachePlan;
use crate::sequence::Sequence;
use pyo3::prelude::*;
use pyo3::types::PyType;

#[pyclass(module = "dlengine._dlengine_rust")]
pub struct RoutingStrategy;

#[pymethods]
impl RoutingStrategy {
    #[classattr]
    pub const RoundRobin: i32 = 0;
    #[classattr]
    pub const LeastBatch: i32 = 1;
    #[classattr]
    pub const LeastCache: i32 = 2;
    #[classattr]
    pub const SessionPrefix: i32 = 3;

    #[classmethod]
    fn __class_getitem__(_cls: &Bound<'_, PyType>, name: &str) -> PyResult<i32> {
        match name {
            "RoundRobin" => Ok(Self::RoundRobin),
            "LeastBatch" => Ok(Self::LeastBatch),
            "LeastCache" => Ok(Self::LeastCache),
            "SessionPrefix" => Ok(Self::SessionPrefix),
            _ => Err(pyo3::exceptions::PyKeyError::new_err(name.to_string())),
        }
    }
}

#[pyclass(module = "dlengine._dlengine_rust")]
#[derive(Clone)]
pub struct SchedulerConfig {
    #[pyo3(get, set)]
    pub engine_id: String,
    #[pyo3(get, set)]
    pub num_speculative_tokens: i32,
    #[pyo3(get, set)]
    pub max_num_seqs: i32,
    #[pyo3(get, set)]
    pub max_num_batched_tokens: i32,
    #[pyo3(get, set)]
    pub max_model_len: i32,
    #[pyo3(get, set)]
    pub eos_ids: Vec<i32>,
    #[pyo3(get, set)]
    pub attention_dp: i32,
    #[pyo3(get, set)]
    pub group_size: i32,
    #[pyo3(get, set)]
    pub num_kvcache_blocks: i32,
    #[pyo3(get, set)]
    pub kvcache_block_size: i32,
    #[pyo3(get, set)]
    pub mode: String,
    #[pyo3(get, set)]
    pub routing_strategy: i32,
    #[pyo3(get, set)]
    pub gdn_state_cache_slots: i32,
    #[pyo3(get, set)]
    pub cache_plan: CachePlan,
}

#[pymethods]
impl SchedulerConfig {
    #[new]
    #[pyo3(signature = (
        engine_id = String::new(),
        num_speculative_tokens = 0,
        max_num_seqs = 0,
        max_num_batched_tokens = 0,
        max_model_len = 0,
        eos_ids = Vec::new(),
        attention_dp = 1,
        group_size = 1,
        num_kvcache_blocks = 0,
        kvcache_block_size = 0,
        mode = String::from("hybrid"),
        routing_strategy = RoutingStrategy::RoundRobin,
        gdn_state_cache_slots = 0,
        cache_plan = CachePlan::new(0)
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        engine_id: String,
        num_speculative_tokens: i32,
        max_num_seqs: i32,
        max_num_batched_tokens: i32,
        max_model_len: i32,
        eos_ids: Vec<i32>,
        attention_dp: i32,
        group_size: i32,
        num_kvcache_blocks: i32,
        kvcache_block_size: i32,
        mode: String,
        routing_strategy: i32,
        gdn_state_cache_slots: i32,
        cache_plan: CachePlan,
    ) -> Self {
        Self {
            engine_id,
            num_speculative_tokens,
            max_num_seqs,
            max_num_batched_tokens,
            max_model_len,
            eos_ids,
            attention_dp,
            group_size,
            num_kvcache_blocks,
            kvcache_block_size,
            mode,
            routing_strategy,
            gdn_state_cache_slots,
            cache_plan,
        }
    }
}

#[pyclass(module = "dlengine._dlengine_rust", unsendable)]
#[derive(Default)]
pub struct ScheduleResult {
    #[pyo3(get, set)]
    pub dp_seqs: Vec<Vec<Py<Sequence>>>,
    #[pyo3(get, set)]
    pub dp_group_seqs: Vec<Vec<Py<Sequence>>>,
    #[pyo3(get, set)]
    pub filtered_dp_group_seqs: Vec<Vec<Py<Sequence>>>,
    #[pyo3(get, set)]
    pub is_prefill: bool,
    #[pyo3(get, set)]
    pub group_send_counts: Vec<Vec<i32>>,
    #[pyo3(get, set)]
    pub group_recv_counts: Vec<Vec<i32>>,
    #[pyo3(get, set)]
    pub group_q_matrix: Vec<Vec<Vec<i32>>>,
    #[pyo3(get, set)]
    pub waiting_head_blocks: i32,
    #[pyo3(get, set)]
    pub waiting_total_blocks: i32,
}

#[pymethods]
impl ScheduleResult {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}
