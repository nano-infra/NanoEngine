use crate::config::CachePlan;
use crate::snapshots::SchedulerMetricSnapshot;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyType;

#[pyclass(module = "dlengine._engine")]
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

#[pyclass(module = "dlengine._engine")]
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
    pub enable_prefix_cache: bool,
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
        enable_prefix_cache = true,
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
        enable_prefix_cache: bool,
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
            enable_prefix_cache,
            cache_plan,
        }
    }
}

#[pyclass(module = "dlengine._engine", unsendable)]
#[derive(Default)]
pub struct ScheduleResult {
    #[pyo3(get, set)]
    pub scheduler_metric: Option<SchedulerMetricSnapshot>,
    #[pyo3(get, set)]
    pub postprocess_timing: Option<PostprocessTiming>,
    #[pyo3(get, set)]
    pub schedule_begin_s: f64,
    #[pyo3(get, set)]
    pub schedule_end_s: f64,
    #[pyo3(get, set)]
    pub schedule_latency_ms: f64,
    #[pyo3(get, set)]
    pub dp_seq_ids: Vec<Vec<u64>>,
    #[pyo3(get, set)]
    pub dp_group_seq_ids: Vec<Vec<u64>>,
    #[pyo3(get, set)]
    pub filtered_dp_group_seq_ids: Vec<Vec<u64>>,
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

#[pyclass(module = "dlengine._engine", unsendable)]
pub struct StepResult {
    #[pyo3(get, set)]
    pub outputs: Vec<PyObject>,
    #[pyo3(get, set)]
    pub prefill_tokens: i32,
    #[pyo3(get, set)]
    pub decode_tokens: i32,
    #[pyo3(get, set)]
    pub real_bs: i32,
    #[pyo3(get, set)]
    pub schedule_latency_ms: f64,
    #[pyo3(get, set)]
    pub postprocess_latency_ms: f64,
    #[pyo3(get, set)]
    pub status_message: Option<String>,
    #[pyo3(get, set)]
    pub log_messages: Vec<String>,
    #[pyo3(get, set)]
    pub completed_seq_ids: Vec<u64>,
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Default)]
pub struct PostprocessTiming {
    #[pyo3(get, set)]
    pub begin_s: f64,
    #[pyo3(get, set)]
    pub end_s: f64,
    #[pyo3(get, set)]
    pub latency_ms: f64,
}

#[pymethods]
impl ScheduleResult {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    pub(crate) fn debug_summary(
        &self,
        py: Python<'_>,
        dp_size: usize,
        sp_size: usize,
        free_blocks: Vec<Vec<i32>>,
    ) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        let mut group_batch_sizes = Vec::with_capacity(dp_size);
        for dp_idx in 0..dp_size {
            let mut per_sp = Vec::with_capacity(sp_size);
            for sp_idx in 0..sp_size {
                let idx = dp_idx * sp_size + sp_idx;
                per_sp.push(
                    self.filtered_dp_group_seq_ids
                        .get(idx)
                        .map(|seqs| seqs.len() as i32)
                        .unwrap_or(0),
                );
            }
            group_batch_sizes.push(per_sp);
        }
        dict.set_item("mode", if self.is_prefill { "prefill" } else { "decode" })?;
        dict.set_item("group_batch_sizes", group_batch_sizes)?;
        dict.set_item("group_send_counts", self.group_send_counts.clone())?;
        dict.set_item("group_recv_counts", self.group_recv_counts.clone())?;
        dict.set_item("waiting_head_blocks", self.waiting_head_blocks)?;
        dict.set_item("waiting_total_blocks", self.waiting_total_blocks)?;
        dict.set_item("group_q_matrix", self.group_q_matrix.clone())?;
        dict.set_item("free_blocks", free_blocks)?;
        Ok(dict.into())
    }
}
