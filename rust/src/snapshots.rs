use pyo3::prelude::*;

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Default)]
pub struct StepMetricSnapshot {
    #[pyo3(get, set)]
    pub prefill_tokens: i32,
    #[pyo3(get, set)]
    pub decode_tokens: i32,
    #[pyo3(get, set)]
    pub real_bs: i32,
    #[pyo3(get, set)]
    pub prefill_tokens_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub decode_tokens_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub prefix_cached_tokens_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub prefix_prompt_tokens_per_dp: Vec<i32>,
}

#[pymethods]
impl StepMetricSnapshot {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

#[pyclass(module = "dlengine._engine")]
#[derive(Clone, Default)]
pub struct SchedulerMetricSnapshot {
    #[pyo3(get, set)]
    pub running_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub total_waiting: i32,
    #[pyo3(get, set)]
    pub total_waiting_migration: i32,
    #[pyo3(get, set)]
    pub waiting_migration_head_tokens: i32,
    #[pyo3(get, set)]
    pub total_blocks_per_dp: i32,
    #[pyo3(get, set)]
    pub used_blocks_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub total_host_blocks_per_dp: i32,
    #[pyo3(get, set)]
    pub used_host_blocks_per_dp: Vec<i32>,
    #[pyo3(get, set)]
    pub free_blocks: Vec<Vec<i32>>,
    #[pyo3(get, set)]
    pub used_hisparse_slots: i32,
    #[pyo3(get, set)]
    pub total_hisparse_slots: i32,
}

#[pymethods]
impl SchedulerMetricSnapshot {
    #[new]
    fn new() -> Self {
        Self {
            waiting_migration_head_tokens: -1,
            ..Self::default()
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<StepMetricSnapshot>()?;
    m.add_class::<SchedulerMetricSnapshot>()?;
    Ok(())
}
