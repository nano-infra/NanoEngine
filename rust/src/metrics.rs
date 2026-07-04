use crate::common::now_seconds;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

#[pyclass(module = "dlengine._engine", subclass)]
pub struct SequenceMetric {
    #[pyo3(get, set)]
    pub seq_id: u64,
    #[pyo3(get, set)]
    pub arrival_time: Option<f64>,
    #[pyo3(get, set)]
    pub first_scheduled_time: Option<f64>,
    #[pyo3(get, set)]
    pub decode_arrival_time: Option<f64>,
    #[pyo3(get, set)]
    pub decode_scheduled_time: Option<f64>,
    #[pyo3(get, set)]
    pub first_token_time: Option<f64>,
    #[pyo3(get, set)]
    pub completion_time: Option<f64>,
    #[pyo3(get, set)]
    pub num_prompt_tokens: i32,
    #[pyo3(get, set)]
    pub num_generated_tokens: i32,
    #[pyo3(get, set)]
    pub itl_samples: Vec<f64>,
    #[pyo3(get, set)]
    pub last_token_time: Option<f64>,
    #[pyo3(get, set)]
    pub num_prefill_chunks: i32,
    #[pyo3(get, set)]
    pub prefill_chunk_samples: Vec<f64>,
    #[pyo3(get, set)]
    pub last_chunk_time: Option<f64>,
}

#[pymethods]
impl SequenceMetric {
    #[new]
    #[pyo3(signature = (seq_id, num_prompt_tokens = 0))]
    fn new(seq_id: u64, num_prompt_tokens: i32) -> Self {
        Self {
            seq_id,
            arrival_time: None,
            first_scheduled_time: None,
            decode_arrival_time: None,
            decode_scheduled_time: None,
            first_token_time: None,
            completion_time: None,
            num_prompt_tokens,
            num_generated_tokens: 0,
            itl_samples: Vec::new(),
            last_token_time: None,
            num_prefill_chunks: 0,
            prefill_chunk_samples: Vec::new(),
            last_chunk_time: None,
        }
    }

    pub(crate) fn record_arrival(&mut self) {
        self.arrival_time = Some(now_seconds());
    }
    pub(crate) fn record_first_scheduled(&mut self) {
        self.first_scheduled_time = Some(now_seconds());
    }
    pub(crate) fn record_decode_arrival(&mut self) {
        self.decode_arrival_time = Some(now_seconds());
    }
    pub(crate) fn record_decode_scheduled(&mut self) {
        self.decode_scheduled_time = Some(now_seconds());
    }
    pub(crate) fn record_first_token(&mut self) {
        let now = now_seconds();
        if self.first_token_time.is_none() {
            self.first_token_time = Some(now);
        }
        self.last_token_time = Some(now);
        self.num_generated_tokens += 1;
    }
    pub(crate) fn record_token(&mut self) {
        let now = now_seconds();
        if let Some(last) = self.last_token_time {
            self.itl_samples.push((now - last) * 1000.0);
        }
        self.last_token_time = Some(now);
        self.num_generated_tokens += 1;
    }
    pub(crate) fn record_completion(&mut self) {
        self.completion_time = Some(now_seconds());
    }
    pub(crate) fn record_prefill_chunk(&mut self) {
        let now = now_seconds();
        let last = self
            .last_chunk_time
            .or(self.first_scheduled_time)
            .unwrap_or(now);
        self.prefill_chunk_samples.push((now - last) * 1000.0);
        self.last_chunk_time = Some(now);
        self.num_prefill_chunks += 1;
    }

    #[getter]
    fn ttft(&self) -> Option<f64> {
        Some((self.first_token_time? - self.arrival_time?) * 1000.0)
    }
    #[getter]
    fn e2e_latency(&self) -> Option<f64> {
        Some((self.completion_time? - self.arrival_time?) * 1000.0)
    }
    #[getter]
    fn avg_tpot_wo_queueing(&self) -> Option<f64> {
        if self.itl_samples.is_empty() {
            None
        } else {
            Some(self.itl_samples.iter().sum::<f64>() / self.itl_samples.len() as f64)
        }
    }
    #[getter]
    fn avg_tpot_with_queueing(&self) -> Option<f64> {
        let total = (self.completion_time? - self.arrival_time?) * 1000.0;
        if self.num_generated_tokens <= 0 {
            None
        } else {
            Some(total / self.num_generated_tokens as f64)
        }
    }
    #[getter]
    fn queueing_time_ms(&self) -> Option<f64> {
        Some((self.first_scheduled_time? - self.arrival_time?) * 1000.0)
    }
    #[getter]
    fn decode_queue_time_ms(&self) -> Option<f64> {
        Some((self.decode_scheduled_time? - self.decode_arrival_time?) * 1000.0)
    }
    #[getter]
    fn avg_itl(&self) -> Option<f64> {
        self.avg_tpot_wo_queueing()
    }
    #[getter]
    fn p50_itl(&self) -> Option<f64> {
        percentile(&self.itl_samples, 0.50)
    }
    #[getter]
    fn p99_itl(&self) -> Option<f64> {
        percentile(&self.itl_samples, 0.99)
    }
    #[getter]
    fn prefill_time_ms(&self) -> Option<f64> {
        if self.prefill_chunk_samples.is_empty() {
            None
        } else {
            Some(self.prefill_chunk_samples.iter().sum())
        }
    }
}

fn percentile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(idx).copied()
}
#[pyclass(module = "dlengine._engine")]
pub struct ServerMetric {
    #[pyo3(get, set)]
    pub total_tokens: i64,
    #[pyo3(get, set)]
    pub total_prompt_tokens: i64,
    #[pyo3(get, set)]
    pub total_generated_tokens: i64,
    #[pyo3(get, set)]
    pub num_running_requests: i32,
    #[pyo3(get, set)]
    pub num_waiting_requests: i32,
    #[pyo3(get, set)]
    pub num_waiting_migration_requests: i32,
    #[pyo3(get, set)]
    pub num_completed_requests: i32,
    #[pyo3(get, set)]
    pub num_waiting_head_blocks: i32,
    #[pyo3(get, set)]
    pub num_waiting_total_blocks: i32,
    #[pyo3(get, set)]
    pub prefill_throughput_samples: Vec<f64>,
    #[pyo3(get, set)]
    pub decode_throughput_samples: Vec<f64>,
    #[pyo3(get, set)]
    pub token_usage_by_dp: HashMap<i32, i64>,
    #[pyo3(get, set)]
    pub group_send_request_counts: HashMap<i32, i64>,
    #[pyo3(get, set)]
    pub group_recv_request_counts: HashMap<i32, i64>,
    #[pyo3(get, set)]
    pub start_time: f64,
}

#[pymethods]
impl ServerMetric {
    #[new]
    fn new() -> Self {
        Self {
            total_tokens: 0,
            total_prompt_tokens: 0,
            total_generated_tokens: 0,
            num_running_requests: 0,
            num_waiting_requests: 0,
            num_waiting_migration_requests: 0,
            num_completed_requests: 0,
            num_waiting_head_blocks: 0,
            num_waiting_total_blocks: 0,
            prefill_throughput_samples: Vec::new(),
            decode_throughput_samples: Vec::new(),
            token_usage_by_dp: HashMap::new(),
            group_send_request_counts: HashMap::new(),
            group_recv_request_counts: HashMap::new(),
            start_time: now_seconds(),
        }
    }

    pub fn update_running_requests(&mut self, count: i32) {
        self.num_running_requests = count;
    }

    pub fn update_waiting_requests(&mut self, count: i32) {
        self.num_waiting_requests = count;
    }

    pub fn update_waiting_migration_requests(&mut self, count: i32) {
        self.num_waiting_migration_requests = count;
    }

    fn add_completed_request(&mut self) {
        self.num_completed_requests += 1;
    }

    pub fn update_waiting_blocks(&mut self, head_blocks: i32, total_blocks: i32) {
        self.num_waiting_head_blocks = head_blocks;
        self.num_waiting_total_blocks = total_blocks;
    }

    #[pyo3(signature = (num_prompt = 0, num_generated = 0))]
    fn add_tokens(&mut self, num_prompt: i64, num_generated: i64) {
        self.total_prompt_tokens += num_prompt;
        self.total_generated_tokens += num_generated;
        self.total_tokens += num_prompt + num_generated;
    }

    fn record_prefill_throughput(&mut self, num_tokens: i64, duration: f64) {
        if duration > 0.0 {
            self.prefill_throughput_samples
                .push(num_tokens as f64 / duration);
        }
    }

    fn record_decode_throughput(&mut self, num_tokens: i64, duration: f64) {
        if duration > 0.0 {
            self.decode_throughput_samples
                .push(num_tokens as f64 / duration);
        }
    }

    fn update_token_usage(&mut self, dp_idx: i32, num_tokens: i64) {
        *self.token_usage_by_dp.entry(dp_idx).or_insert(0) += num_tokens;
    }

    pub fn update_group_stats(
        &mut self,
        group_send_counts: Vec<Vec<i32>>,
        group_recv_counts: Vec<Vec<i32>>,
    ) {
        for (dp_idx, counts) in group_send_counts.iter().enumerate() {
            let group_size = counts.len();
            for (group_id, count) in counts.iter().enumerate() {
                let rank = (dp_idx * group_size + group_id) as i32;
                *self.group_send_request_counts.entry(rank).or_insert(0) += *count as i64;
            }
        }
        for (dp_idx, counts) in group_recv_counts.iter().enumerate() {
            let group_size = counts.len();
            for (group_id, count) in counts.iter().enumerate() {
                let rank = (dp_idx * group_size + group_id) as i32;
                *self.group_recv_request_counts.entry(rank).or_insert(0) += *count as i64;
            }
        }
    }

    #[getter]
    fn total_token_usage(&self) -> i64 {
        self.token_usage_by_dp.values().sum()
    }

    #[getter]
    fn uptime(&self) -> f64 {
        now_seconds() - self.start_time
    }

    #[getter]
    fn current_prefill_throughput(&self) -> f64 {
        self.prefill_throughput_samples
            .last()
            .copied()
            .unwrap_or(0.0)
    }

    #[getter]
    fn current_decode_throughput(&self) -> f64 {
        self.decode_throughput_samples
            .last()
            .copied()
            .unwrap_or(0.0)
    }

    #[getter]
    fn avg_prefill_throughput(&self) -> f64 {
        average(&self.prefill_throughput_samples)
    }

    #[getter]
    fn avg_decode_throughput(&self) -> f64 {
        average(&self.decode_throughput_samples)
    }

    fn get_metric_report(&self, include_detailed: bool) -> String {
        let mut report = format!(
            "ServerMetric - Running/Waiting/Waiting migration: {}/{}/{}, Completed: {}, Total tokens: {}",
            self.num_running_requests,
            self.num_waiting_requests,
            self.num_waiting_migration_requests,
            self.num_completed_requests,
            self.total_tokens
        );
        if include_detailed {
            report.push_str(&format!(
                ", Token usage by DP: {:?}, Waiting blocks: {}/{}",
                self.token_usage_by_dp, self.num_waiting_head_blocks, self.num_waiting_total_blocks
            ));
        }
        report
    }

    fn get_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let summary = PyDict::new(py);
        summary.set_item("uptime_seconds", self.uptime())?;
        summary.set_item("total_tokens", self.total_tokens)?;
        summary.set_item("total_prompt_tokens", self.total_prompt_tokens)?;
        summary.set_item("total_generated_tokens", self.total_generated_tokens)?;
        summary.set_item("num_running_requests", self.num_running_requests)?;
        summary.set_item("num_waiting_requests", self.num_waiting_requests)?;
        summary.set_item(
            "num_waiting_migration_requests",
            self.num_waiting_migration_requests,
        )?;
        summary.set_item("num_completed_requests", self.num_completed_requests)?;
        summary.set_item(
            "current_prefill_throughput",
            self.current_prefill_throughput(),
        )?;
        summary.set_item(
            "current_decode_throughput",
            self.current_decode_throughput(),
        )?;
        summary.set_item("avg_prefill_throughput", self.avg_prefill_throughput())?;
        summary.set_item("avg_decode_throughput", self.avg_decode_throughput())?;
        summary.set_item("total_token_usage", self.total_token_usage())?;
        summary.set_item("token_usage_by_dp", self.token_usage_by_dp.clone())?;
        summary.set_item("num_waiting_head_blocks", self.num_waiting_head_blocks)?;
        summary.set_item("num_waiting_total_blocks", self.num_waiting_total_blocks)?;
        Ok(summary.into())
    }
}

fn average(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SequenceMetric>()?;
    m.add_class::<ServerMetric>()?;
    Ok(())
}
