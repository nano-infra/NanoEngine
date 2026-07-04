use super::runner::RunnerMetricSource;
use super::stats::average;
use crate::common::now_seconds;
use crate::proto::RunnerOut;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

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
        Self::default()
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

    pub(crate) fn add_completed_request(&mut self) {
        self.num_completed_requests += 1;
    }

    pub fn update_waiting_blocks(&mut self, head_blocks: i32, total_blocks: i32) {
        self.num_waiting_head_blocks = head_blocks;
        self.num_waiting_total_blocks = total_blocks;
    }

    #[pyo3(signature = (num_prompt = 0, num_generated = 0))]
    pub(crate) fn add_tokens(&mut self, num_prompt: i64, num_generated: i64) {
        self.total_prompt_tokens += num_prompt;
        self.total_generated_tokens += num_generated;
        self.total_tokens += num_prompt + num_generated;
    }

    pub(crate) fn add_runner_out_tokens(&mut self, runner_out: &RunnerOut) {
        self.add_tokens(0, runner_out.generated_token_count());
    }

    pub(crate) fn record_prefill_throughput(&mut self, num_tokens: i64, duration: f64) {
        if duration > 0.0 {
            self.prefill_throughput_samples
                .push(num_tokens as f64 / duration);
        }
    }

    pub(crate) fn record_decode_throughput(&mut self, num_tokens: i64, duration: f64) {
        if duration > 0.0 {
            self.decode_throughput_samples
                .push(num_tokens as f64 / duration);
        }
    }

    pub(crate) fn update_token_usage(&mut self, dp_idx: i32, num_tokens: i64) {
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
    pub(crate) fn total_token_usage(&self) -> i64 {
        self.token_usage_by_dp.values().sum()
    }

    #[getter]
    pub(crate) fn uptime(&self) -> f64 {
        now_seconds() - self.start_time
    }

    #[getter]
    pub(crate) fn current_prefill_throughput(&self) -> f64 {
        self.prefill_throughput_samples
            .last()
            .copied()
            .unwrap_or(0.0)
    }

    #[getter]
    pub(crate) fn current_decode_throughput(&self) -> f64 {
        self.decode_throughput_samples
            .last()
            .copied()
            .unwrap_or(0.0)
    }

    #[getter]
    pub(crate) fn avg_prefill_throughput(&self) -> f64 {
        average(&self.prefill_throughput_samples)
    }

    #[getter]
    pub(crate) fn avg_decode_throughput(&self) -> f64 {
        average(&self.decode_throughput_samples)
    }

    pub(crate) fn get_metric_report(&self, include_detailed: bool) -> String {
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

    pub(crate) fn get_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
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

impl Default for ServerMetric {
    fn default() -> Self {
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
}
