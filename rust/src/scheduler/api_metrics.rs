use super::*;
use crate::common;
use crate::metrics::{sequence_metric_new, SequenceMetric, ServerMetric};
use crate::proto::RunnerOut;
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::types::PyDict;

impl Scheduler {
    pub(super) fn set_sequence_metric_api(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        metric: Py<SequenceMetric>,
    ) -> bool {
        self.set_sequence_metric_impl(py, seq_id, metric)
    }

    pub(super) fn register_sequence_metric_api(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        num_prompt_tokens: i32,
    ) -> PyResult<bool> {
        let metric = Py::new(py, sequence_metric_new(seq_id, num_prompt_tokens))?;
        self.sequence_metrics.insert(seq_id, metric.clone_ref(py));
        self.record_prompt_tokens_api(num_prompt_tokens as i64);
        Ok(self.set_sequence_metric_impl(py, seq_id, metric))
    }

    pub(super) fn metric_snapshot_api(&self) -> SchedulerMetricSnapshot {
        self.metric_snapshot_impl()
    }

    pub(super) fn update_server_metric_api(
        &self,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> SchedulerMetricSnapshot {
        self.update_server_metric_impl(metric, result)
    }

    pub(super) fn update_server_metric_and_log_api(
        &self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> PyResult<SchedulerMetricSnapshot> {
        self.update_server_metric_and_log_impl(py, metric, result)
    }

    pub(super) fn record_step_metric_api(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        dp_group_token_ids: Option<Vec<Vec<Vec<i32>>>>,
    ) -> StepMetricSnapshot {
        self.record_step_metric_impl(py, metric, result, dp_group_token_ids)
    }

    pub(super) fn record_step_metric_runner_outs_api(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        runner_outs: Vec<RunnerOut>,
    ) -> StepMetricSnapshot {
        let (token_ids, _) = Self::runner_outs_to_step_output(&runner_outs);
        self.record_step_metric_impl(py, metric, result, Some(token_ids))
    }

    pub(super) fn record_step_metrics_and_report_api(
        &mut self,
        py: Python<'_>,
        result: &ScheduleResult,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
        runner_outs: Option<Vec<RunnerOut>>,
    ) -> PyResult<StepResult> {
        let postprocess_timing = result.postprocess_timing.clone().unwrap_or_else(|| {
            let now = common::now_seconds();
            PostprocessTiming {
                begin_s: now,
                end_s: now,
                latency_ms: 0.0,
            }
        });
        let scheduler_metric = result
            .scheduler_metric
            .clone()
            .unwrap_or_else(|| self.metric_snapshot_impl());
        let mut runtime = std::mem::take(&mut self.runtime_metrics);
        let mut metric = std::mem::take(&mut self.server_metric);
        let (metric_snapshot, status_message) = self.record_step_metrics_and_report_impl(
            py,
            &mut runtime,
            &mut metric,
            &scheduler_metric,
            result,
            postprocess_timing.latency_ms,
            postprocess_timing.begin_s,
            runner_outs,
        );
        self.runtime_metrics = runtime;
        self.server_metric = metric;
        let (outputs, completed_seq_ids) = self.collect_sequence_events_impl(
            py,
            result.dp_seq_ids.clone(),
            track_running,
            previous_running,
        )?;
        for seq_id in &completed_seq_ids {
            self.clear_finished_metric_state_impl(*seq_id);
        }
        let mut log_messages = Vec::new();
        for seq_id in &completed_seq_ids {
            if let Some(metric) = self.sequence_metrics.get(seq_id).map(|m| m.clone_ref(py)) {
                if let Some(message) = self.complete_sequence_metric_py(py, &metric) {
                    log_messages.push(message);
                }
            }
        }
        Ok(StepResult {
            outputs,
            prefill_tokens: metric_snapshot.prefill_tokens,
            decode_tokens: metric_snapshot.decode_tokens,
            real_bs: metric_snapshot.real_bs,
            schedule_latency_ms: result.schedule_latency_ms,
            postprocess_latency_ms: postprocess_timing.latency_ms,
            status_message,
            log_messages,
            completed_seq_ids,
        })
    }

    pub(super) fn record_prompt_tokens_api(&mut self, num_prompt_tokens: i64) {
        self.server_metric.add_tokens(num_prompt_tokens, 0);
    }

    pub(super) fn record_sequence_completion_api(
        &mut self,
        _py: Python<'_>,
        metric: &mut SequenceMetric,
    ) -> bool {
        let mut runtime = std::mem::take(&mut self.runtime_metrics);
        let mut server_metric = std::mem::take(&mut self.server_metric);
        let should_log = runtime.record_sequence_completion(metric, &mut server_metric);
        self.runtime_metrics = runtime;
        self.server_metric = server_metric;
        should_log
    }

    pub(super) fn complete_sequence_metric_api(
        &mut self,
        _py: Python<'_>,
        metric: &mut SequenceMetric,
    ) -> Option<String> {
        self.record_sequence_completion_api(_py, metric)
            .then(|| metric.metric_report())
    }

    pub(super) fn complete_sequence_by_id_api(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
    ) -> Option<String> {
        let metric = self
            .sequence_metrics
            .get(&seq_id)
            .map(|m| m.clone_ref(py))?;
        self.complete_sequence_metric_py(py, &metric)
    }

    pub(super) fn record_step_throughput_api(
        &mut self,
        prefill_tokens: i32,
        decode_tokens: i32,
        duration_s: f64,
    ) {
        if prefill_tokens > 0 {
            self.server_metric
                .record_prefill_throughput(prefill_tokens as i64, duration_s);
        }
        if decode_tokens > 0 {
            self.server_metric
                .record_decode_throughput(decode_tokens as i64, duration_s);
        }
    }

    pub(super) fn metric_report_api(&self, include_detailed: bool) -> String {
        self.server_metric.get_metric_report(include_detailed)
    }

    pub(super) fn metric_summary_api(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.server_metric.get_summary(py)
    }

    pub(super) fn metrics_prometheus_api(&self) -> String {
        self.runtime_metrics.to_prometheus(&self.server_metric)
    }

    pub(super) fn final_metric_report_api(&self, py: Python<'_>) -> PyResult<String> {
        let mut lines = Vec::new();
        let sep = "=".repeat(60);
        lines.push(sep.clone());
        lines.push("Final Server Metrics Summary".to_string());
        lines.push(sep.clone());
        lines.push(self.server_metric.get_metric_report(true));

        let summary = self.server_metric.get_summary(py)?;
        let dict = summary.bind(py).downcast::<PyDict>()?;
        for item in dict.items() {
            let key = item.get_item(0)?;
            let value = item.get_item(1)?;
            if !value.is_none() {
                lines.push(format!("  {}: {}", key.str()?, value.str()?));
            }
        }

        let borrowed = self
            .sequence_metrics
            .values()
            .map(|m| m.borrow(py))
            .collect::<Vec<_>>();
        let itl_values = borrowed
            .iter()
            .filter_map(|m| m.avg_tpot_wo_queueing())
            .collect::<Vec<_>>();
        if !itl_values.is_empty() {
            lines.push(format!(
                "  Per-Sequence ITL (from timestamps): mean={:.2}ms, median={:.2}ms, p99={:.2}ms, n={}",
                average_f64(&itl_values),
                percentile_f64(&itl_values, 0.50).unwrap_or(0.0),
                percentile_f64(&itl_values, 0.99).unwrap_or(0.0),
                itl_values.len()
            ));
        }

        let mut all_itl = Vec::new();
        for metric in &borrowed {
            all_itl.extend(metric.itl_samples.iter().copied());
        }
        if !all_itl.is_empty() {
            lines.push(format!(
                "  ITL w/o first token (per-token samples): mean={:.2}ms, median={:.2}ms, p99={:.2}ms, n={}",
                average_f64(&all_itl),
                percentile_f64(&all_itl, 0.50).unwrap_or(0.0),
                percentile_f64(&all_itl, 0.99).unwrap_or(0.0),
                all_itl.len()
            ));
        }

        let uptime = self.server_metric.uptime();
        let total_gen = self.server_metric.total_generated_tokens;
        if uptime > 0.0 && total_gen > 0 {
            lines.push(format!(
                "  Effective decode throughput (wall-clock): {:.0} tok/s ({} tokens / {:.1}s)",
                total_gen as f64 / uptime,
                total_gen,
                uptime
            ));
        }
        lines.push(sep);
        Ok(lines.join("\n"))
    }
}
