use super::*;
use crate::common;
use crate::metrics::{RuntimeMetrics, ServerMetric};
use crate::proto::wire::{
    bytes_arg, decode_binary, migrate_request_to_sequence, request_to_sequence,
    sequence_refs_migrate_batch_bytes, sequence_refs_runner_in_bytes, WireRequestIn,
    WireRequestMigrate, WireSamplingParams, WireVisionSlot,
};
use crate::proto::RunnerOut;
use crate::sampling::SamplingParams;
use crate::sequence::set_sequence_block_size;
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::types::{PyAny, PyBytes};
use std::time::Instant;

#[pymethods]
impl Scheduler {
    #[new]
    pub(super) fn new(config: SchedulerConfig) -> Self {
        set_sequence_block_size(config.kvcache_block_size);
        let dp = config.attention_dp.max(1) as usize;
        let group = config.group_size.max(1) as usize;
        let cache = PrefixCacheCoordinator::new(&config, dp, group);
        Self {
            engine_id_: config.engine_id.clone(),
            routing_strategy: config.routing_strategy,
            config,
            seq_table: HashMap::new(),
            waiting: Vec::new(),
            waiting_migration: Vec::new(),
            running: (0..dp).map(|_| Vec::new()).collect(),
            prefilling: (0..dp).map(|_| Vec::new()).collect(),
            to_be_migrated: HashMap::new(),
            dummy_seq_ids: HashSet::new(),
            cache,
            current_step: 0,
            rr_cursor: 0,
            server_metric: ServerMetric::default(),
            runtime_metrics: RuntimeMetrics::default(),
            sequence_metrics: HashMap::new(),
        }
    }

    #[pyo3(signature = (seq_id, prompt_token_ids, sampling_params, affinity_key = 0, vision_slots = None))]
    pub(super) fn add_request(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        prompt_token_ids: Vec<i32>,
        sampling_params: Py<SamplingParams>,
        affinity_key: u64,
        vision_slots: Option<Vec<(String, i32, i32, i32, i32)>>,
    ) -> PyResult<(u64, i32)> {
        let prompt_len = prompt_token_ids.len() as i32;
        let sampling_params = sampling_params.borrow(py);
        let request = WireRequestIn {
            seq_id,
            prompt_token_ids,
            sampling_params: WireSamplingParams::from(&*sampling_params),
            affinity_key,
            vision_slots: vision_slots
                .unwrap_or_default()
                .into_iter()
                .map(
                    |(
                        encoder_engine_id,
                        slot_idx,
                        num_tokens,
                        hidden_size,
                        max_tokens_per_slot,
                    )| {
                        WireVisionSlot {
                            encoder_engine_id,
                            slot_idx,
                            num_tokens,
                            hidden_size,
                            max_tokens_per_slot,
                        }
                    },
                )
                .collect(),
        };
        let seq = request_to_sequence(request);
        self.add_sequence(seq);
        Ok((seq_id, prompt_len))
    }

    fn add_request_bytes(
        &mut self,
        _py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(u64, i32)>> {
        let data = bytes_arg(data)?;
        if let Ok(requests) = decode_binary::<Vec<WireRequestIn>>(&data, "add request") {
            let mut added = Vec::with_capacity(requests.len());
            for request in requests {
                let seq_id = request.seq_id;
                let prompt_len = request.prompt_token_ids.len() as i32;
                let seq = request_to_sequence(request);
                self.add_sequence(seq);
                added.push((seq_id, prompt_len));
            }
            return Ok(added);
        }

        let request: WireRequestMigrate = decode_binary(&data, "migration request")?;
        let seq_id = request.seq_id;
        let prompt_len = request.num_prompt_tokens;
        let seq = migrate_request_to_sequence(request);
        self.add_sequence(seq);
        Ok(vec![(seq_id, prompt_len)])
    }

    fn schedule(&mut self, py: Python<'_>) -> PyResult<ScheduleResult> {
        let schedule_begin_s = unix_time_s();
        let schedule_begin = Instant::now();
        self.current_step = self.current_step.saturating_add(1);
        let prefill = self.schedule_prefill(py)?;
        let has_prefill = prefill.iter().any(|seqs| !seqs.is_empty());
        let dp_seq_ids = if has_prefill {
            prefill
        } else {
            self.schedule_decode(py)?
        };
        let mut result = self.make_schedule_result(py, dp_seq_ids, has_prefill)?;
        result.schedule_begin_s = schedule_begin_s;
        result.schedule_end_s = unix_time_s();
        result.schedule_latency_ms = schedule_begin.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }

    fn schedule_with_metrics(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
    ) -> PyResult<ScheduleResult> {
        let mut result = self.schedule(py)?;
        result.scheduler_metric =
            Some(self.update_server_metric_and_log_impl(py, metric, &result)?);
        Ok(result)
    }

    fn record_schedule_metrics(
        &mut self,
        py: Python<'_>,
        result: &mut ScheduleResult,
    ) -> PyResult<()> {
        let mut metric = std::mem::take(&mut self.server_metric);
        result.scheduler_metric =
            Some(self.update_server_metric_and_log_impl(py, &mut metric, result)?);
        self.server_metric = metric;
        Ok(())
    }

    #[pyo3(signature = (dp_group_seq_ids, dp_group_token_ids, _update_metrics=None, dp_group_token_logprobs=None))]
    fn postprocess(
        &mut self,
        py: Python<'_>,
        dp_group_seq_ids: Vec<Vec<u64>>,
        dp_group_token_ids: Vec<Vec<Vec<i32>>>,
        _update_metrics: Option<bool>,
        dp_group_token_logprobs: Option<Vec<Vec<Vec<f32>>>>,
    ) {
        self.postprocess_impl(
            py,
            dp_group_seq_ids,
            dp_group_token_ids,
            dp_group_token_logprobs,
        );
    }

    #[pyo3(signature = (dp_group_seq_ids, runner_outs, _update_metrics=None))]
    pub(super) fn postprocess_runner_outs(
        &mut self,
        py: Python<'_>,
        dp_group_seq_ids: Vec<Vec<u64>>,
        runner_outs: Vec<RunnerOut>,
        _update_metrics: Option<bool>,
    ) -> PostprocessTiming {
        let begin_s = common::now_seconds();
        let (token_ids, token_logprobs) = Self::runner_outs_to_step_output(&runner_outs);
        self.postprocess_impl(py, dp_group_seq_ids, token_ids, token_logprobs);
        let end_s = common::now_seconds();
        PostprocessTiming {
            begin_s,
            end_s,
            latency_ms: (end_s - begin_s) * 1000.0,
        }
    }

    #[pyo3(signature = (result, runner_outs, _update_metrics=None))]
    fn postprocess_schedule_runner_outs(
        &mut self,
        py: Python<'_>,
        result: &mut ScheduleResult,
        runner_outs: Vec<RunnerOut>,
        _update_metrics: Option<bool>,
    ) {
        let dp_group_seq_ids = result.filtered_dp_group_seq_ids.clone();
        result.postprocess_timing =
            Some(self.postprocess_runner_outs(py, dp_group_seq_ids, runner_outs, _update_metrics));
    }

    fn serialize_run_batches(
        &self,
        py: Python<'_>,
        dp_group_seq_ids: Vec<Vec<u64>>,
        is_prefill: bool,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::with_capacity(dp_group_seq_ids.len() * tp_size.max(1));
        for seq_ids in dp_group_seq_ids {
            for _ in 0..tp_size.max(1) {
                let borrowed = seq_ids
                    .iter()
                    .filter_map(|seq_id| self.seq_table.get(seq_id))
                    .collect::<Vec<_>>();
                let bytes = sequence_refs_runner_in_bytes(borrowed, is_prefill)?;
                out.push(PyBytes::new(py, &bytes).into());
            }
        }
        Ok(out)
    }

    fn serialize_run_batches_for_result(
        &self,
        py: Python<'_>,
        result: &ScheduleResult,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.serialize_run_batches(
            py,
            result.dp_group_seq_ids.clone(),
            result.is_prefill,
            tp_size,
        )
    }

    fn serialize_migrate_batches(
        &self,
        py: Python<'_>,
        dp_group_seq_ids: Vec<Vec<u64>>,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::with_capacity(dp_group_seq_ids.len() * tp_size.max(1));
        for seq_ids in dp_group_seq_ids {
            for _ in 0..tp_size.max(1) {
                let borrowed = seq_ids
                    .iter()
                    .filter_map(|seq_id| self.seq_table.get(seq_id))
                    .collect::<Vec<_>>();
                let bytes = sequence_refs_migrate_batch_bytes(borrowed)?;
                out.push(PyBytes::new(py, &bytes).into());
            }
        }
        Ok(out)
    }

    fn serialize_migrate_batches_for_result(
        &self,
        py: Python<'_>,
        result: &ScheduleResult,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        self.serialize_migrate_batches(py, result.dp_group_seq_ids.clone(), tp_size)
    }

    fn complete_host_swap_outs(&mut self, tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>) {
        self.complete_host_swap_outs_impl(tasks)
    }

    fn complete_host_swap_ins(&mut self, tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>) {
        self.complete_host_swap_ins_impl(tasks)
    }

    fn collect_sequence_events(
        &mut self,
        py: Python<'_>,
        dp_seq_ids: Vec<Vec<u64>>,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
    ) -> PyResult<Vec<PyObject>> {
        Ok(self
            .collect_sequence_events_impl(py, dp_seq_ids, track_running, previous_running)?
            .0)
    }

    #[pyo3(signature = (
        result,
        track_running = false,
        previous_running = std::collections::HashSet::new(),
        runner_outs = None
    ))]
    fn record_complete_step(
        &mut self,
        py: Python<'_>,
        result: &ScheduleResult,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
        runner_outs: Option<Vec<RunnerOut>>,
    ) -> PyResult<StepResult> {
        self.record_step_metrics_and_report_api(
            py,
            result,
            track_running,
            previous_running,
            runner_outs,
        )
    }

    fn num_waiting_migration(&self) -> i32 {
        self.num_waiting_migration_api()
    }

    pub(super) fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.set_prefix_caching_enabled_api(enabled)
    }

    fn is_finished(&self) -> bool {
        self.is_finished_api()
    }

    fn has_runnable_work(&self) -> bool {
        self.has_runnable_work_api()
    }

    fn num_waiting(&self) -> i32 {
        self.num_waiting_api()
    }

    fn prefix_cached_tokens(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_api(seq_id)
    }

    fn clear_finished_metric_state(&mut self, seq_id: u64) {
        self.clear_finished_metric_state_api(seq_id)
    }

    fn abort(&mut self, seq_id: u64) -> bool {
        self.abort_api(seq_id)
    }

    fn abort_many(&mut self, seq_ids: Vec<u64>) -> Vec<u64> {
        self.abort_many_api(seq_ids)
    }

    fn free_to_be_migrated_ids(&mut self, py: Python<'_>, seq_ids: Vec<u64>) {
        self.free_to_be_migrated_ids_api(py, seq_ids)
    }

    fn set_sequence_metric(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        metric: Py<SequenceMetric>,
    ) -> bool {
        self.set_sequence_metric_api(py, seq_id, metric)
    }

    fn register_sequence_metric(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        num_prompt_tokens: i32,
    ) -> PyResult<bool> {
        self.register_sequence_metric_api(py, seq_id, num_prompt_tokens)
    }

    fn metric_snapshot(&self) -> SchedulerMetricSnapshot {
        self.metric_snapshot_api()
    }

    fn update_server_metric(
        &self,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> SchedulerMetricSnapshot {
        self.update_server_metric_api(metric, result)
    }

    fn update_server_metric_and_log(
        &self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> PyResult<SchedulerMetricSnapshot> {
        self.update_server_metric_and_log_api(py, metric, result)
    }

    #[pyo3(signature = (metric, result, dp_group_token_ids=None))]
    fn record_step_metric(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        dp_group_token_ids: Option<Vec<Vec<Vec<i32>>>>,
    ) -> StepMetricSnapshot {
        self.record_step_metric_api(py, metric, result, dp_group_token_ids)
    }

    pub(super) fn record_step_metric_runner_outs(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        runner_outs: Vec<RunnerOut>,
    ) -> StepMetricSnapshot {
        self.record_step_metric_runner_outs_api(py, metric, result, runner_outs)
    }

    #[pyo3(signature = (
        result,
        track_running = false,
        previous_running = std::collections::HashSet::new(),
        runner_outs = None
    ))]
    fn record_step_metrics_and_report(
        &mut self,
        py: Python<'_>,
        result: &ScheduleResult,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
        runner_outs: Option<Vec<RunnerOut>>,
    ) -> PyResult<StepResult> {
        self.record_step_metrics_and_report_api(
            py,
            result,
            track_running,
            previous_running,
            runner_outs,
        )
    }

    fn record_prompt_tokens(&mut self, num_prompt_tokens: i64) {
        self.record_prompt_tokens_api(num_prompt_tokens)
    }

    pub(super) fn record_sequence_completion(
        &mut self,
        py: Python<'_>,
        metric: &mut SequenceMetric,
    ) -> bool {
        self.record_sequence_completion_api(py, metric)
    }

    fn complete_sequence_metric(
        &mut self,
        py: Python<'_>,
        metric: &mut SequenceMetric,
    ) -> Option<String> {
        self.complete_sequence_metric_api(py, metric)
    }

    fn complete_sequence_by_id(&mut self, py: Python<'_>, seq_id: u64) -> Option<String> {
        self.complete_sequence_by_id_api(py, seq_id)
    }

    fn record_step_throughput(&mut self, prefill_tokens: i32, decode_tokens: i32, duration_s: f64) {
        self.record_step_throughput_api(prefill_tokens, decode_tokens, duration_s)
    }

    fn metric_report(&self, include_detailed: bool) -> String {
        self.metric_report_api(include_detailed)
    }

    fn metric_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.metric_summary_api(py)
    }

    fn metrics_prometheus(&self) -> String {
        self.metrics_prometheus_api()
    }

    fn final_metric_report(&self, py: Python<'_>) -> PyResult<String> {
        self.final_metric_report_api(py)
    }
}
