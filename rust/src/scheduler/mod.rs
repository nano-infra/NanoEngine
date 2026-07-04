use crate::common;
use crate::metrics::{sequence_metric_new, RuntimeMetrics, SequenceMetric, ServerMetric};
use crate::proto::wire::{
    add_request_to_sequence, bytes_arg, decode_binary, migration_request_to_sequence,
    sequence_migrate_batch_bytes, sequence_runner_in_bytes, sequence_to_migration_request,
    WireRequestIn, WireRequestMigrate, WireSamplingParams, WireVisionSlot,
};
use crate::proto::RunnerOut;
use crate::sequence::{set_sequence_block_size, SamplingParams, Sequence};
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use crate::table::block::{BlockPool, CompressedPool};
use crate::table::slot::SlotPool;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod group_manager;
mod lifecycle;
mod metrics;
mod resource;
mod router;
mod schedule;
mod session_state_cache;
mod types;

use session_state_cache::ParkedSession;
pub use types::{PostprocessTiming, RoutingStrategy, ScheduleResult, SchedulerConfig, StepResult};

#[pyclass(module = "dlengine._engine", unsendable)]
pub struct Scheduler {
    #[pyo3(get)]
    pub engine_id_: String,
    #[pyo3(get)]
    pub routing_strategy: i32,
    config: SchedulerConfig,
    waiting: Vec<Py<Sequence>>,
    waiting_migration: Vec<Py<Sequence>>,
    running: Vec<Vec<Py<Sequence>>>,
    prefilling: Vec<Vec<Py<Sequence>>>,
    to_be_migrated: HashMap<u64, (Py<Sequence>, usize)>,
    hbm_pools: Vec<BlockPool>,
    seq_assignment: HashMap<u64, (usize, usize)>,
    rr_cursor: usize,
    session_affinity: HashMap<u64, usize>,
    session_wait: HashMap<u64, i32>,
    parked_sessions: HashMap<u64, ParkedSession>,
    parked_lru: Vec<u64>,
    state_slots: SlotPool,
    hisparse_slots: SlotPool,
    compressed_pools: HashMap<i32, CompressedPool>,
    prefix_caching_allowed: bool,
    prefix_cached_tokens_by_seq: HashMap<u64, i32>,
    prefix_counted_seq_ids: HashSet<u64>,
    server_metric: ServerMetric,
    runtime_metrics: RuntimeMetrics,
    sequence_metrics: HashMap<u64, Py<SequenceMetric>>,
}

#[pymethods]
impl Scheduler {
    #[new]
    fn new(config: SchedulerConfig) -> Self {
        set_sequence_block_size(config.kvcache_block_size);
        let dp = config.attention_dp.max(1) as usize;
        let group = config.group_size.max(1) as usize;
        let num_blocks = config.num_kvcache_blocks.max(0);
        let prefix_caching_allowed =
            config.enable_prefix_cache && config.cache_plan.flags & (1 << 2) == 0;
        let mut hbm_pools = (0..(dp * group))
            .map(|_| BlockPool::new(num_blocks, config.kvcache_block_size))
            .collect::<Vec<_>>();
        if !prefix_caching_allowed {
            for pool in &mut hbm_pools {
                pool.set_prefix_caching_enabled(false);
            }
        }
        let mut compressed_pools = HashMap::new();
        if config.cache_plan.flags & ((1 << 3) | (1 << 4)) != 0 {
            for spec in [
                (
                    config.cache_plan.hca.compression_ratio,
                    config.cache_plan.hca.num_pages,
                    config.cache_plan.hca.page_size,
                    config.cache_plan.hca.max_blocks_per_seq,
                ),
                (
                    config.cache_plan.csa.compression_ratio,
                    config.cache_plan.csa.num_pages,
                    config.cache_plan.csa.page_size,
                    config.cache_plan.csa.max_blocks_per_seq,
                ),
            ] {
                let (ratio, num_pages, page_size, max_blocks_per_seq) = spec;
                if ratio > 0 && num_pages > 0 && page_size > 0 && max_blocks_per_seq > 0 {
                    compressed_pools.insert(
                        ratio,
                        CompressedPool {
                            ratio,
                            page_size,
                            max_blocks_per_seq,
                            free_pages: (0..num_pages).rev().collect(),
                            seq_pages: HashMap::new(),
                        },
                    );
                }
            }
        }
        let state_slots = if config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) != 0 {
            let configured = config.cache_plan.gdn.state_slots;
            configured
                .max(config.max_num_seqs + config.gdn_state_cache_slots)
                .max(config.max_num_seqs)
        } else {
            0
        };
        let hisparse_slots = if config.cache_plan.flags & (1 << 6) != 0 {
            config
                .cache_plan
                .hisparse
                .max_num_seqs
                .max(config.max_num_seqs)
        } else {
            0
        };
        Self {
            engine_id_: config.engine_id.clone(),
            routing_strategy: config.routing_strategy,
            config,
            waiting: Vec::new(),
            waiting_migration: Vec::new(),
            running: (0..dp).map(|_| Vec::new()).collect(),
            prefilling: (0..dp).map(|_| Vec::new()).collect(),
            to_be_migrated: HashMap::new(),
            hbm_pools,
            seq_assignment: HashMap::new(),
            rr_cursor: 0,
            session_affinity: HashMap::new(),
            session_wait: HashMap::new(),
            parked_sessions: HashMap::new(),
            parked_lru: Vec::new(),
            state_slots: SlotPool::new(state_slots),
            hisparse_slots: SlotPool::new(hisparse_slots),
            compressed_pools,
            prefix_caching_allowed,
            prefix_cached_tokens_by_seq: HashMap::new(),
            prefix_counted_seq_ids: HashSet::new(),
            server_metric: ServerMetric::default(),
            runtime_metrics: RuntimeMetrics::default(),
            sequence_metrics: HashMap::new(),
        }
    }

    fn add(&mut self, seq: Py<Sequence>) {
        if self.config.mode == "decode" {
            self.waiting_migration.push(seq);
        } else {
            self.waiting.push(seq);
        }
    }

    #[pyo3(signature = (seq_id, prompt_token_ids, sampling_params, affinity_key = 0, vision_slots = None))]
    fn add_request(
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
        let seq = add_request_to_sequence(py, request)?;
        self.add(seq);
        Ok((seq_id, prompt_len))
    }

    fn add_request_bytes(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(u64, i32)>> {
        let data = bytes_arg(data)?;
        if let Ok(requests) = decode_binary::<Vec<WireRequestIn>>(&data, "add request") {
            let mut added = Vec::with_capacity(requests.len());
            for request in requests {
                let seq_id = request.seq_id;
                let prompt_len = request.prompt_token_ids.len() as i32;
                let seq = add_request_to_sequence(py, request)?;
                self.add(seq);
                added.push((seq_id, prompt_len));
            }
            return Ok(added);
        }

        let request: WireRequestMigrate = decode_binary(&data, "migration request")?;
        let seq_id = request.seq_id;
        let prompt_len = request.num_prompt_tokens;
        let seq = migration_request_to_sequence(py, request)?;
        self.add(seq);
        Ok(vec![(seq_id, prompt_len)])
    }

    fn set_sequence_metric(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        metric: Py<SequenceMetric>,
    ) -> bool {
        self.set_sequence_metric_impl(py, seq_id, metric)
    }

    fn register_sequence_metric(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        num_prompt_tokens: i32,
    ) -> PyResult<bool> {
        let metric = Py::new(py, sequence_metric_new(seq_id, num_prompt_tokens))?;
        self.sequence_metrics.insert(seq_id, metric.clone_ref(py));
        self.record_prompt_tokens(num_prompt_tokens as i64);
        Ok(self.set_sequence_metric_impl(py, seq_id, metric))
    }

    fn schedule(&mut self, py: Python<'_>) -> PyResult<ScheduleResult> {
        let schedule_begin_s = unix_time_s();
        let schedule_begin = Instant::now();
        let prefill = self.schedule_prefill(py)?;
        let has_prefill = prefill.iter().any(|seqs| !seqs.is_empty());
        let dp_seqs = if has_prefill {
            prefill
        } else {
            self.schedule_decode(py)?
        };
        let mut result = self.make_schedule_result(py, dp_seqs, has_prefill)?;
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

    fn num_waiting_migration(&self) -> i32 {
        self.waiting_migration.len() as i32
    }

    fn set_session_cache_slots(&mut self, capacity: i32) {
        self.config.gdn_state_cache_slots = capacity.max(0);
    }

    fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        let enabled = enabled && self.prefix_caching_allowed;
        for pool in &mut self.hbm_pools {
            pool.set_prefix_caching_enabled(enabled);
        }
    }

    fn preempt(&mut self, py: Python<'_>, dp_idx: usize, seq: Py<Sequence>) -> PyResult<()> {
        self.preempt_impl(py, dp_idx, seq)
    }

    fn num_parked_sessions(&self) -> i32 {
        self.parked_sessions.len() as i32
    }

    fn parked_session_keys(&self) -> Vec<u64> {
        self.parked_lru.clone()
    }

    fn clear_session_cache(&mut self) {
        let keys = self.parked_lru.clone();
        for key in keys {
            self.evict_parked_by_key(key);
        }
    }

    #[pyo3(signature = (dp_group_seqs, dp_group_token_ids, _update_metrics=None, dp_group_token_logprobs=None))]
    fn postprocess(
        &mut self,
        py: Python<'_>,
        dp_group_seqs: Vec<Vec<Py<Sequence>>>,
        dp_group_token_ids: Vec<Vec<Vec<i32>>>,
        _update_metrics: Option<bool>,
        dp_group_token_logprobs: Option<Vec<Vec<Vec<f32>>>>,
    ) {
        self.postprocess_impl(
            py,
            dp_group_seqs,
            dp_group_token_ids,
            dp_group_token_logprobs,
        );
    }

    #[pyo3(signature = (dp_group_seqs, runner_outs, _update_metrics=None))]
    fn postprocess_runner_outs(
        &mut self,
        py: Python<'_>,
        dp_group_seqs: Vec<Vec<Py<Sequence>>>,
        runner_outs: Vec<RunnerOut>,
        _update_metrics: Option<bool>,
    ) -> PostprocessTiming {
        let begin_s = common::now_seconds();
        let (token_ids, token_logprobs) = Self::runner_outs_to_step_output(&runner_outs);
        self.postprocess_impl(py, dp_group_seqs, token_ids, token_logprobs);
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
        let dp_group_seqs = result
            .filtered_dp_group_seqs
            .iter()
            .map(|seqs| seqs.iter().map(|seq| seq.clone_ref(py)).collect())
            .collect();
        result.postprocess_timing =
            Some(self.postprocess_runner_outs(py, dp_group_seqs, runner_outs, _update_metrics));
    }

    fn is_finished(&self) -> bool {
        self.waiting.is_empty()
            && self.waiting_migration.is_empty()
            && self.to_be_migrated.is_empty()
            && self.running.iter().all(|seqs| seqs.is_empty())
            && self.prefilling.iter().all(|seqs| seqs.is_empty())
    }

    fn num_waiting(&self) -> i32 {
        self.waiting.len() as i32
    }

    fn metric_snapshot(&self) -> SchedulerMetricSnapshot {
        self.metric_snapshot_impl()
    }

    fn update_server_metric(
        &self,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> SchedulerMetricSnapshot {
        self.update_server_metric_impl(metric, result)
    }

    fn update_server_metric_and_log(
        &self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> PyResult<SchedulerMetricSnapshot> {
        self.update_server_metric_and_log_impl(py, metric, result)
    }

    #[pyo3(signature = (metric, result, dp_group_token_ids=None))]
    fn record_step_metric(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        dp_group_token_ids: Option<Vec<Vec<Vec<i32>>>>,
    ) -> StepMetricSnapshot {
        self.record_step_metric_impl(py, metric, result, dp_group_token_ids)
    }

    fn record_step_metric_runner_outs(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        runner_outs: Vec<RunnerOut>,
    ) -> StepMetricSnapshot {
        let (token_ids, _) = Self::runner_outs_to_step_output(&runner_outs);
        self.record_step_metric_impl(py, metric, result, Some(token_ids))
    }

    #[pyo3(signature = (
        runtime,
        metric,
        scheduler_metric,
        result,
        postprocess_ms,
        runner_outs = None
    ))]
    fn record_step_metrics_and_report(
        &mut self,
        py: Python<'_>,
        runtime: &mut RuntimeMetrics,
        metric: &mut ServerMetric,
        scheduler_metric: &SchedulerMetricSnapshot,
        result: &ScheduleResult,
        postprocess_ms: f64,
        runner_outs: Option<Vec<RunnerOut>>,
    ) -> (StepMetricSnapshot, Option<String>) {
        self.record_step_metrics_and_report_impl(
            py,
            runtime,
            metric,
            scheduler_metric,
            result,
            postprocess_ms,
            common::now_seconds(),
            runner_outs,
        )
    }

    fn prefix_cached_tokens(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_impl(seq_id)
    }

    fn clear_finished_metric_state(&mut self, seq_id: u64) {
        self.clear_finished_metric_state_impl(seq_id)
    }

    fn abort(&mut self, seq_id: u64) -> bool {
        self.abort_impl(seq_id)
    }

    fn abort_many(&mut self, seq_ids: Vec<u64>) -> Vec<u64> {
        self.abort_many_impl(seq_ids)
    }

    fn free_to_be_migrated(&mut self, py: Python<'_>, seqs: &Bound<'_, PyAny>) -> PyResult<()> {
        self.free_to_be_migrated_impl(py, seqs)
    }

    fn free_to_be_migrated_ids(&mut self, py: Python<'_>, seq_ids: Vec<u64>) {
        self.free_to_be_migrated_ids_impl(py, seq_ids)
    }

    fn serialize_run_batches(
        &self,
        py: Python<'_>,
        dp_group_seqs: Vec<Vec<Py<Sequence>>>,
        is_prefill: bool,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::with_capacity(dp_group_seqs.len() * tp_size.max(1));
        for seqs in dp_group_seqs {
            for _ in 0..tp_size.max(1) {
                let bytes = sequence_runner_in_bytes(
                    py,
                    seqs.iter().map(|seq| seq.clone_ref(py)).collect(),
                    is_prefill,
                )?;
                out.push(PyBytes::new(py, &bytes).into());
            }
        }
        Ok(out)
    }

    fn serialize_migrate_batches(
        &self,
        py: Python<'_>,
        dp_group_seqs: Vec<Vec<Py<Sequence>>>,
        tp_size: usize,
    ) -> PyResult<Vec<PyObject>> {
        let mut out = Vec::with_capacity(dp_group_seqs.len() * tp_size.max(1));
        for seqs in dp_group_seqs {
            for _ in 0..tp_size.max(1) {
                let bytes = sequence_migrate_batch_bytes(
                    py,
                    seqs.iter().map(|s| s.clone_ref(py)).collect(),
                )?;
                out.push(PyBytes::new(py, &bytes).into());
            }
        }
        Ok(out)
    }

    fn collect_sequence_events(
        &mut self,
        py: Python<'_>,
        dp_seqs: Vec<Vec<Py<Sequence>>>,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
    ) -> PyResult<Vec<PyObject>> {
        Ok(self
            .collect_sequence_events_impl(py, dp_seqs, track_running, previous_running)?
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
            result
                .dp_seqs
                .iter()
                .map(|seqs| seqs.iter().map(|seq| seq.clone_ref(py)).collect())
                .collect(),
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

    fn record_prompt_tokens(&mut self, num_prompt_tokens: i64) {
        self.server_metric.add_tokens(num_prompt_tokens, 0);
    }

    fn record_sequence_completion(&mut self, _py: Python<'_>, metric: &mut SequenceMetric) -> bool {
        let mut runtime = std::mem::take(&mut self.runtime_metrics);
        let mut server_metric = std::mem::take(&mut self.server_metric);
        let should_log = runtime.record_sequence_completion(metric, &mut server_metric);
        self.runtime_metrics = runtime;
        self.server_metric = server_metric;
        should_log
    }

    fn complete_sequence_metric(
        &mut self,
        _py: Python<'_>,
        metric: &mut SequenceMetric,
    ) -> Option<String> {
        self.record_sequence_completion(_py, metric)
            .then(|| metric.metric_report())
    }

    fn complete_sequence_by_id(&mut self, py: Python<'_>, seq_id: u64) -> Option<String> {
        let metric = self
            .sequence_metrics
            .get(&seq_id)
            .map(|m| m.clone_ref(py))?;
        self.complete_sequence_metric_py(py, &metric)
    }

    fn record_step_throughput(&mut self, prefill_tokens: i32, decode_tokens: i32, duration_s: f64) {
        if prefill_tokens > 0 {
            self.server_metric
                .record_prefill_throughput(prefill_tokens as i64, duration_s);
        }
        if decode_tokens > 0 {
            self.server_metric
                .record_decode_throughput(decode_tokens as i64, duration_s);
        }
    }

    fn metric_report(&self, include_detailed: bool) -> String {
        self.server_metric.get_metric_report(include_detailed)
    }

    fn metric_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.server_metric.get_summary(py)
    }

    fn metrics_prometheus(&self) -> String {
        self.runtime_metrics.to_prometheus(&self.server_metric)
    }

    fn final_metric_report(&self, py: Python<'_>) -> PyResult<String> {
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

fn average_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile_f64(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted.get(idx).copied()
}

fn unix_time_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

impl Scheduler {
    fn complete_sequence_metric_py(
        &mut self,
        py: Python<'_>,
        metric: &Py<SequenceMetric>,
    ) -> Option<String> {
        let mut metric = metric.borrow_mut(py);
        self.record_sequence_completion(py, &mut metric)
            .then(|| metric.metric_report())
    }

    fn collect_sequence_events_impl(
        &mut self,
        py: Python<'_>,
        dp_seqs: Vec<Vec<Py<Sequence>>>,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
    ) -> PyResult<(Vec<PyObject>, Vec<u64>)> {
        let mut out = Vec::new();
        let mut completed = Vec::new();
        for seqs in dp_seqs {
            for seq in seqs {
                let (
                    seq_id,
                    last_token,
                    num_tokens,
                    is_finished,
                    is_to_be_migrated,
                    source_engine_id,
                    vision_slots,
                ) = {
                    let s = seq.borrow(py);
                    (
                        s.seq_id,
                        s.last_token,
                        s.num_tokens,
                        s.status == 2,
                        s.status == 3,
                        s.migrate_engine_id.clone(),
                        s.vision_slots
                            .iter()
                            .map(|slot| {
                                (
                                    slot.encoder_engine_id.clone(),
                                    slot.slot_idx,
                                    slot.num_tokens,
                                    slot.hidden_size,
                                    slot.max_tokens_per_slot,
                                )
                            })
                            .collect::<Vec<_>>(),
                    )
                };
                if seq_id < 8 {
                    continue;
                }
                let event = PyDict::new(py);
                event.set_item("seq_id", seq_id)?;
                event.set_item("last_token", last_token)?;
                event.set_item("num_tokens", num_tokens)?;
                event.set_item("is_finished", is_finished)?;
                event.set_item("is_to_be_migrated", is_to_be_migrated)?;
                event.set_item(
                    "is_new_running",
                    track_running && !previous_running.contains(&seq_id),
                )?;
                event.set_item("source_engine_id", source_engine_id)?;
                let vision_list = PyList::empty(py);
                for (encoder_engine_id, slot_idx, num_tokens, hidden_size, max_tokens_per_slot) in
                    &vision_slots
                {
                    let d = PyDict::new(py);
                    d.set_item("encoder_engine_id", encoder_engine_id)?;
                    d.set_item("slot_idx", slot_idx)?;
                    d.set_item("num_tokens", num_tokens)?;
                    d.set_item("hidden_size", hidden_size)?;
                    d.set_item("max_tokens_per_slot", max_tokens_per_slot)?;
                    vision_list.append(d)?;
                }
                event.set_item("vision_slots", vision_list)?;
                if is_to_be_migrated {
                    let migration = sequence_to_migration_request(py, &seq);
                    let bytes = crate::proto::wire::encode_binary(&migration, "migration request")?;
                    event.set_item("migration_payload", PyBytes::new(py, &bytes))?;
                }
                if !vision_slots.is_empty() {
                    seq.borrow_mut(py).vision_slots.clear();
                }
                if is_finished || is_to_be_migrated {
                    completed.push(seq_id);
                }
                out.push(event.unbind().into());
            }
        }
        Ok((out, completed))
    }

    fn dp(&self) -> usize {
        self.config.attention_dp.max(1) as usize
    }

    fn group(&self) -> usize {
        self.config.group_size.max(1) as usize
    }

    fn flat_idx(&self, dp_idx: usize, group_id: usize) -> usize {
        dp_idx * self.group() + group_id
    }

    fn runner_outs_to_step_output(
        runner_outs: &[RunnerOut],
    ) -> (Vec<Vec<Vec<i32>>>, Option<Vec<Vec<Vec<f32>>>>) {
        let mut token_ids = Vec::with_capacity(runner_outs.len());
        let mut logprobs = Vec::with_capacity(runner_outs.len());
        let mut any_logprobs = false;
        for out in runner_outs {
            token_ids.push(
                out.token_ids
                    .iter()
                    .map(|seq_tokens| seq_tokens.iter().copied().map(|t| t as i32).collect())
                    .collect(),
            );
            if let Some(values) = &out.logprobs {
                logprobs.push(values.clone());
                any_logprobs = true;
            } else {
                logprobs.push(Vec::new());
            }
        }
        (token_ids, any_logprobs.then_some(logprobs))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RoutingStrategy>()?;
    m.add_class::<SchedulerConfig>()?;
    m.add_class::<ScheduleResult>()?;
    m.add_class::<StepResult>()?;
    m.add_class::<PostprocessTiming>()?;
    m.add_class::<Scheduler>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CachePlan;
    use crate::metrics::ServerMetric;

    fn make_scheduler() -> Scheduler {
        make_scheduler_with_flags(1)
    }

    fn make_scheduler_with_flags(flags: u32) -> Scheduler {
        make_scheduler_with_flags_and_prefix_cache(flags, true)
    }

    fn make_scheduler_with_flags_and_prefix_cache(
        flags: u32,
        enable_prefix_cache: bool,
    ) -> Scheduler {
        Scheduler::new(SchedulerConfig {
            engine_id: "engine".to_string(),
            num_speculative_tokens: 0,
            max_num_seqs: 8,
            max_num_batched_tokens: 64,
            max_model_len: 128,
            eos_ids: Vec::new(),
            attention_dp: 1,
            group_size: 1,
            num_kvcache_blocks: 16,
            kvcache_block_size: 4,
            mode: "hybrid".to_string(),
            routing_strategy: RoutingStrategy::RoundRobin,
            gdn_state_cache_slots: 0,
            enable_prefix_cache,
            cache_plan: CachePlan::new(flags),
        })
    }

    fn add_tokens(
        py: Python<'_>,
        scheduler: &mut Scheduler,
        seq_id: u64,
        tokens: Vec<i32>,
    ) -> PyResult<()> {
        let sampling = Py::new(py, SamplingParams::new(1.0, 16, false, false))?;
        scheduler.add_request(py, seq_id, tokens, sampling, 0, None)?;
        Ok(())
    }

    fn run_one_prefill(py: Python<'_>, scheduler: &mut Scheduler) -> PyResult<Vec<Py<Sequence>>> {
        let scheduled = scheduler.schedule_prefill(py)?;
        assert_eq!(scheduled.len(), 1);
        assert_eq!(scheduled[0].len(), 1);
        let seqs = scheduled[0]
            .iter()
            .map(|seq| seq.clone_ref(py))
            .collect::<Vec<_>>();
        let postprocess_seqs = seqs.iter().map(|seq| seq.clone_ref(py)).collect();
        scheduler.postprocess_impl(py, vec![postprocess_seqs], vec![vec![Vec::new()]], None);
        Ok(seqs)
    }

    fn server_metric() -> ServerMetric {
        ServerMetric {
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
            token_usage_by_dp: std::collections::HashMap::new(),
            group_send_request_counts: std::collections::HashMap::new(),
            group_recv_request_counts: std::collections::HashMap::new(),
            start_time: 0.0,
        }
    }

    #[test]
    fn scheduler_reuses_hbm_prefix_after_release() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler();
            add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
            let first = run_one_prefill(py, &mut scheduler).unwrap();
            let first_table = first[0].borrow(py).active_block_table.clone();
            scheduler.release_seq(first[0].borrow(py).seq_id);

            add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
            let second = run_one_prefill(py, &mut scheduler).unwrap();
            assert_eq!(second[0].borrow(py).active_block_table, first_table);
            assert_eq!(scheduler.prefix_cached_tokens_impl(101), 7);
        });
    }

    #[test]
    fn scheduler_prefix_cache_can_be_disabled() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler();
            add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
            let first = run_one_prefill(py, &mut scheduler).unwrap();
            scheduler.release_seq(first[0].borrow(py).seq_id);

            scheduler.set_prefix_caching_enabled(false);
            add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
            run_one_prefill(py, &mut scheduler).unwrap();
            assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
        });
    }

    #[test]
    fn step_metric_reports_prefix_cache_hits_per_dp() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler();
            add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
            let first = run_one_prefill(py, &mut scheduler).unwrap();
            scheduler.release_seq(first[0].borrow(py).seq_id);

            add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
            let scheduled = scheduler.schedule_prefill(py).unwrap();
            assert_eq!(scheduled[0].len(), 1);
            let result = ScheduleResult {
                dp_seqs: vec![vec![scheduled[0][0].clone_ref(py)]],
                is_prefill: true,
                ..ScheduleResult::default()
            };
            let mut metric = server_metric();
            let snapshot = scheduler.record_step_metric_impl(py, &mut metric, &result, None);
            assert_eq!(snapshot.prefill_tokens_per_dp, vec![1]);
            assert_eq!(snapshot.prefix_cached_tokens_per_dp, vec![7]);
            assert_eq!(snapshot.prefix_prompt_tokens_per_dp, vec![8]);
        });
    }

    #[test]
    fn gdn_scheduler_never_enables_prefix_cache() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler_with_flags(1 << 2);
            scheduler.set_prefix_caching_enabled(true);
            add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
            let first = run_one_prefill(py, &mut scheduler).unwrap();
            scheduler.release_seq(first[0].borrow(py).seq_id);

            add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
            run_one_prefill(py, &mut scheduler).unwrap();
            assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
        });
    }

    #[test]
    fn scheduler_config_can_disable_prefix_cache() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler_with_flags_and_prefix_cache(1, false);
            scheduler.set_prefix_caching_enabled(true);
            add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
            let first = run_one_prefill(py, &mut scheduler).unwrap();
            scheduler.release_seq(first[0].borrow(py).seq_id);

            add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
            run_one_prefill(py, &mut scheduler).unwrap();
            assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
        });
    }

    #[test]
    fn prefill_postprocess_preserves_existing_running_sequences() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler();
            add_tokens(py, &mut scheduler, 100, vec![1, 2, 3]).unwrap();
            run_one_prefill(py, &mut scheduler).unwrap();
            assert_eq!(scheduler.running[0].len(), 1);
            assert_eq!(scheduler.running[0][0].borrow(py).seq_id, 100);

            add_tokens(py, &mut scheduler, 101, vec![4, 5, 6]).unwrap();
            let scheduled = scheduler.schedule_prefill(py).unwrap();
            assert_eq!(scheduled[0].len(), 1);
            assert_eq!(scheduled[0][0].borrow(py).seq_id, 101);
            scheduler.postprocess_impl(
                py,
                vec![vec![scheduled[0][0].clone_ref(py)]],
                vec![vec![Vec::new()]],
                None,
            );

            let running_ids = scheduler.running[0]
                .iter()
                .map(|seq| seq.borrow(py).seq_id)
                .collect::<Vec<_>>();
            assert_eq!(running_ids, vec![100, 101]);
        });
    }

    #[test]
    fn runner_out_postprocess_and_metrics_entrypoints_work() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let mut scheduler = make_scheduler();
            add_tokens(py, &mut scheduler, 100, vec![1, 2, 3]).unwrap();
            let scheduled = scheduler.schedule_prefill(py).unwrap();
            let seq = scheduled[0][0].clone_ref(py);
            let runner_out = RunnerOut {
                token_ids: vec![vec![4]],
                logprobs: Some(vec![vec![0.25]]),
                server_handler_ns: 0,
            };
            scheduler.postprocess_runner_outs(
                py,
                vec![vec![seq.clone_ref(py)]],
                vec![runner_out.clone()],
                None,
            );
            assert_eq!(seq.borrow(py).last_token, 4);
            assert_eq!(seq.borrow(py).completion_logprobs, vec![0.25]);

            let result = ScheduleResult {
                dp_seqs: vec![vec![seq]],
                is_prefill: false,
                ..ScheduleResult::default()
            };
            let mut metric = server_metric();
            let snapshot = scheduler.record_step_metric_runner_outs(
                py,
                &mut metric,
                &result,
                vec![runner_out],
            );
            assert_eq!(snapshot.decode_tokens, 1);
            assert_eq!(snapshot.decode_tokens_per_dp, vec![1]);
        });
    }
}
