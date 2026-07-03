use crate::metrics::{SequenceMetric, ServerMetric};
use crate::proto::wire::{
    add_request_to_sequence, bytes_arg, decode_binary, migration_request_to_sequence,
    sequence_migrate_batch_bytes, sequence_runner_in_bytes, sequence_to_migration_request,
    WireRequestIn, WireRequestMigrate, WireSamplingParams, WireVisionSlot,
};
use crate::sequence::{set_sequence_block_size, SamplingParams, Sequence};
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::{HashMap, HashSet};

mod group_manager;
mod lifecycle;
mod metrics;
mod resource;
mod router;
mod schedule;
mod session_state_cache;
mod types;

use resource::{CompressedPool, GroupResource};
use session_state_cache::ParkedSession;
pub use types::{RoutingStrategy, ScheduleResult, SchedulerConfig};

#[pyclass(module = "dlengine._dlengine_rust", unsendable)]
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
    group_resources: Vec<GroupResource>,
    seq_assignment: HashMap<u64, (usize, usize)>,
    rr_cursor: usize,
    session_affinity: HashMap<u64, usize>,
    session_wait: HashMap<u64, i32>,
    parked_sessions: HashMap<u64, ParkedSession>,
    parked_lru: Vec<u64>,
    state_free: Vec<i32>,
    seq_state_slots: HashMap<u64, i32>,
    hisparse_free: Vec<i32>,
    seq_hisparse_slots: HashMap<u64, i32>,
    compressed_pools: HashMap<i32, CompressedPool>,
    prefix_cached_tokens_by_seq: HashMap<u64, i32>,
    prefix_counted_seq_ids: HashSet<u64>,
}

#[pymethods]
impl Scheduler {
    #[new]
    fn new(config: SchedulerConfig) -> Self {
        set_sequence_block_size(config.kvcache_block_size);
        let dp = config.attention_dp.max(1) as usize;
        let group = config.group_size.max(1) as usize;
        let num_blocks = config.num_kvcache_blocks.max(0);
        let group_resources = (0..(dp * group))
            .map(|_| GroupResource {
                free_blocks: (0..num_blocks).rev().collect(),
                seq_blocks: HashMap::new(),
            })
            .collect();
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
            group_resources,
            seq_assignment: HashMap::new(),
            rr_cursor: 0,
            session_affinity: HashMap::new(),
            session_wait: HashMap::new(),
            parked_sessions: HashMap::new(),
            parked_lru: Vec::new(),
            state_free: (0..state_slots.max(0)).rev().collect(),
            seq_state_slots: HashMap::new(),
            hisparse_free: (0..hisparse_slots.max(0)).rev().collect(),
            seq_hisparse_slots: HashMap::new(),
            compressed_pools,
            prefix_cached_tokens_by_seq: HashMap::new(),
            prefix_counted_seq_ids: HashSet::new(),
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

    fn schedule(&mut self, py: Python<'_>) -> PyResult<ScheduleResult> {
        let prefill = self.schedule_prefill(py)?;
        let has_prefill = prefill.iter().any(|seqs| !seqs.is_empty());
        let dp_seqs = if has_prefill {
            prefill
        } else {
            self.schedule_decode(py)?
        };
        self.make_schedule_result(py, dp_seqs, has_prefill)
    }

    fn num_waiting_migration(&self) -> i32 {
        self.waiting_migration.len() as i32
    }

    fn set_session_cache_slots(&mut self, capacity: i32) {
        self.config.gdn_state_cache_slots = capacity.max(0);
    }

    fn set_prefix_caching_enabled(&mut self, _enabled: bool) {}

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

    fn prefix_cached_tokens(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_impl(seq_id)
    }

    fn clear_finished_metric_state(&mut self, seq_id: u64) {
        self.clear_finished_metric_state_impl(seq_id)
    }

    fn abort(&mut self, seq_id: u64) -> bool {
        self.abort_impl(seq_id)
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
        let mut out = Vec::new();
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
                out.push(event.unbind().into());
            }
        }
        Ok(out)
    }
}

impl Scheduler {
    fn dp(&self) -> usize {
        self.config.attention_dp.max(1) as usize
    }

    fn group(&self) -> usize {
        self.config.group_size.max(1) as usize
    }

    fn flat_idx(&self, dp_idx: usize, group_id: usize) -> usize {
        dp_idx * self.group() + group_id
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RoutingStrategy>()?;
    m.add_class::<SchedulerConfig>()?;
    m.add_class::<ScheduleResult>()?;
    m.add_class::<Scheduler>()?;
    Ok(())
}
