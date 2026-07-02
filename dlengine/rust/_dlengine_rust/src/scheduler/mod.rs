use crate::cache_plan::CachePlan;
use crate::metrics::ServerMetric;
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::{HashMap, HashSet};

mod group_manager;
mod resource;
mod router;
mod session_state_cache;

use resource::{CompressedPool, GroupResource};
use session_state_cache::ParkedSession;

#[pyclass(module = "dlengine._dlengine_rust")]
pub struct RoutingStrategy;

#[pymethods]
impl RoutingStrategy {
    #[classattr]
    const RoundRobin: i32 = 0;
    #[classattr]
    const LeastBatch: i32 = 1;
    #[classattr]
    const LeastCache: i32 = 2;
    #[classattr]
    const SessionPrefix: i32 = 3;

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
    pub dp_seqs: Vec<Vec<Py<PyAny>>>,
    #[pyo3(get, set)]
    pub dp_group_seqs: Vec<Vec<Py<PyAny>>>,
    #[pyo3(get, set)]
    pub filtered_dp_group_seqs: Vec<Vec<Py<PyAny>>>,
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

#[pyclass(module = "dlengine._dlengine_rust", unsendable)]
pub struct Scheduler {
    #[pyo3(get)]
    pub engine_id_: String,
    #[pyo3(get)]
    pub routing_strategy: i32,
    config: SchedulerConfig,
    waiting: Vec<Py<PyAny>>,
    waiting_migration: Vec<Py<PyAny>>,
    running: Vec<Vec<Py<PyAny>>>,
    prefilling: Vec<Vec<Py<PyAny>>>,
    to_be_migrated: HashMap<u64, (Py<PyAny>, usize)>,
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

    fn add(&mut self, seq: Py<PyAny>) {
        if self.config.mode == "decode" {
            self.waiting_migration.push(seq);
        } else {
            self.waiting.push(seq);
        }
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

    fn preempt(&mut self, py: Python<'_>, dp_idx: usize, seq: Py<PyAny>) -> PyResult<()> {
        let obj = seq.bind(py);
        let seq_id = obj.getattr("seq_id")?.extract::<u64>()?;
        obj.setattr("status", 0)?;
        let total_tokens = obj.getattr("token_ids")?.extract::<Vec<i32>>()?.len() as i32;
        obj.setattr("num_tokens", total_tokens)?;
        obj.setattr("num_checkpointed_tokens", total_tokens)?;
        self.release_seq(seq_id);
        self.running[dp_idx].retain(|s| {
            s.bind(py)
                .getattr("seq_id")
                .and_then(|v| v.extract::<u64>())
                .map(|id| id != seq_id)
                .unwrap_or(true)
        });
        self.waiting.insert(0, seq);
        Ok(())
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
        dp_group_seqs: Vec<Vec<Py<PyAny>>>,
        dp_group_token_ids: Vec<Vec<Vec<i32>>>,
        _update_metrics: Option<bool>,
        dp_group_token_logprobs: Option<Vec<Vec<Vec<f32>>>>,
    ) {
        let eos_ids = self.config.eos_ids.clone();
        let mut next_running: Vec<Vec<Py<PyAny>>> = (0..self.dp()).map(|_| Vec::new()).collect();

        for (group_idx, seqs) in dp_group_seqs.iter().enumerate() {
            let dp_idx = group_idx / self.group();
            let group_id = group_idx % self.group();
            let Some(group_tokens) = dp_group_token_ids.get(group_idx) else {
                continue;
            };
            for (seq_idx, seq) in seqs.iter().enumerate() {
                let Some(tokens) = group_tokens.get(seq_idx) else {
                    next_running[dp_idx].push(seq.clone_ref(py));
                    continue;
                };

                let obj = seq.bind(py);
                let seq_id = obj
                    .getattr("seq_id")
                    .and_then(|v| v.extract::<u64>())
                    .unwrap_or(0);
                let prefill_target = obj
                    .getattr("num_prompt_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0)
                    .max(
                        obj.getattr("num_checkpointed_tokens")
                            .and_then(|v| v.extract::<i32>())
                            .unwrap_or(0),
                    );
                let cur_tokens = obj
                    .getattr("num_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0);
                if cur_tokens < prefill_target {
                    let _ = obj.setattr("num_cached_tokens", cur_tokens);
                    let _ = obj.setattr("status", 4);
                    if let Ok(metric) = obj.getattr("metric") {
                        if !metric.is_none() {
                            let _ = metric.call_method0("record_prefill_chunk");
                        }
                    }
                    self.prefilling[dp_idx].push(seq.clone_ref(py));
                    continue;
                }

                if obj
                    .getattr("status")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0)
                    == 4
                {
                    let _ = obj.setattr("status", 1);
                }

                for (token_offset, token) in tokens.iter().copied().enumerate() {
                    let logprob = dp_group_token_logprobs
                        .as_ref()
                        .and_then(|groups| groups.get(group_idx))
                        .and_then(|group| group.get(seq_idx))
                        .and_then(|values| values.get(token_offset))
                        .copied();
                    let _ = obj
                        .call_method1("append_token", (token, 0, Some(group_id as i32), logprob));
                    let generated = obj
                        .getattr("num_tokens")
                        .and_then(|v| v.extract::<i32>())
                        .unwrap_or(0)
                        - obj
                            .getattr("num_prompt_tokens")
                            .and_then(|v| v.extract::<i32>())
                            .unwrap_or(0);
                    if let Ok(metric) = obj.getattr("metric") {
                        if !metric.is_none() {
                            let method = if generated <= 1 {
                                "record_first_token"
                            } else {
                                "record_token"
                            };
                            let _ = metric.call_method0(method);
                        }
                    }
                }

                let sampling_params = obj.getattr("sampling_params").ok();
                let max_tokens = sampling_params
                    .as_ref()
                    .and_then(|sp| sp.getattr("max_tokens").ok())
                    .and_then(|v| v.extract::<i32>().ok())
                    .unwrap_or(self.config.max_model_len);
                let ignore_eos = sampling_params
                    .as_ref()
                    .and_then(|sp| sp.getattr("ignore_eos").ok())
                    .and_then(|v| v.extract::<bool>().ok())
                    .unwrap_or(false);
                let num_tokens = obj
                    .getattr("num_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0);
                let num_prompt_tokens = obj
                    .getattr("num_prompt_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0);
                let generated = num_tokens - num_prompt_tokens;
                let last_token = obj
                    .getattr("last_token")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(-1);
                let hit_eos = !ignore_eos && eos_ids.contains(&last_token);
                let hit_limit = generated >= max_tokens || num_tokens >= self.config.max_model_len;

                if hit_eos || hit_limit {
                    let _ = obj.setattr("status", 2);
                    self.park_or_release(py, seq);
                } else if self.config.mode == "prefill" {
                    let _ = obj.setattr("status", 3);
                    self.to_be_migrated
                        .insert(seq_id, (seq.clone_ref(py), dp_idx));
                } else {
                    let _ = self.ensure_blocks_for_seq(py, seq, false);
                    let _ = obj.setattr("status", 1);
                    next_running[dp_idx].push(seq.clone_ref(py));
                }
            }
        }

        self.running = next_running;
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
        SchedulerMetricSnapshot {
            running_per_dp: {
                let mut running = vec![0; self.config.attention_dp.max(1) as usize];
                for (idx, seqs) in self.running.iter().enumerate() {
                    if idx < running.len() {
                        running[idx] = seqs.len() as i32;
                    }
                }
                running
            },
            total_waiting: self.waiting.len() as i32,
            total_waiting_migration: self.waiting_migration.len() as i32,
            waiting_migration_head_tokens: -1,
            total_blocks_per_dp: self.config.num_kvcache_blocks * self.config.group_size,
            used_blocks_per_dp: {
                let dp = self.config.attention_dp.max(1) as usize;
                let group = self.config.group_size.max(1) as usize;
                (0..dp)
                    .map(|dp_idx| {
                        (0..group)
                            .map(|group_id| {
                                let idx = dp_idx * group + group_id;
                                self.config.num_kvcache_blocks.max(0)
                                    - self.group_resources[idx].free_blocks.len() as i32
                            })
                            .sum()
                    })
                    .collect()
            },
            free_blocks: {
                let dp = self.config.attention_dp.max(1) as usize;
                let group = self.config.group_size.max(1) as usize;
                (0..dp)
                    .map(|dp_idx| {
                        (0..group)
                            .map(|group_id| {
                                let idx = dp_idx * group + group_id;
                                self.group_resources[idx].free_blocks.len() as i32
                            })
                            .collect()
                    })
                    .collect()
            },
        }
    }

    fn update_server_metric(
        &self,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> SchedulerMetricSnapshot {
        let snapshot = self.metric_snapshot();
        metric.update_running_requests(self.running.iter().map(|seqs| seqs.len() as i32).sum());
        metric.update_waiting_requests(snapshot.total_waiting);
        metric.update_waiting_migration_requests(snapshot.total_waiting_migration);
        metric.update_group_stats(
            result.group_send_counts.clone(),
            result.group_recv_counts.clone(),
        );
        metric.update_waiting_blocks(result.waiting_head_blocks, result.waiting_total_blocks);
        snapshot
    }

    #[pyo3(signature = (metric, result, dp_group_token_ids=None))]
    fn record_step_metric(
        &mut self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        dp_group_token_ids: Option<Vec<Vec<Vec<i32>>>>,
    ) -> StepMetricSnapshot {
        let mut snapshot = StepMetricSnapshot::default();
        metric.update_running_requests(self.running.iter().map(|seqs| seqs.len() as i32).sum());
        metric.update_waiting_requests(self.waiting.len() as i32);
        metric.update_waiting_migration_requests(self.waiting_migration.len() as i32);
        snapshot.real_bs = result.dp_seqs.iter().map(|seqs| seqs.len() as i32).sum();
        if result.is_prefill {
            for (dp_idx, seqs) in result.dp_seqs.iter().enumerate() {
                for seq in seqs {
                    let obj = seq.bind(py);
                    let seq_id = obj
                        .getattr("seq_id")
                        .and_then(|v| v.extract::<u64>())
                        .unwrap_or(0);
                    let n = obj
                        .getattr("num_tokens")
                        .and_then(|v| v.extract::<i32>())
                        .unwrap_or(0);
                    let cached = obj
                        .getattr("num_cached_tokens")
                        .and_then(|v| v.extract::<i32>())
                        .unwrap_or(0);
                    let new_tokens = (n - cached).max(0);
                    snapshot.prefill_tokens += new_tokens;
                    if dp_idx < snapshot.prefill_tokens_per_dp.len() {
                        snapshot.prefill_tokens_per_dp[dp_idx] += new_tokens;
                        if !self.prefix_counted_seq_ids.insert(seq_id) {
                            continue;
                        }
                        self.prefix_cached_tokens_by_seq.insert(seq_id, cached);
                        snapshot.prefix_cached_tokens_per_dp[dp_idx] += cached;
                        snapshot.prefix_prompt_tokens_per_dp[dp_idx] += obj
                            .getattr("num_prompt_tokens")
                            .and_then(|v| v.extract::<i32>())
                            .unwrap_or(0);
                    }
                }
            }
        } else if let Some(tokens) = dp_group_token_ids {
            snapshot.decode_tokens = tokens
                .iter()
                .flat_map(|group| group.iter())
                .map(|seq_tokens| seq_tokens.len() as i32)
                .sum();
        }
        snapshot.prefill_tokens_per_dp = vec![0; self.config.attention_dp.max(1) as usize];
        snapshot.decode_tokens_per_dp = vec![0; self.config.attention_dp.max(1) as usize];
        snapshot.prefix_cached_tokens_per_dp = vec![0; self.config.attention_dp.max(1) as usize];
        snapshot.prefix_prompt_tokens_per_dp = vec![0; self.config.attention_dp.max(1) as usize];
        snapshot
    }

    fn prefix_cached_tokens(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_by_seq
            .get(&seq_id)
            .copied()
            .unwrap_or(0)
    }

    fn clear_finished_metric_state(&mut self, seq_id: u64) {
        self.prefix_counted_seq_ids.remove(&seq_id);
        self.prefix_cached_tokens_by_seq.remove(&seq_id);
    }

    fn abort(&mut self, _seq_id: u64) -> bool {
        self.release_seq(_seq_id);
        self.waiting.retain(|seq| {
            Python::with_gil(|py| {
                seq.bind(py)
                    .getattr("seq_id")
                    .and_then(|v| v.extract::<u64>())
                    .map(|id| id != _seq_id)
                    .unwrap_or(true)
            })
        });
        self.waiting_migration.retain(|seq| {
            Python::with_gil(|py| {
                seq.bind(py)
                    .getattr("seq_id")
                    .and_then(|v| v.extract::<u64>())
                    .map(|id| id != _seq_id)
                    .unwrap_or(true)
            })
        });
        for queue in &mut self.running {
            queue.retain(|seq| {
                Python::with_gil(|py| {
                    seq.bind(py)
                        .getattr("seq_id")
                        .and_then(|v| v.extract::<u64>())
                        .map(|id| id != _seq_id)
                        .unwrap_or(true)
                })
            });
        }
        for queue in &mut self.prefilling {
            queue.retain(|seq| {
                Python::with_gil(|py| {
                    seq.bind(py)
                        .getattr("seq_id")
                        .and_then(|v| v.extract::<u64>())
                        .map(|id| id != _seq_id)
                        .unwrap_or(true)
                })
            });
        }
        self.to_be_migrated.remove(&_seq_id);
        self.session_wait.remove(&_seq_id);
        true
    }

    fn free_to_be_migrated(&mut self, py: Python<'_>, seqs: &Bound<'_, PyAny>) -> PyResult<()> {
        let seqs: Vec<Py<PyAny>> = seqs.extract()?;
        for seq in seqs {
            if let Ok(seq_id) = seq
                .bind(py)
                .getattr("seq_id")
                .and_then(|v| v.extract::<u64>())
            {
                if let Some((seq, _dp_idx)) = self.to_be_migrated.remove(&seq_id) {
                    self.release_seq(seq_id);
                    let _ = seq.bind(py).setattr("status", 2);
                } else {
                    self.release_seq(seq_id);
                }
            }
        }
        Ok(())
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

    fn prompt_target(&self, obj: &Bound<'_, PyAny>) -> i32 {
        obj.getattr("num_prompt_tokens")
            .and_then(|v| v.extract::<i32>())
            .unwrap_or(0)
            .max(
                obj.getattr("num_checkpointed_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0),
            )
    }

    fn schedule_prefill(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<Py<PyAny>>> = (0..dp).map(|_| Vec::new()).collect();
        let mut num_seqs = vec![vec![0i32; group]; dp];
        let mut num_tokens = vec![vec![0i32; group]; dp];

        for dp_idx in 0..dp {
            let mut rest = Vec::new();
            let continuations = std::mem::take(&mut self.prefilling[dp_idx]);
            for seq in continuations {
                let obj = seq.bind(py);
                let group_id = obj
                    .getattr("active_group_id")
                    .and_then(|v| v.extract::<usize>())
                    .unwrap_or(0)
                    .min(group - 1);
                let target = self.prompt_target(&obj);
                let cur = obj.getattr("num_tokens")?.extract::<i32>()?;
                let budget =
                    self.config.max_num_batched_tokens.max(1) - num_tokens[dp_idx][group_id];
                let new_tokens = (target - cur).min(budget).max(0);
                if new_tokens <= 0 {
                    rest.push(seq);
                    continue;
                }
                obj.setattr("num_tokens", cur + new_tokens)?;
                obj.call_method1(
                    "set_active_dispatched_tokens",
                    (self.dispatch_for_master(group_id, cur + new_tokens),),
                )?;
                num_seqs[dp_idx][group_id] += 1;
                num_tokens[dp_idx][group_id] += new_tokens;
                self.running[dp_idx].push(seq.clone_ref(py));
                scheduled[dp_idx].push(seq);
            }
            self.prefilling[dp_idx] = rest;
        }

        let use_migration_queue = self.config.mode == "decode";
        loop {
            let seq = if use_migration_queue {
                self.waiting_migration.first().map(|s| s.clone_ref(py))
            } else {
                self.waiting.first().map(|s| s.clone_ref(py))
            };
            let Some(seq) = seq else {
                break;
            };
            let mut placed = false;
            for dp_idx in self.route_candidates(py, &seq) {
                if let Some(new_tokens) = self.try_allocate_prefill(
                    py,
                    &seq,
                    dp_idx,
                    &num_seqs[dp_idx],
                    &num_tokens[dp_idx],
                )? {
                    let group_id = seq
                        .bind(py)
                        .getattr("active_group_id")
                        .and_then(|v| v.extract::<usize>())
                        .unwrap_or(0)
                        .min(group - 1);
                    num_seqs[dp_idx][group_id] += 1;
                    num_tokens[dp_idx][group_id] += new_tokens;
                    if use_migration_queue {
                        self.waiting_migration.remove(0);
                    } else {
                        self.waiting.remove(0);
                    }
                    self.running[dp_idx].push(seq.clone_ref(py));
                    scheduled[dp_idx].push(seq.clone_ref(py));
                    let affinity = seq
                        .bind(py)
                        .getattr("affinity_key")
                        .and_then(|v| v.extract::<u64>())
                        .unwrap_or(0);
                    if affinity != 0 {
                        self.session_affinity.insert(affinity, dp_idx);
                    }
                    if let Ok(seq_id) = seq
                        .bind(py)
                        .getattr("seq_id")
                        .and_then(|v| v.extract::<u64>())
                    {
                        self.session_wait.remove(&seq_id);
                    }
                    placed = true;
                    break;
                }
            }
            if !placed {
                break;
            }
        }
        Ok(scheduled)
    }

    fn schedule_decode(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<Py<PyAny>>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<Py<PyAny>>> = (0..dp).map(|_| Vec::new()).collect();
        for dp_idx in 0..dp {
            let mut skipped = Vec::new();
            let mut per_group = vec![0i32; group];
            let mut group_lens = vec![0i32; group];
            let queue = std::mem::take(&mut self.running[dp_idx]);
            for seq in queue {
                let obj = seq.bind(py);
                let group_id = obj
                    .getattr("active_group_id")
                    .and_then(|v| v.extract::<usize>())
                    .unwrap_or(0)
                    .min(group - 1);
                if per_group[group_id] >= self.config.max_num_seqs.max(1) {
                    skipped.push(seq);
                    continue;
                }
                if let Err(_e) = self.ensure_blocks_for_seq(py, &seq, false) {
                    self.preempt(py, dp_idx, seq)?;
                    continue;
                }
                per_group[group_id] += 1;
                group_lens[group_id] += obj.getattr("num_tokens")?.extract::<i32>()?;
                scheduled[dp_idx].push(seq.clone_ref(py));
                skipped.push(seq);
            }
            self.running[dp_idx] = skipped;
            for group_id in 0..group {
                if group_lens[group_id] == 0 && group > 1 {
                    scheduled[dp_idx].push(self.make_dummy_seq(py, group_id as i32)?);
                }
            }
        }
        Ok(scheduled)
    }

    fn make_schedule_result(
        &self,
        py: Python<'_>,
        dp_seqs: Vec<Vec<Py<PyAny>>>,
        is_prefill: bool,
    ) -> PyResult<ScheduleResult> {
        let dp = self.dp();
        let group = self.group();
        let mut result = ScheduleResult::default();
        result.is_prefill = is_prefill;
        result.dp_seqs = dp_seqs
            .iter()
            .map(|seqs| seqs.iter().map(|s| s.clone_ref(py)).collect())
            .collect();
        result.dp_group_seqs = Vec::with_capacity(dp * group);
        result.filtered_dp_group_seqs = Vec::with_capacity(dp * group);
        result.group_send_counts = vec![vec![0; group]; dp];
        result.group_recv_counts = vec![vec![0; group]; dp];
        result.group_q_matrix = vec![vec![vec![0; group]; group]; dp];

        for dp_idx in 0..dp {
            for group_id in 0..group {
                result.dp_group_seqs.push(
                    dp_seqs[dp_idx]
                        .iter()
                        .map(|seq| seq.clone_ref(py))
                        .collect(),
                );
                let filtered = dp_seqs[dp_idx]
                    .iter()
                    .filter(|seq| {
                        seq.bind(py)
                            .getattr("active_group_id")
                            .and_then(|v| v.extract::<usize>())
                            .map(|gid| gid == group_id)
                            .unwrap_or(false)
                    })
                    .map(|seq| seq.clone_ref(py))
                    .collect::<Vec<_>>();
                result.filtered_dp_group_seqs.push(filtered);
            }
        }

        for dp_idx in 0..dp {
            for seq in &dp_seqs[dp_idx] {
                let obj = seq.bind(py);
                let master = obj
                    .getattr("active_group_id")
                    .and_then(|v| v.extract::<usize>())
                    .unwrap_or(0)
                    .min(group - 1);
                let tokens = obj
                    .getattr("active_dispatched_tokens")
                    .and_then(|v| v.extract::<Vec<i32>>())
                    .unwrap_or_default();
                let active = tokens.iter().filter(|count| **count > 0).count();
                if active > 1 {
                    result.group_send_counts[dp_idx][master] += 1;
                    for gid in 0..group {
                        if tokens.get(gid).copied().unwrap_or(0) > 0 {
                            result.group_recv_counts[dp_idx][gid] += 1;
                            result.group_q_matrix[dp_idx][master][gid] += 1;
                        }
                    }
                }
            }
        }

        let wait_queue = if self.config.mode == "decode" {
            &self.waiting_migration
        } else {
            &self.waiting
        };
        if let Some(head) = wait_queue.first() {
            let n = head
                .bind(py)
                .getattr("num_tokens")
                .and_then(|v| v.extract::<i32>())
                .unwrap_or(0);
            result.waiting_head_blocks = self.blocks_needed_for_tokens(n) as i32;
        }
        result.waiting_total_blocks = wait_queue
            .iter()
            .map(|seq| {
                let n = seq
                    .bind(py)
                    .getattr("num_tokens")
                    .and_then(|v| v.extract::<i32>())
                    .unwrap_or(0);
                self.blocks_needed_for_tokens(n) as i32
            })
            .sum();
        Ok(result)
    }

    fn try_allocate_prefill(
        &mut self,
        py: Python<'_>,
        seq: &Py<PyAny>,
        dp_idx: usize,
        batch_seqs: &[i32],
        batch_tokens: &[i32],
    ) -> PyResult<Option<i32>> {
        let obj = seq.bind(py);
        let seq_id = obj.getattr("seq_id")?.extract::<u64>()?;
        let full_len = self.prompt_target(&obj);
        let cached = obj
            .getattr("num_cached_tokens")
            .and_then(|v| v.extract::<i32>())
            .unwrap_or(0);
        if let Some(new_tokens) = self.try_adopt_session(py, seq, dp_idx, batch_tokens)? {
            return Ok(Some(new_tokens));
        }
        let Some(master) = self.choose_master_group(py, dp_idx, batch_seqs, batch_tokens) else {
            return Ok(None);
        };
        let budget = (self.config.max_num_batched_tokens.max(1)
            - batch_tokens.get(master).copied().unwrap_or(0))
        .max(0);
        if budget <= 0 {
            return Ok(None);
        }
        let new_tokens = (full_len - cached).min(budget).max(0);
        if new_tokens <= 0 {
            return Ok(None);
        }
        let chunk_end = cached + new_tokens;
        self.seq_assignment.insert(seq_id, (dp_idx, master));
        obj.setattr("active_dp_idx", dp_idx as i32)?;
        obj.setattr("active_group_id", master as i32)?;
        obj.setattr("migrate_group_id", master as i32)?;
        obj.setattr("num_tokens", chunk_end)?;
        obj.setattr("status", 1)?;
        let dispatch = self.compute_dispatch(py, dp_idx, master, full_len);
        obj.call_method1("set_active_dispatched_tokens", (dispatch.clone(),))?;

        for (gid, count) in dispatch.iter().copied().enumerate() {
            if count > 0 {
                self.ensure_group_blocks(py, seq, seq_id, dp_idx, gid, count, true)?;
            }
        }
        obj.setattr("active_group_id", master as i32)?;
        obj.setattr("migrate_group_id", master as i32)?;
        self.ensure_state_slot(py, &obj, seq_id)?;
        self.ensure_hisparse_slot(py, &obj, seq_id)?;
        self.ensure_compressed_pages(py, &obj, seq_id, full_len)?;
        self.prefix_cached_tokens_by_seq.insert(seq_id, cached);
        Ok(Some(new_tokens))
    }

    fn ensure_group_blocks(
        &mut self,
        py: Python<'_>,
        seq: &Py<PyAny>,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        tokens: i32,
        set_migrate: bool,
    ) -> PyResult<Vec<i32>> {
        let needed_blocks = self.blocks_needed_for_tokens(tokens);
        let flat = self.flat_idx(dp_idx, group_id);
        let blocks = {
            let resource = &mut self.group_resources[flat];
            let blocks = resource.seq_blocks.entry(seq_id).or_default();
            while blocks.len() < needed_blocks {
                let Some(block) = resource.free_blocks.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={} have={}",
                        needed_blocks,
                        blocks.len()
                    )));
                };
                blocks.push(block);
            }
            blocks.clone()
        };
        let obj = seq.bind(py);
        obj.call_method1(
            "set_active_group_block_table",
            (group_id as i32, blocks.clone()),
        )?;
        if set_migrate {
            obj.call_method1(
                "set_migrate_group_block_table",
                (group_id as i32, blocks.clone()),
            )?;
            obj.call_method1("set_migrate_engine_id", (self.engine_id_.clone(),))?;
            obj.setattr("migrate_num_kvcache_blocks", self.config.num_kvcache_blocks)?;
            obj.setattr("migrate_group_size", self.config.group_size)?;
            obj.setattr("migrate_dp_idx", dp_idx as i32)?;
            obj.setattr("migrate_group_id", group_id as i32)?;
        }
        Ok(blocks)
    }

    fn compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
    }

    fn ensure_blocks_for_seq(
        &mut self,
        py: Python<'_>,
        seq: &Py<PyAny>,
        is_prefill: bool,
    ) -> PyResult<()> {
        let obj = seq.bind(py);
        let seq_id = obj.getattr("seq_id")?.extract::<u64>()?;
        let num_tokens = obj.getattr("num_tokens")?.extract::<i32>()?;
        let needed_tokens = if is_prefill {
            num_tokens
        } else {
            num_tokens + 1
        };
        let needed_blocks = self.blocks_needed_for_tokens(needed_tokens);
        let (dp_idx, group_id) = match self.seq_assignment.get(&seq_id).copied() {
            Some(assignment) => assignment,
            None => {
                let assignment = self.choose_assignment();
                self.seq_assignment.insert(seq_id, assignment);
                assignment
            }
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let blocks = {
            let resource = &mut self.group_resources[flat];
            let blocks = resource.seq_blocks.entry(seq_id).or_default();
            while blocks.len() < needed_blocks {
                let Some(block) = resource.free_blocks.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={} have={}",
                        needed_blocks,
                        blocks.len()
                    )));
                };
                blocks.push(block);
            }
            blocks.clone()
        };
        obj.call_method1(
            "set_active_group_block_table",
            (group_id as i32, blocks.clone()),
        )?;
        obj.setattr("active_dp_idx", dp_idx as i32)?;
        obj.setattr("active_group_id", group_id as i32)?;

        self.ensure_state_slot(py, &obj, seq_id)?;
        self.ensure_hisparse_slot(py, &obj, seq_id)?;
        self.ensure_compressed_pages(py, &obj, seq_id, needed_tokens)?;

        if is_prefill {
            obj.call_method1(
                "set_migrate_group_block_table",
                (group_id as i32, blocks.clone()),
            )?;
            obj.call_method1("set_migrate_engine_id", (self.engine_id_.clone(),))?;
            obj.setattr("migrate_num_kvcache_blocks", self.config.num_kvcache_blocks)?;
            obj.setattr("migrate_group_size", self.config.group_size)?;
            obj.setattr("migrate_dp_idx", dp_idx as i32)?;
            obj.setattr("migrate_group_id", group_id as i32)?;
            if let Some(slot) = self.seq_state_slots.get(&seq_id).copied() {
                obj.setattr("migrate_state_slot", slot)?;
            }
            if let Some(slot) = self.seq_hisparse_slots.get(&seq_id).copied() {
                obj.setattr("migrate_hisparse_slot", slot)?;
            }
            for (ratio, pool) in &self.compressed_pools {
                if let Some(pages) = pool.seq_pages.get(&seq_id) {
                    obj.call_method1(
                        "set_migrate_compressed_block_table",
                        (*ratio, pages.clone()),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn ensure_state_slot(
        &mut self,
        _py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        seq_id: u64,
    ) -> PyResult<()> {
        if self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) == 0 {
            return Ok(());
        }
        let slot = match self.seq_state_slots.get(&seq_id).copied() {
            Some(slot) => slot,
            None => {
                let Some(slot) = self.state_free.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of state slots: seq_id={seq_id}"
                    )));
                };
                self.seq_state_slots.insert(seq_id, slot);
                slot
            }
        };
        obj.setattr("active_state_slot", slot)?;
        Ok(())
    }

    fn ensure_hisparse_slot(
        &mut self,
        _py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        seq_id: u64,
    ) -> PyResult<()> {
        if self.config.cache_plan.flags & (1 << 6) == 0 {
            return Ok(());
        }
        let slot = match self.seq_hisparse_slots.get(&seq_id).copied() {
            Some(slot) => slot,
            None => {
                let Some(slot) = self.hisparse_free.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of HiSparse slots: seq_id={seq_id}"
                    )));
                };
                self.seq_hisparse_slots.insert(seq_id, slot);
                slot
            }
        };
        obj.setattr("active_hisparse_slot", slot)?;
        Ok(())
    }

    fn ensure_compressed_pages(
        &mut self,
        _py: Python<'_>,
        obj: &Bound<'_, PyAny>,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<()> {
        if self.compressed_pools.is_empty() {
            return Ok(());
        }
        obj.call_method0("clear_active_compressed_block_tables")?;
        for (ratio, pool) in self.compressed_pools.iter_mut() {
            let needed_pages = Self::compressed_pages_needed_for_tokens(pool, needed_tokens);
            let pages = pool.seq_pages.entry(seq_id).or_default();
            while pages.len() < needed_pages {
                let Some(page) = pool.free_pages.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of compressed cache pages: seq_id={seq_id} ratio={ratio} need={} have={}",
                        needed_pages,
                        pages.len()
                    )));
                };
                pages.push(page);
            }
            obj.call_method1("set_active_compressed_block_table", (*ratio, pages.clone()))?;
        }
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RoutingStrategy>()?;
    m.add_class::<SchedulerConfig>()?;
    m.add_class::<ScheduleResult>()?;
    m.add_class::<Scheduler>()?;
    Ok(())
}
