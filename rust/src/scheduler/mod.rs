use crate::metrics::{RuntimeMetrics, SequenceMetric, ServerMetric};
use crate::proto::wire::sequence_to_migration_request_ref;
use crate::proto::RunnerOut;
use crate::sequence::Sequence;
use crate::table::block::{BlockPool, CompressedPool};
use crate::table::slot::SlotPool;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

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

const HOST_SWAP_IN_COOLDOWN_STEPS: u64 = 2;
const HOST_PREFIX_CACHE_SEQ_ID: u64 = u64::MAX;

#[derive(Clone, Debug)]
struct PendingHostSwap {
    dp_idx: usize,
    group_id: usize,
}

#[pyclass(module = "dlengine._engine", unsendable)]
pub struct Scheduler {
    #[pyo3(get)]
    pub engine_id_: String,
    #[pyo3(get)]
    pub routing_strategy: i32,
    config: SchedulerConfig,
    seq_table: HashMap<u64, Sequence>,
    waiting: Vec<u64>,
    waiting_migration: Vec<u64>,
    running: Vec<Vec<u64>>,
    prefilling: Vec<Vec<u64>>,
    to_be_migrated: HashMap<u64, usize>,
    dummy_seq_ids: HashSet<u64>,
    hbm_pools: Vec<BlockPool>,
    host_pools: Vec<BlockPool>,
    pending_host_swaps: HashMap<u64, PendingHostSwap>,
    pending_swap_out_tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    pending_swap_in_tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    current_step: u64,
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
    prefix_caching_enabled: bool,
    prefix_cached_tokens_by_seq: HashMap<u64, i32>,
    prefix_counted_seq_ids: HashSet<u64>,
    server_metric: ServerMetric,
    runtime_metrics: RuntimeMetrics,
    sequence_metrics: HashMap<u64, Py<SequenceMetric>>,
}

mod api;
mod api_control;
mod api_metrics;

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
    fn add_sequence(&mut self, seq: Sequence) {
        let seq_id = seq.seq_id;
        self.seq_table.insert(seq_id, seq);
        if self.config.mode == "decode" {
            self.waiting_migration.push(seq_id);
        } else {
            self.waiting.push(seq_id);
        }
    }

    fn complete_sequence_metric_py(
        &mut self,
        py: Python<'_>,
        metric: &Py<SequenceMetric>,
    ) -> Option<String> {
        let mut metric = metric.borrow_mut(py);
        self.record_sequence_completion_api(py, &mut metric)
            .then(|| metric.metric_report())
    }

    fn collect_sequence_events_impl(
        &mut self,
        py: Python<'_>,
        dp_seq_ids: Vec<Vec<u64>>,
        track_running: bool,
        previous_running: std::collections::HashSet<u64>,
    ) -> PyResult<(Vec<PyObject>, Vec<u64>)> {
        let mut out = Vec::new();
        let mut completed = Vec::new();
        for seqs in dp_seq_ids {
            for seq_id in seqs {
                if self.dummy_seq_ids.contains(&seq_id) {
                    continue;
                }
                let (
                    last_token,
                    num_tokens,
                    is_finished,
                    is_to_be_migrated,
                    source_engine_id,
                    vision_slots,
                ) = {
                    let Some(s) = self.seq_table.get(&seq_id) else {
                        continue;
                    };
                    (
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
                    let Some(seq) = self.seq_table.get(&seq_id) else {
                        continue;
                    };
                    let migration = sequence_to_migration_request_ref(seq);
                    let bytes = crate::proto::wire::encode_binary(&migration, "migration request")?;
                    event.set_item("migration_payload", PyBytes::new(py, &bytes))?;
                }
                if !vision_slots.is_empty() {
                    if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                        seq.vision_slots.clear();
                    }
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
mod tests;
