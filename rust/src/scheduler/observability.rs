use super::{ScheduleResult, Scheduler};
use crate::logging;
use crate::metrics::{RuntimeMetrics, ServerMetric};
use crate::proto::RunnerOut;
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn metric_snapshot_impl(&self) -> SchedulerMetricSnapshot {
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
            total_host_blocks_per_dp: self.config.num_host_kvcache_blocks * self.config.group_size,
            used_blocks_per_dp: {
                let dp = self.config.attention_dp.max(1) as usize;
                let group = self.config.group_size.max(1) as usize;
                (0..dp)
                    .map(|dp_idx| {
                        (0..group)
                            .map(|group_id| {
                                let idx = dp_idx * group + group_id;
                                self.cache.hbm_pools[idx].num_used_blocks()
                            })
                            .sum()
                    })
                    .collect()
            },
            used_host_blocks_per_dp: {
                let dp = self.config.attention_dp.max(1) as usize;
                let group = self.config.group_size.max(1) as usize;
                (0..dp)
                    .map(|dp_idx| {
                        (0..group)
                            .map(|group_id| {
                                let idx = dp_idx * group + group_id;
                                self.cache.host_pools[idx].num_used_blocks()
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
                                self.cache.hbm_pools[idx].num_free_blocks() as i32
                            })
                            .collect()
                    })
                    .collect()
            },
            used_hisparse_slots: self.cache.hisparse_slots.num_used_slots(),
            total_hisparse_slots: self.cache.hisparse_slots.num_slots(),
        }
    }

    pub(super) fn update_server_metric_impl(
        &self,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> SchedulerMetricSnapshot {
        let snapshot = self.metric_snapshot_impl();
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

    pub(super) fn update_server_metric_and_log_impl(
        &self,
        py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
    ) -> PyResult<SchedulerMetricSnapshot> {
        let snapshot = self.update_server_metric_impl(metric, result);
        if logging::get_log_level() >= 2 {
            let debug = result.debug_summary(
                py,
                self.config.attention_dp.max(1) as usize,
                self.config.group_size.max(1) as usize,
                snapshot.free_blocks.clone(),
            )?;
            let repr = debug.bind(py).repr()?.to_str()?.to_owned();
            eprintln!("[DLENGINE][DEBUG] scheduler - {repr}");
        }
        Ok(snapshot)
    }

    pub(super) fn record_step_metric_impl(
        &mut self,
        _py: Python<'_>,
        metric: &mut ServerMetric,
        result: &ScheduleResult,
        dp_group_token_ids: Option<Vec<Vec<Vec<i32>>>>,
    ) -> StepMetricSnapshot {
        let mut snapshot = StepMetricSnapshot::default();
        let dp = self.config.attention_dp.max(1) as usize;
        snapshot.prefill_tokens_per_dp = vec![0; dp];
        snapshot.decode_tokens_per_dp = vec![0; dp];
        snapshot.prefix_cached_tokens_per_dp = vec![0; dp];
        snapshot.prefix_prompt_tokens_per_dp = vec![0; dp];

        metric.update_running_requests(self.running.iter().map(|seqs| seqs.len() as i32).sum());
        metric.update_waiting_requests(self.waiting.len() as i32);
        metric.update_waiting_migration_requests(self.waiting_migration.len() as i32);
        snapshot.real_bs = result.dp_seq_ids.iter().map(|seqs| seqs.len() as i32).sum();
        if result.is_prefill {
            for (dp_idx, seqs) in result.dp_seq_ids.iter().enumerate() {
                for seq_id in seqs {
                    let Some(s) = self.seq_table.get(seq_id) else {
                        continue;
                    };
                    let seq_id = s.seq_id;
                    let n = s.num_tokens;
                    let cached = s.num_cached_tokens;
                    let new_tokens = (n - s.prefill_start_offset).max(0);
                    snapshot.prefill_tokens += new_tokens;
                    if dp_idx < snapshot.prefill_tokens_per_dp.len() {
                        snapshot.prefill_tokens_per_dp[dp_idx] += new_tokens;
                        if !self.cache.prefix_counted_seq_ids.insert(seq_id) {
                            continue;
                        }
                        self.cache
                            .prefix_cached_tokens_by_seq
                            .insert(seq_id, cached);
                        snapshot.prefix_cached_tokens_per_dp[dp_idx] += cached;
                        snapshot.prefix_prompt_tokens_per_dp[dp_idx] += s.num_prompt_tokens;
                    }
                }
            }
        } else if let Some(tokens) = dp_group_token_ids {
            snapshot.decode_tokens = tokens
                .iter()
                .flat_map(|group| group.iter())
                .map(|seq_tokens| seq_tokens.len() as i32)
                .sum();
            let group = self.config.group_size.max(1) as usize;
            for (group_idx, group_tokens) in tokens.iter().enumerate() {
                let dp_idx = group_idx / group;
                if dp_idx >= snapshot.decode_tokens_per_dp.len() {
                    continue;
                }
                snapshot.decode_tokens_per_dp[dp_idx] += group_tokens
                    .iter()
                    .map(|seq_tokens| seq_tokens.len() as i32)
                    .sum::<i32>();
            }
        }
        snapshot
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_step_metrics_and_report_impl(
        &mut self,
        py: Python<'_>,
        runtime: &mut RuntimeMetrics,
        metric: &mut ServerMetric,
        scheduler_metric: &SchedulerMetricSnapshot,
        result: &ScheduleResult,
        postprocess_ms: f64,
        post_sch_begin: f64,
        runner_outs: Option<Vec<RunnerOut>>,
    ) -> (StepMetricSnapshot, Option<String>) {
        let token_ids = runner_outs.as_ref().map(|outs| {
            outs.iter()
                .map(|out| {
                    out.token_ids
                        .iter()
                        .map(|seq_tokens| seq_tokens.iter().copied().map(|t| t as i32).collect())
                        .collect()
                })
                .collect()
        });
        let step_metric = self.record_step_metric_impl(py, metric, result, token_ids);
        let resource_metric = self.metric_snapshot_impl();
        let message = runtime.maybe_report_step_status(
            metric,
            self.config.engine_id.clone(),
            self.config.mode.clone(),
            scheduler_metric,
            &resource_metric,
            &step_metric,
            result.schedule_latency_ms,
            postprocess_ms,
            result.schedule_end_s,
            post_sch_begin,
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        );
        (step_metric, message)
    }
}
