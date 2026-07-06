use super::{SequenceMetric, ServerMetric};
use crate::common::now_seconds;
use crate::snapshots::{SchedulerMetricSnapshot, StepMetricSnapshot};
use pyo3::prelude::*;

const TTFT_BUCKETS: [f64; 12] = [
    0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0,
];
const TPOT_BUCKETS: [f64; 9] = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];

fn prom_value(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

fn human_bytes(n: f64) -> String {
    let mut step = n;
    for unit in ["B", "KiB", "MiB", "GiB"] {
        if step < 1024.0 || unit == "GiB" {
            if unit == "B" {
                return format!("{}B", step as i64);
            }
            return format!("{step:.1}{unit}");
        }
        step /= 1024.0;
    }
    format!("{step:.1}GiB")
}

fn rates(numerators: &[i32], denominators: &[i32]) -> Vec<f64> {
    numerators
        .iter()
        .zip(denominators.iter())
        .map(|(n, d)| if *d > 0 { *n as f64 / *d as f64 } else { 0.0 })
        .collect()
}

#[pyclass(module = "dlengine._engine")]
pub struct RuntimeMetrics {
    #[pyo3(get)]
    pub engine_id: String,
    #[pyo3(get)]
    pub engine_mode: String,
    #[pyo3(get)]
    pub running_per_dp: Vec<i32>,
    #[pyo3(get)]
    pub used_blocks_per_dp: Vec<i32>,
    #[pyo3(get)]
    pub total_blocks_per_dp: i32,
    #[pyo3(get)]
    pub used_host_blocks_per_dp: Vec<i32>,
    #[pyo3(get)]
    pub total_host_blocks_per_dp: i32,
    #[pyo3(get)]
    pub used_hisparse_slots: i32,
    #[pyo3(get)]
    pub total_hisparse_slots: i32,
    #[pyo3(get)]
    pub last_schedule_ms: f64,
    #[pyo3(get)]
    pub last_forward_ms: f64,
    #[pyo3(get)]
    pub last_postprocess_ms: f64,
    #[pyo3(get)]
    pub last_step_count: i32,
    #[pyo3(get)]
    pub last_forward_tx_bytes: i64,
    #[pyo3(get)]
    pub last_forward_rx_bytes: i64,
    #[pyo3(get)]
    pub last_transfer_ms: f64,
    #[pyo3(get)]
    pub last_wwi_ms: f64,
    #[pyo3(get)]
    pub last_immrecv_ms: f64,
    #[pyo3(get)]
    pub last_net_ms: f64,
    #[pyo3(get)]
    pub last_serialize_ms: f64,
    #[pyo3(get)]
    pub last_prefix_cache_hit_rate: f64,
    #[pyo3(get)]
    pub last_prefix_cache_hit_rate_per_dp: Vec<f64>,
    #[pyo3(get)]
    pub last_ttft_ms: f64,
    #[pyo3(get)]
    pub last_tpot_ms: f64,
    ttft_sum_s: f64,
    ttft_count: i64,
    tpot_sum_s: f64,
    tpot_count: i64,
    ttft_bucket_counts: Vec<i64>,
    tpot_bucket_counts: Vec<i64>,
    report_interval_s: f64,
    last_report_time: f64,
    report_prefill_tokens_per_dp: Option<Vec<i32>>,
    report_decode_tokens_per_dp: Option<Vec<i32>>,
    report_step_count: i32,
    report_sched_ms: f64,
    report_forward_ms: f64,
    report_post_ms: f64,
    report_fwd_tx_bytes: i64,
    report_fwd_rx_bytes: i64,
    report_transfer_ms: f64,
    report_wwi_ms: f64,
    report_immrecv_ms: f64,
    report_net_ms: f64,
    report_serialize_ms: f64,
    report_prefix_cached_tokens_per_dp: Option<Vec<i32>>,
    report_prefix_prompt_tokens_per_dp: Option<Vec<i32>>,
}

#[pymethods]
impl RuntimeMetrics {
    #[new]
    #[pyo3(signature = (report_interval_s = 5.0))]
    fn new(report_interval_s: f64) -> Self {
        runtime_metrics_with_report_interval(report_interval_s)
    }

    pub(crate) fn record_sequence_completion(
        &mut self,
        metric: &mut SequenceMetric,
        server_metric: &mut ServerMetric,
    ) -> bool {
        metric.record_completion();
        server_metric.add_tokens(0, metric.num_generated_tokens as i64);
        server_metric.add_completed_request();
        if let Some(ttft_ms) = metric.ttft() {
            let ttft_s = ttft_ms / 1000.0;
            self.ttft_sum_s += ttft_s;
            self.ttft_count += 1;
            self.last_ttft_ms = ttft_ms;
            for (idx, bucket) in TTFT_BUCKETS.iter().enumerate() {
                if ttft_s <= *bucket {
                    self.ttft_bucket_counts[idx] += 1;
                }
            }
        }
        if let Some(tpot_ms) = metric.avg_tpot_wo_queueing() {
            let tpot_s = tpot_ms / 1000.0;
            self.tpot_sum_s += tpot_s;
            self.tpot_count += 1;
            self.last_tpot_ms = tpot_ms;
            for (idx, bucket) in TPOT_BUCKETS.iter().enumerate() {
                if tpot_s <= *bucket {
                    self.tpot_bucket_counts[idx] += 1;
                }
            }
        }
        metric.first_token_time.is_some() || metric.num_generated_tokens > 0
    }

    #[pyo3(signature = (
        server_metric,
        engine_id,
        mode,
        scheduler_metric,
        resource_metric,
        step_metric,
        schedule_ms,
        postprocess_ms,
        sch_end,
        post_sch_begin,
        forward_tx_bytes=0,
        forward_rx_bytes=0,
        transfer_ms=0.0,
        wwi_ms=0.0,
        immrecv_ms=0.0,
        net_ms=0.0,
        serialize_ms=0.0
    ))]
    pub(crate) fn maybe_report_step_status(
        &mut self,
        server_metric: &mut ServerMetric,
        engine_id: String,
        mode: String,
        scheduler_metric: &SchedulerMetricSnapshot,
        resource_metric: &SchedulerMetricSnapshot,
        step_metric: &StepMetricSnapshot,
        schedule_ms: f64,
        postprocess_ms: f64,
        sch_end: f64,
        post_sch_begin: f64,
        forward_tx_bytes: i64,
        forward_rx_bytes: i64,
        transfer_ms: f64,
        wwi_ms: f64,
        immrecv_ms: f64,
        net_ms: f64,
        serialize_ms: f64,
    ) -> Option<String> {
        let forward_ms = if post_sch_begin != 0.0 {
            (post_sch_begin - sch_end) * 1000.0
        } else {
            0.0
        };
        self.maybe_report_engine_status(
            server_metric,
            engine_id,
            mode,
            scheduler_metric.running_per_dp.clone(),
            scheduler_metric.total_waiting,
            scheduler_metric.total_waiting_migration,
            Some(resource_metric.used_blocks_per_dp.clone()),
            resource_metric.total_blocks_per_dp,
            step_metric.prefill_tokens_per_dp.clone(),
            step_metric.decode_tokens_per_dp.clone(),
            Some(step_metric.prefix_cached_tokens_per_dp.clone()),
            Some(step_metric.prefix_prompt_tokens_per_dp.clone()),
            schedule_ms,
            forward_ms,
            postprocess_ms,
            forward_tx_bytes,
            forward_rx_bytes,
            transfer_ms,
            wwi_ms,
            immrecv_ms,
            net_ms,
            serialize_ms,
            Some(resource_metric.used_host_blocks_per_dp.clone()),
            resource_metric.total_host_blocks_per_dp,
            resource_metric.used_hisparse_slots,
            resource_metric.total_hisparse_slots,
        )
    }

    #[pyo3(signature = (
        server_metric,
        engine_id,
        mode,
        running_per_dp,
        waiting,
        waiting_migration,
        used_blocks_per_dp,
        total_blocks,
        prefill_tokens_per_dp,
        decode_tokens_per_dp,
        prefix_cached_tokens_per_dp=None,
        prefix_prompt_tokens_per_dp=None,
        schedule_ms=0.0,
        forward_ms=0.0,
        postprocess_ms=0.0,
        forward_tx_bytes=0,
        forward_rx_bytes=0,
        transfer_ms=0.0,
        wwi_ms=0.0,
        immrecv_ms=0.0,
        net_ms=0.0,
        serialize_ms=0.0,
        used_host_blocks_per_dp=None,
        total_host_blocks=0,
        used_hisparse_slots=0,
        total_hisparse_slots=0
    ))]
    fn maybe_report_engine_status(
        &mut self,
        server_metric: &mut ServerMetric,
        engine_id: String,
        mode: String,
        running_per_dp: Vec<i32>,
        waiting: i32,
        waiting_migration: i32,
        used_blocks_per_dp: Option<Vec<i32>>,
        total_blocks: i32,
        prefill_tokens_per_dp: Vec<i32>,
        decode_tokens_per_dp: Vec<i32>,
        prefix_cached_tokens_per_dp: Option<Vec<i32>>,
        prefix_prompt_tokens_per_dp: Option<Vec<i32>>,
        schedule_ms: f64,
        forward_ms: f64,
        postprocess_ms: f64,
        forward_tx_bytes: i64,
        forward_rx_bytes: i64,
        transfer_ms: f64,
        wwi_ms: f64,
        immrecv_ms: f64,
        net_ms: f64,
        serialize_ms: f64,
        used_host_blocks_per_dp: Option<Vec<i32>>,
        total_host_blocks: i32,
        used_hisparse_slots: i32,
        total_hisparse_slots: i32,
    ) -> Option<String> {
        let force_report = prefill_tokens_per_dp.iter().any(|tokens| *tokens > 0);
        if self.report_prefill_tokens_per_dp.is_none() {
            self.report_prefill_tokens_per_dp = Some(vec![0; prefill_tokens_per_dp.len()]);
            self.report_decode_tokens_per_dp = Some(vec![0; decode_tokens_per_dp.len()]);
        }
        if let Some(acc) = self.report_prefill_tokens_per_dp.as_mut() {
            for (idx, tokens) in prefill_tokens_per_dp.iter().enumerate() {
                if idx < acc.len() {
                    acc[idx] += *tokens;
                }
            }
        }
        if let Some(acc) = self.report_decode_tokens_per_dp.as_mut() {
            for (idx, tokens) in decode_tokens_per_dp.iter().enumerate() {
                if idx < acc.len() {
                    acc[idx] += *tokens;
                }
            }
        }
        self.report_step_count += 1;
        self.report_sched_ms += schedule_ms;
        self.report_forward_ms += forward_ms;
        self.report_post_ms += postprocess_ms;
        self.report_fwd_tx_bytes += forward_tx_bytes;
        self.report_fwd_rx_bytes += forward_rx_bytes;
        self.report_transfer_ms += transfer_ms;
        self.report_wwi_ms += wwi_ms;
        self.report_immrecv_ms += immrecv_ms;
        self.report_net_ms += net_ms;
        self.report_serialize_ms += serialize_ms;

        if let Some(cached) = prefix_cached_tokens_per_dp {
            let prompt = prefix_prompt_tokens_per_dp.unwrap_or_else(|| vec![0; cached.len()]);
            if self.report_prefix_cached_tokens_per_dp.is_none() {
                self.report_prefix_cached_tokens_per_dp = Some(vec![0; cached.len()]);
                self.report_prefix_prompt_tokens_per_dp = Some(vec![0; prompt.len()]);
            }
            if let Some(acc) = self.report_prefix_cached_tokens_per_dp.as_mut() {
                for (idx, tokens) in cached.iter().enumerate() {
                    if idx < acc.len() {
                        acc[idx] += *tokens;
                    }
                }
            }
            if let Some(acc) = self.report_prefix_prompt_tokens_per_dp.as_mut() {
                for (idx, tokens) in prompt.iter().enumerate() {
                    if idx < acc.len() {
                        acc[idx] += *tokens;
                    }
                }
            }
            self.refresh_prefix_rates();
        }

        let now = now_seconds();
        let elapsed = now - self.last_report_time;
        if !force_report && elapsed < self.report_interval_s {
            return None;
        }
        let steps = self.report_step_count.max(1);
        let prefill = self
            .report_prefill_tokens_per_dp
            .clone()
            .unwrap_or_default();
        let decode = self.report_decode_tokens_per_dp.clone().unwrap_or_default();
        let cached = self.report_prefix_cached_tokens_per_dp.clone();
        let prompt = self.report_prefix_prompt_tokens_per_dp.clone();
        let message = self.log_engine_status(
            server_metric,
            engine_id,
            mode,
            running_per_dp,
            waiting,
            waiting_migration,
            used_blocks_per_dp,
            total_blocks,
            prefill,
            decode,
            elapsed,
            self.report_sched_ms / steps as f64,
            self.report_forward_ms / steps as f64,
            self.report_post_ms / steps as f64,
            self.report_step_count,
            self.report_fwd_tx_bytes,
            self.report_fwd_rx_bytes,
            self.report_transfer_ms / steps as f64,
            self.report_wwi_ms / steps as f64,
            self.report_immrecv_ms / steps as f64,
            self.report_net_ms / steps as f64,
            self.report_serialize_ms / steps as f64,
            cached,
            prompt,
            used_host_blocks_per_dp,
            total_host_blocks,
            used_hisparse_slots,
            total_hisparse_slots,
        );
        self.last_report_time = now;
        self.reset_window();
        Some(message)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        server_metric,
        engine_id,
        mode,
        running_per_dp,
        waiting,
        waiting_migration,
        used_blocks_per_dp,
        total_blocks,
        prefill_tokens_per_dp,
        decode_tokens_per_dp,
        elapsed,
        avg_schedule_ms=0.0,
        avg_forward_ms=0.0,
        avg_postprocess_ms=0.0,
        steps=0,
        fwd_tx_bytes=0,
        fwd_rx_bytes=0,
        avg_transfer_ms=0.0,
        avg_wwi_ms=0.0,
        avg_immrecv_ms=0.0,
        avg_net_ms=0.0,
        avg_serialize_ms=0.0,
        prefix_cached_tokens_per_dp=None,
        prefix_prompt_tokens_per_dp=None,
        used_host_blocks_per_dp=None,
        total_host_blocks=0,
        used_hisparse_slots=0,
        total_hisparse_slots=0
    ))]
    fn log_engine_status(
        &mut self,
        server_metric: &mut ServerMetric,
        engine_id: String,
        mode: String,
        running_per_dp: Vec<i32>,
        waiting: i32,
        waiting_migration: i32,
        used_blocks_per_dp: Option<Vec<i32>>,
        total_blocks: i32,
        prefill_tokens_per_dp: Vec<i32>,
        decode_tokens_per_dp: Vec<i32>,
        elapsed: f64,
        avg_schedule_ms: f64,
        avg_forward_ms: f64,
        avg_postprocess_ms: f64,
        steps: i32,
        fwd_tx_bytes: i64,
        fwd_rx_bytes: i64,
        avg_transfer_ms: f64,
        avg_wwi_ms: f64,
        avg_immrecv_ms: f64,
        avg_net_ms: f64,
        avg_serialize_ms: f64,
        prefix_cached_tokens_per_dp: Option<Vec<i32>>,
        prefix_prompt_tokens_per_dp: Option<Vec<i32>>,
        used_host_blocks_per_dp: Option<Vec<i32>>,
        total_host_blocks: i32,
        used_hisparse_slots: i32,
        total_hisparse_slots: i32,
    ) -> String {
        let prefill_tokens: i32 = prefill_tokens_per_dp.iter().sum();
        let decode_tokens: i32 = decode_tokens_per_dp.iter().sum();
        if prefill_tokens > 0 {
            server_metric.record_prefill_throughput(prefill_tokens as i64, elapsed);
        }
        if decode_tokens > 0 {
            server_metric.record_decode_throughput(decode_tokens as i64, elapsed);
        }

        self.engine_id = engine_id.clone();
        self.engine_mode = mode.clone();
        self.running_per_dp = running_per_dp.clone();
        self.used_blocks_per_dp = used_blocks_per_dp.clone().unwrap_or_default();
        self.total_blocks_per_dp = total_blocks;
        self.used_host_blocks_per_dp = used_host_blocks_per_dp.clone().unwrap_or_default();
        self.total_host_blocks_per_dp = total_host_blocks;
        self.used_hisparse_slots = used_hisparse_slots;
        self.total_hisparse_slots = total_hisparse_slots;
        self.last_schedule_ms = prom_value(avg_schedule_ms);
        self.last_forward_ms = prom_value(avg_forward_ms);
        self.last_postprocess_ms = prom_value(avg_postprocess_ms);
        self.last_step_count = steps;
        self.last_forward_tx_bytes = fwd_tx_bytes;
        self.last_forward_rx_bytes = fwd_rx_bytes;
        self.last_transfer_ms = prom_value(avg_transfer_ms);
        self.last_wwi_ms = prom_value(avg_wwi_ms);
        self.last_immrecv_ms = prom_value(avg_immrecv_ms);
        self.last_net_ms = prom_value(avg_net_ms);
        self.last_serialize_ms = prom_value(avg_serialize_ms);

        if let Some(prompt) = prefix_prompt_tokens_per_dp.as_ref() {
            let cached = prefix_cached_tokens_per_dp
                .as_ref()
                .cloned()
                .unwrap_or_else(|| vec![0; prompt.len()]);
            self.last_prefix_cache_hit_rate_per_dp = rates(&cached, prompt);
            let total_prompt: i32 = prompt.iter().sum();
            if total_prompt > 0 {
                self.last_prefix_cache_hit_rate =
                    cached.iter().sum::<i32>() as f64 / total_prompt as f64;
            }
        }

        let used_str = used_blocks_per_dp
            .as_ref()
            .map(|v| {
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .unwrap_or_else(|| "?".to_string());
        let kv = format!("{used_str}/{total_blocks}");
        let host_str = used_host_blocks_per_dp
            .as_ref()
            .map(|v| {
                v.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join("|")
            })
            .unwrap_or_else(|| "?".to_string());
        let host_kv = format!("{host_str}/{total_host_blocks}");
        let hisparse_str = if total_hisparse_slots > 0 {
            format!(" hisparse_hot={used_hisparse_slots}/{total_hisparse_slots} slot")
        } else {
            String::new()
        };
        let run_str = running_per_dp
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join("|");
        let pf_str = if elapsed > 0.0 {
            prefill_tokens_per_dp
                .iter()
                .map(|t| format!("{:.0}", *t as f64 / elapsed))
                .collect::<Vec<_>>()
                .join("|")
        } else {
            prefill_tokens_per_dp
                .iter()
                .map(|_| "0".to_string())
                .collect::<Vec<_>>()
                .join("|")
        };
        let dec_str = if elapsed > 0.0 {
            decode_tokens_per_dp
                .iter()
                .map(|t| format!("{:.0}", *t as f64 / elapsed))
                .collect::<Vec<_>>()
                .join("|")
        } else {
            decode_tokens_per_dp
                .iter()
                .map(|_| "0".to_string())
                .collect::<Vec<_>>()
                .join("|")
        };
        let mut cache_str = String::new();
        if let Some(prompt) = prefix_prompt_tokens_per_dp.as_ref() {
            let total_prompt: i32 = prompt.iter().sum();
            if total_prompt > 0 {
                let cached = prefix_cached_tokens_per_dp
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| vec![0; prompt.len()]);
                let ratio_str = cached
                    .iter()
                    .zip(prompt.iter())
                    .map(|(c, p)| {
                        let miss = (*p - *c).max(0);
                        format!("{c}/{miss}")
                    })
                    .collect::<Vec<_>>()
                    .join("|");
                let total_cached = cached.iter().sum::<i32>();
                let total_miss = (total_prompt - total_cached).max(0);
                cache_str = format!(
                    " | prefix-cache hit/miss {ratio_str} ({total_cached}/{total_miss} all)"
                );
            }
        }
        let total_ms = avg_schedule_ms + avg_forward_ms + avg_postprocess_ms;
        let lat_str = format!(
            " | lat sch={avg_schedule_ms:.2} fwd={avg_forward_ms:.2} post={avg_postprocess_ms:.2} total={total_ms:.2} ms/step n={steps}"
        );
        let mut xfer_str = String::new();
        if fwd_tx_bytes > 0 {
            let n = steps.max(1) as f64;
            xfer_str = format!(
                " | xfer fwd_tx={} fwd_rx={} (tx {}/step)",
                human_bytes(fwd_tx_bytes as f64),
                human_bytes(fwd_rx_bytes as f64),
                human_bytes(fwd_tx_bytes as f64 / n)
            );
            if avg_serialize_ms != 0.0 {
                xfer_str.push_str(&format!(" serialize={avg_serialize_ms:.2} ms/step"));
            }
            if avg_transfer_ms != 0.0 {
                xfer_str.push_str(&format!(" transfer={avg_transfer_ms:.2} ms/step"));
            }
            if avg_wwi_ms != 0.0 || avg_immrecv_ms != 0.0 {
                xfer_str.push_str(&format!(
                    " [wwi={avg_wwi_ms:.2} immrecv={avg_immrecv_ms:.2} ms/step]"
                ));
            }
            xfer_str.push_str(&format!(" net={avg_net_ms:.2} ms/step"));
        }
        format!(
            "[engine {} {mode}] run={run_str} wait={waiting} mig={waiting_migration} | kv={kv} blk host={host_kv} blk{hisparse_str} | tput pf={pf_str} dec={dec_str} tok/s ({elapsed:.0}s) | done={} tok={}p/{}g{cache_str}{lat_str}{xfer_str}",
            engine_id.chars().take(8).collect::<String>(),
            server_metric.num_completed_requests,
            server_metric.total_prompt_tokens,
            server_metric.total_generated_tokens
        )
    }

    pub(crate) fn to_prometheus(&self, server_metric: &ServerMetric) -> String {
        let mut lines = vec![
            "# HELP dlengine_up DLEngine metrics exporter health.".to_string(),
            "# TYPE dlengine_up gauge".to_string(),
            "dlengine_up 1".to_string(),
            "# HELP dlengine_uptime_seconds DLEngine process uptime.".to_string(),
            "# TYPE dlengine_uptime_seconds gauge".to_string(),
            format!("dlengine_uptime_seconds {}", prom_value(server_metric.uptime())),
            "# HELP dlengine_requests Number of requests by state.".to_string(),
            "# TYPE dlengine_requests gauge".to_string(),
            format!(
                "dlengine_requests{{state=\"running\"}} {}",
                server_metric.num_running_requests
            ),
            format!(
                "dlengine_requests{{state=\"waiting\"}} {}",
                server_metric.num_waiting_requests
            ),
            format!(
                "dlengine_requests{{state=\"waiting_migration\"}} {}",
                server_metric.num_waiting_migration_requests
            ),
            format!(
                "dlengine_requests{{state=\"completed\"}} {}",
                server_metric.num_completed_requests
            ),
            "# HELP dlengine_tokens Total token counters.".to_string(),
            "# TYPE dlengine_tokens counter".to_string(),
            format!(
                "dlengine_tokens{{type=\"prompt\"}} {}",
                server_metric.total_prompt_tokens
            ),
            format!(
                "dlengine_tokens{{type=\"generated\"}} {}",
                server_metric.total_generated_tokens
            ),
            format!("dlengine_tokens{{type=\"all\"}} {}", server_metric.total_tokens),
            "# HELP dlengine_throughput_tokens_per_second Current token throughput."
                .to_string(),
            "# TYPE dlengine_throughput_tokens_per_second gauge".to_string(),
            format!(
                "dlengine_throughput_tokens_per_second{{phase=\"prefill\"}} {}",
                prom_value(server_metric.current_prefill_throughput())
            ),
            format!(
                "dlengine_throughput_tokens_per_second{{phase=\"decode\"}} {}",
                prom_value(server_metric.current_decode_throughput())
            ),
            "# HELP dlengine_avg_throughput_tokens_per_second Average token throughput."
                .to_string(),
            "# TYPE dlengine_avg_throughput_tokens_per_second gauge".to_string(),
            format!(
                "dlengine_avg_throughput_tokens_per_second{{phase=\"prefill\"}} {}",
                prom_value(server_metric.avg_prefill_throughput())
            ),
            format!(
                "dlengine_avg_throughput_tokens_per_second{{phase=\"decode\"}} {}",
                prom_value(server_metric.avg_decode_throughput())
            ),
            "# HELP dlengine_waiting_blocks Waiting queue KV block demand.".to_string(),
            "# TYPE dlengine_waiting_blocks gauge".to_string(),
            format!(
                "dlengine_waiting_blocks{{type=\"head\"}} {}",
                server_metric.num_waiting_head_blocks
            ),
            format!(
                "dlengine_waiting_blocks{{type=\"total\"}} {}",
                server_metric.num_waiting_total_blocks
            ),
            "# HELP dlengine_step_latency_ms Last reported average step latency.".to_string(),
            "# TYPE dlengine_step_latency_ms gauge".to_string(),
            format!(
                "dlengine_step_latency_ms{{phase=\"schedule\"}} {}",
                self.last_schedule_ms
            ),
            format!(
                "dlengine_step_latency_ms{{phase=\"forward\"}} {}",
                self.last_forward_ms
            ),
            format!(
                "dlengine_step_latency_ms{{phase=\"postprocess\"}} {}",
                self.last_postprocess_ms
            ),
            "# HELP dlengine_step_count Last reported heartbeat window step count.".to_string(),
            "# TYPE dlengine_step_count gauge".to_string(),
            format!("dlengine_step_count {}", self.last_step_count),
            "# HELP dlengine_forward_bytes Last reported forward transfer bytes.".to_string(),
            "# TYPE dlengine_forward_bytes gauge".to_string(),
            format!(
                "dlengine_forward_bytes{{direction=\"tx\"}} {}",
                self.last_forward_tx_bytes
            ),
            format!(
                "dlengine_forward_bytes{{direction=\"rx\"}} {}",
                self.last_forward_rx_bytes
            ),
            "# HELP dlengine_transfer_latency_ms Last reported transfer latency.".to_string(),
            "# TYPE dlengine_transfer_latency_ms gauge".to_string(),
            format!(
                "dlengine_transfer_latency_ms{{phase=\"transfer\"}} {}",
                self.last_transfer_ms
            ),
            format!(
                "dlengine_transfer_latency_ms{{phase=\"write_with_imm\"}} {}",
                self.last_wwi_ms
            ),
            format!(
                "dlengine_transfer_latency_ms{{phase=\"imm_recv\"}} {}",
                self.last_immrecv_ms
            ),
            format!(
                "dlengine_transfer_latency_ms{{phase=\"network\"}} {}",
                self.last_net_ms
            ),
            format!(
                "dlengine_transfer_latency_ms{{phase=\"serialize\"}} {}",
                self.last_serialize_ms
            ),
            "# HELP dlengine_prefix_cache_hit_rate Prefix-cache hit rate of admitted prefills (overall).".to_string(),
            "# TYPE dlengine_prefix_cache_hit_rate gauge".to_string(),
            format!(
                "dlengine_prefix_cache_hit_rate {}",
                prom_value(self.last_prefix_cache_hit_rate)
            ),
            "# HELP dlengine_ttft_seconds Time to first token, per completed request."
                .to_string(),
            "# TYPE dlengine_ttft_seconds histogram".to_string(),
        ];
        for (bucket, count) in TTFT_BUCKETS.iter().zip(self.ttft_bucket_counts.iter()) {
            lines.push(format!(
                "dlengine_ttft_seconds_bucket{{le=\"{bucket}\"}} {count}"
            ));
        }
        lines.extend([
            format!(
                "dlengine_ttft_seconds_bucket{{le=\"+Inf\"}} {}",
                self.ttft_count
            ),
            format!("dlengine_ttft_seconds_sum {}", self.ttft_sum_s),
            format!("dlengine_ttft_seconds_count {}", self.ttft_count),
            "# HELP dlengine_ttft_seconds_avg Average time to first token.".to_string(),
            "# TYPE dlengine_ttft_seconds_avg gauge".to_string(),
            format!(
                "dlengine_ttft_seconds_avg {}",
                if self.ttft_count > 0 {
                    self.ttft_sum_s / self.ttft_count as f64
                } else {
                    0.0
                }
            ),
            "# HELP dlengine_ttft_ms_last Most recent time to first token (ms).".to_string(),
            "# TYPE dlengine_ttft_ms_last gauge".to_string(),
            format!("dlengine_ttft_ms_last {}", self.last_ttft_ms),
            "# HELP dlengine_tpot_seconds Time per output token (excl. queueing).".to_string(),
            "# TYPE dlengine_tpot_seconds histogram".to_string(),
        ]);
        for (bucket, count) in TPOT_BUCKETS.iter().zip(self.tpot_bucket_counts.iter()) {
            lines.push(format!(
                "dlengine_tpot_seconds_bucket{{le=\"{bucket}\"}} {count}"
            ));
        }
        lines.extend([
            format!(
                "dlengine_tpot_seconds_bucket{{le=\"+Inf\"}} {}",
                self.tpot_count
            ),
            format!("dlengine_tpot_seconds_sum {}", self.tpot_sum_s),
            format!("dlengine_tpot_seconds_count {}", self.tpot_count),
            "# HELP dlengine_tpot_seconds_avg Average time per output token.".to_string(),
            "# TYPE dlengine_tpot_seconds_avg gauge".to_string(),
            format!(
                "dlengine_tpot_seconds_avg {}",
                if self.tpot_count > 0 {
                    self.tpot_sum_s / self.tpot_count as f64
                } else {
                    0.0
                }
            ),
            "# HELP dlengine_tpot_ms_last Most recent time per output token (ms).".to_string(),
            "# TYPE dlengine_tpot_ms_last gauge".to_string(),
            format!("dlengine_tpot_ms_last {}", self.last_tpot_ms),
        ]);
        for (dp_idx, running) in self.running_per_dp.iter().enumerate() {
            if dp_idx == 0 {
                lines.push(
                    "# HELP dlengine_running_requests_per_dp Running requests per DP rank."
                        .to_string(),
                );
                lines.push("# TYPE dlengine_running_requests_per_dp gauge".to_string());
            }
            lines.push(format!(
                "dlengine_running_requests_per_dp{{dp=\"{dp_idx}\"}} {running}"
            ));
        }
        for (dp_idx, used) in self.used_blocks_per_dp.iter().enumerate() {
            if dp_idx == 0 {
                lines.push("# HELP dlengine_kv_blocks KV cache blocks per DP rank.".to_string());
                lines.push("# TYPE dlengine_kv_blocks gauge".to_string());
            }
            lines.push(format!(
                "dlengine_kv_blocks{{dp=\"{dp_idx}\",state=\"used\"}} {used}"
            ));
            lines.push(format!(
                "dlengine_kv_blocks{{dp=\"{dp_idx}\",state=\"total\"}} {}",
                self.total_blocks_per_dp
            ));
        }
        for (dp_idx, rate) in self.last_prefix_cache_hit_rate_per_dp.iter().enumerate() {
            if dp_idx == 0 {
                lines.push(
                    "# HELP dlengine_prefix_cache_hit_rate_per_dp Prefix-cache hit rate per DP rank."
                        .to_string(),
                );
                lines.push("# TYPE dlengine_prefix_cache_hit_rate_per_dp gauge".to_string());
            }
            lines.push(format!(
                "dlengine_prefix_cache_hit_rate_per_dp{{dp=\"{dp_idx}\"}} {}",
                prom_value(*rate)
            ));
        }
        lines.join("\n") + "\n"
    }
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        runtime_metrics_with_report_interval(5.0)
    }
}

fn runtime_metrics_with_report_interval(report_interval_s: f64) -> RuntimeMetrics {
    RuntimeMetrics {
        engine_id: String::new(),
        engine_mode: String::new(),
        running_per_dp: Vec::new(),
        used_blocks_per_dp: Vec::new(),
        total_blocks_per_dp: 0,
        used_host_blocks_per_dp: Vec::new(),
        total_host_blocks_per_dp: 0,
        used_hisparse_slots: 0,
        total_hisparse_slots: 0,
        last_schedule_ms: 0.0,
        last_forward_ms: 0.0,
        last_postprocess_ms: 0.0,
        last_step_count: 0,
        last_forward_tx_bytes: 0,
        last_forward_rx_bytes: 0,
        last_transfer_ms: 0.0,
        last_wwi_ms: 0.0,
        last_immrecv_ms: 0.0,
        last_net_ms: 0.0,
        last_serialize_ms: 0.0,
        last_prefix_cache_hit_rate: 0.0,
        last_prefix_cache_hit_rate_per_dp: Vec::new(),
        last_ttft_ms: 0.0,
        last_tpot_ms: 0.0,
        ttft_sum_s: 0.0,
        ttft_count: 0,
        tpot_sum_s: 0.0,
        tpot_count: 0,
        ttft_bucket_counts: vec![0; TTFT_BUCKETS.len()],
        tpot_bucket_counts: vec![0; TPOT_BUCKETS.len()],
        report_interval_s,
        last_report_time: now_seconds(),
        report_prefill_tokens_per_dp: None,
        report_decode_tokens_per_dp: None,
        report_step_count: 0,
        report_sched_ms: 0.0,
        report_forward_ms: 0.0,
        report_post_ms: 0.0,
        report_fwd_tx_bytes: 0,
        report_fwd_rx_bytes: 0,
        report_transfer_ms: 0.0,
        report_wwi_ms: 0.0,
        report_immrecv_ms: 0.0,
        report_net_ms: 0.0,
        report_serialize_ms: 0.0,
        report_prefix_cached_tokens_per_dp: None,
        report_prefix_prompt_tokens_per_dp: None,
    }
}

impl RuntimeMetrics {
    fn reset_window(&mut self) {
        self.report_prefill_tokens_per_dp = None;
        self.report_decode_tokens_per_dp = None;
        self.report_step_count = 0;
        self.report_sched_ms = 0.0;
        self.report_forward_ms = 0.0;
        self.report_post_ms = 0.0;
        self.report_fwd_tx_bytes = 0;
        self.report_fwd_rx_bytes = 0;
        self.report_transfer_ms = 0.0;
        self.report_wwi_ms = 0.0;
        self.report_immrecv_ms = 0.0;
        self.report_net_ms = 0.0;
        self.report_serialize_ms = 0.0;
        self.report_prefix_cached_tokens_per_dp = None;
        self.report_prefix_prompt_tokens_per_dp = None;
    }

    fn refresh_prefix_rates(&mut self) {
        let Some(prompt) = self.report_prefix_prompt_tokens_per_dp.as_ref() else {
            return;
        };
        let cached = self
            .report_prefix_cached_tokens_per_dp
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![0; prompt.len()]);
        self.last_prefix_cache_hit_rate_per_dp = rates(&cached, prompt);
        let total_prompt: i32 = prompt.iter().sum();
        if total_prompt > 0 {
            self.last_prefix_cache_hit_rate =
                cached.iter().sum::<i32>() as f64 / total_prompt as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report_once(runtime: &mut RuntimeMetrics, prefill: Vec<i32>, decode: Vec<i32>) -> bool {
        let mut server_metric = ServerMetric::default();
        runtime
            .maybe_report_engine_status(
                &mut server_metric,
                "engine".to_string(),
                "hybrid".to_string(),
                vec![0; prefill.len().max(decode.len())],
                0,
                0,
                Some(vec![0; prefill.len().max(decode.len())]),
                10,
                prefill,
                decode,
                None,
                None,
                0.0,
                0.0,
                0.0,
                0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                None,
                0,
                0,
                0,
            )
            .is_some()
    }

    #[test]
    fn prefill_reports_immediately_decode_respects_interval() {
        let mut runtime = runtime_metrics_with_report_interval(60.0);

        assert!(!report_once(&mut runtime, vec![0], vec![1]));
        assert!(report_once(&mut runtime, vec![8], vec![0]));
        assert!(!report_once(&mut runtime, vec![0], vec![1]));
    }
}
