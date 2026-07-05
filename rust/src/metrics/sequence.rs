use super::stats::percentile;
use crate::common::now_seconds;
use pyo3::prelude::*;

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
        sequence_metric_new(seq_id, num_prompt_tokens)
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
    pub(crate) fn ttft(&self) -> Option<f64> {
        Some((self.first_token_time? - self.arrival_time?) * 1000.0)
    }

    #[getter]
    pub(crate) fn e2e_latency(&self) -> Option<f64> {
        Some((self.completion_time? - self.arrival_time?) * 1000.0)
    }

    #[getter]
    pub(crate) fn avg_tpot_wo_queueing(&self) -> Option<f64> {
        if self.itl_samples.is_empty() {
            None
        } else {
            Some(self.itl_samples.iter().sum::<f64>() / self.itl_samples.len() as f64)
        }
    }

    #[getter]
    pub(crate) fn avg_tpot_with_queueing(&self) -> Option<f64> {
        let total = (self.completion_time? - self.arrival_time?) * 1000.0;
        if self.num_generated_tokens <= 0 {
            None
        } else {
            Some(total / self.num_generated_tokens as f64)
        }
    }

    #[getter]
    pub(crate) fn queueing_time_ms(&self) -> Option<f64> {
        Some((self.first_scheduled_time? - self.arrival_time?) * 1000.0)
    }

    #[getter]
    pub(crate) fn decode_queue_time_ms(&self) -> Option<f64> {
        Some((self.decode_scheduled_time? - self.decode_arrival_time?) * 1000.0)
    }

    #[getter]
    pub(crate) fn avg_itl(&self) -> Option<f64> {
        self.avg_tpot_wo_queueing()
    }

    #[getter]
    pub(crate) fn p50_itl(&self) -> Option<f64> {
        percentile(&self.itl_samples, 0.50)
    }

    #[getter]
    pub(crate) fn p99_itl(&self) -> Option<f64> {
        percentile(&self.itl_samples, 0.99)
    }

    #[getter]
    pub(crate) fn prefill_time_ms(&self) -> Option<f64> {
        if self.prefill_chunk_samples.is_empty() {
            None
        } else {
            Some(self.prefill_chunk_samples.iter().sum())
        }
    }

    pub(crate) fn metric_report(&self) -> String {
        fn ms(value: Option<f64>) -> String {
            value
                .map(|v| format!("{v:.2}ms"))
                .unwrap_or_else(|| "N/A".to_string())
        }
        format!(
            "SequenceMetric [{}...] - TTFT: {}, E2E: {}, Prompt Length: {}, Output Length: {}, Queueing Time: {}, Decode Queueing Time: {}, ITL Wo Queue: {}, ITL With Queue: {}",
            self.seq_id.to_string().chars().take(8).collect::<String>(),
            ms(self.ttft()),
            ms(self.e2e_latency()),
            self.num_prompt_tokens,
            self.num_generated_tokens,
            ms(self.queueing_time_ms()),
            ms(self.decode_queue_time_ms()),
            ms(self.avg_tpot_wo_queueing()),
            ms(self.avg_tpot_with_queueing()),
        )
    }

    fn report(&self) -> String {
        self.metric_report()
    }
}

pub(crate) fn sequence_metric_new(seq_id: u64, num_prompt_tokens: i32) -> SequenceMetric {
    SequenceMetric {
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
