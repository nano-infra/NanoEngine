use crate::proto::RunnerOut;

pub(crate) trait RunnerMetricSource {
    fn generated_token_count(&self) -> i64;
}

impl RunnerMetricSource for RunnerOut {
    fn generated_token_count(&self) -> i64 {
        self.token_ids
            .iter()
            .map(|seq_tokens| seq_tokens.len() as i64)
            .sum()
    }
}
