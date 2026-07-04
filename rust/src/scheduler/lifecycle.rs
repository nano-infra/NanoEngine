use super::Scheduler;
use crate::metrics::SequenceMetric;
use crate::sequence::Sequence;
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn preempt_impl(
        &mut self,
        py: Python<'_>,
        dp_idx: usize,
        seq: Py<Sequence>,
    ) -> PyResult<()> {
        let seq_id = {
            let mut s = seq.borrow_mut(py);
            let seq_id = s.seq_id;
            s.status = 0;
            let total_tokens = s.token_ids.len() as i32;
            s.num_tokens = total_tokens;
            s.num_checkpointed_tokens = total_tokens;
            seq_id
        };
        self.release_seq(seq_id);
        self.running[dp_idx].retain(|s| s.borrow(py).seq_id != seq_id);
        self.waiting.insert(0, seq);
        Ok(())
    }

    pub(super) fn postprocess_impl(
        &mut self,
        py: Python<'_>,
        dp_group_seqs: Vec<Vec<Py<Sequence>>>,
        dp_group_token_ids: Vec<Vec<Vec<i32>>>,
        dp_group_token_logprobs: Option<Vec<Vec<Vec<f32>>>>,
    ) {
        let eos_ids = self.config.eos_ids.clone();
        let mut next_running: Vec<Vec<Py<Sequence>>> = (0..self.dp()).map(|_| Vec::new()).collect();

        for (group_idx, seqs) in dp_group_seqs.iter().enumerate() {
            let dp_idx = group_idx / self.group();
            let Some(group_tokens) = dp_group_token_ids.get(group_idx) else {
                continue;
            };

            for (seq_idx, seq) in seqs.iter().enumerate() {
                let Some(tokens) = group_tokens.get(seq_idx) else {
                    next_running[dp_idx].push(seq.clone_ref(py));
                    continue;
                };

                let (seq_id, prefill_target, cur_tokens) = {
                    let s = seq.borrow(py);
                    (
                        s.seq_id,
                        s.num_prompt_tokens.max(s.num_checkpointed_tokens),
                        s.num_tokens,
                    )
                };

                if cur_tokens < prefill_target {
                    self.commit_hbm_blocks(py, seq, cur_tokens);
                    let metric = {
                        let mut s = seq.borrow_mut(py);
                        s.num_cached_tokens = cur_tokens;
                        s.status = 4;
                        s.metric.as_ref().map(|m| m.clone_ref(py))
                    };
                    if let Some(metric) = metric {
                        metric.borrow_mut(py).record_prefill_chunk();
                    }
                    self.prefilling[dp_idx].push(seq.clone_ref(py));
                    continue;
                }

                {
                    let mut s = seq.borrow_mut(py);
                    if s.status == 4 {
                        s.status = 1;
                    }
                }
                if cur_tokens >= prefill_target {
                    self.commit_hbm_blocks(py, seq, prefill_target);
                }

                for (token_offset, token) in tokens.iter().copied().enumerate() {
                    let logprob = dp_group_token_logprobs
                        .as_ref()
                        .and_then(|groups| groups.get(group_idx))
                        .and_then(|group| group.get(seq_idx))
                        .and_then(|values| values.get(token_offset))
                        .copied();
                    let (generated, metric) = {
                        let mut s = seq.borrow_mut(py);
                        s.token_ids.push(token);
                        s.last_token = token;
                        if let Some(logprob) = logprob {
                            s.completion_logprobs.push(logprob);
                        }
                        s.num_tokens = s.token_ids.len() as i32;
                        (
                            s.num_tokens - s.num_prompt_tokens,
                            s.metric.as_ref().map(|m| m.clone_ref(py)),
                        )
                    };
                    if let Some(metric) = metric {
                        let mut metric = metric.borrow_mut(py);
                        if generated <= 1 {
                            metric.record_first_token();
                        } else {
                            metric.record_token();
                        }
                    }
                }

                let (max_tokens, ignore_eos, num_tokens, generated, last_token) = {
                    let s = seq.borrow(py);
                    (
                        s.sampling_params.max_tokens,
                        s.sampling_params.ignore_eos,
                        s.num_tokens,
                        s.num_tokens - s.num_prompt_tokens,
                        s.last_token,
                    )
                };
                let hit_eos = !ignore_eos && eos_ids.contains(&last_token);
                let hit_limit = generated >= max_tokens || num_tokens >= self.config.max_model_len;

                if hit_eos || hit_limit {
                    seq.borrow_mut(py).status = 2;
                    self.park_or_release(py, seq);
                } else if self.config.mode == "prefill" {
                    seq.borrow_mut(py).status = 3;
                    self.to_be_migrated
                        .insert(seq_id, (seq.clone_ref(py), dp_idx));
                } else {
                    let _ = self.ensure_blocks_for_seq(py, seq, false);
                    seq.borrow_mut(py).status = 1;
                    next_running[dp_idx].push(seq.clone_ref(py));
                }
            }
        }

        self.running = next_running;
    }

    pub(super) fn prefix_cached_tokens_impl(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_by_seq
            .get(&seq_id)
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn clear_finished_metric_state_impl(&mut self, seq_id: u64) {
        self.prefix_counted_seq_ids.remove(&seq_id);
        self.prefix_cached_tokens_by_seq.remove(&seq_id);
    }

    pub(super) fn abort_impl(&mut self, seq_id: u64) -> bool {
        self.release_seq(seq_id);
        self.waiting
            .retain(|seq| Python::with_gil(|py| seq.borrow(py).seq_id != seq_id));
        self.waiting_migration
            .retain(|seq| Python::with_gil(|py| seq.borrow(py).seq_id != seq_id));
        for queue in &mut self.running {
            queue.retain(|seq| Python::with_gil(|py| seq.borrow(py).seq_id != seq_id));
        }
        for queue in &mut self.prefilling {
            queue.retain(|seq| Python::with_gil(|py| seq.borrow(py).seq_id != seq_id));
        }
        self.to_be_migrated.remove(&seq_id);
        self.session_wait.remove(&seq_id);
        true
    }

    pub(super) fn free_to_be_migrated_impl(
        &mut self,
        py: Python<'_>,
        seqs: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let seqs: Vec<Py<Sequence>> = seqs.extract()?;
        for seq in seqs {
            let seq_id = seq.borrow(py).seq_id;
            if let Some((seq, _dp_idx)) = self.to_be_migrated.remove(&seq_id) {
                self.release_seq(seq_id);
                seq.borrow_mut(py).status = 2;
            } else {
                self.release_seq(seq_id);
            }
        }
        Ok(())
    }

    pub(super) fn free_to_be_migrated_ids_impl(&mut self, py: Python<'_>, seq_ids: Vec<u64>) {
        for seq_id in seq_ids {
            if let Some((seq, _dp_idx)) = self.to_be_migrated.remove(&seq_id) {
                self.release_seq(seq_id);
                seq.borrow_mut(py).status = 2;
            } else {
                self.release_seq(seq_id);
            }
        }
    }

    pub(super) fn set_sequence_metric_impl(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        metric: Py<SequenceMetric>,
    ) -> bool {
        for queue in [&mut self.waiting, &mut self.waiting_migration] {
            for seq in queue.iter() {
                if seq.borrow(py).seq_id == seq_id {
                    seq.borrow_mut(py).metric = Some(metric.clone_ref(py));
                    return true;
                }
            }
        }
        for queue in self.running.iter_mut().chain(self.prefilling.iter_mut()) {
            for seq in queue.iter() {
                if seq.borrow(py).seq_id == seq_id {
                    seq.borrow_mut(py).metric = Some(metric.clone_ref(py));
                    return true;
                }
            }
        }
        if let Some((seq, _)) = self.to_be_migrated.get(&seq_id) {
            seq.borrow_mut(py).metric = Some(metric);
            return true;
        }
        false
    }

    fn commit_hbm_blocks(&mut self, py: Python<'_>, seq: &Py<Sequence>, committed_tokens: i32) {
        let (seq_id, dp_idx, group_id, token_ids) = {
            let s = seq.borrow(py);
            (
                s.seq_id,
                s.active_dp_idx.max(0) as usize,
                s.active_group_id.max(0) as usize,
                s.token_ids.clone(),
            )
        };
        if dp_idx >= self.dp() || group_id >= self.group() {
            return;
        }
        let flat = self.flat_idx(dp_idx, group_id);
        self.hbm_pools[flat].commit_ready(seq_id, &token_ids, committed_tokens);
    }
}
