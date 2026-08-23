use super::Scheduler;
use crate::metrics::SequenceMetric;
use pyo3::prelude::*;
use std::collections::HashSet;

impl Scheduler {
    pub(super) fn preempt_impl(
        &mut self,
        _py: Python<'_>,
        dp_idx: usize,
        seq_id: u64,
    ) -> PyResult<()> {
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.status = 0;
            let total_tokens = seq.token_ids.len() as i32;
            seq.num_tokens = total_tokens;
            seq.num_checkpointed_tokens = total_tokens;
            seq.prefill_start_offset = 0;
        }
        let swapped_to_host = self.cache_prepare_host_swap_out(seq_id);
        if !swapped_to_host {
            self.release_seq(seq_id);
        }
        self.running[dp_idx].retain(|id| *id != seq_id);
        if swapped_to_host {
            self.waiting.push(seq_id);
        } else {
            self.waiting.insert(0, seq_id);
        }
        Ok(())
    }

    pub(super) fn preempt_one_for_allocation(
        &mut self,
        py: Python<'_>,
        dp_idx: usize,
        exclude_seq_id: u64,
    ) -> PyResult<bool> {
        let victim = self.running.get(dp_idx).and_then(|queue| {
            queue
                .iter()
                .copied()
                .filter(|seq_id| *seq_id != exclude_seq_id && !self.dummy_seq_ids.contains(seq_id))
                .min_by_key(|seq_id| {
                    self.seq_table
                        .get(seq_id)
                        .map(|seq| seq.last_scheduled_step)
                        .unwrap_or(0)
                })
        });
        let Some(victim) = victim else {
            return Ok(false);
        };
        self.preempt_impl(py, dp_idx, victim)?;
        Ok(true)
    }

    pub(super) fn complete_host_swap_outs_impl(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        self.cache_complete_host_swap_outs(tasks);
    }

    pub(super) fn complete_host_swap_ins_impl(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        self.cache_complete_host_swap_ins(tasks);
    }

    pub(super) fn postprocess_impl(
        &mut self,
        py: Python<'_>,
        dp_group_seq_ids: Vec<Vec<u64>>,
        dp_group_token_ids: Vec<Vec<Vec<i32>>>,
        dp_group_token_logprobs: Option<Vec<Vec<Vec<f32>>>>,
    ) {
        self.last_step_token_ids.clear();
        let mut scheduled_ids: Vec<HashSet<u64>> = (0..self.dp()).map(|_| HashSet::new()).collect();
        for (group_idx, seq_ids) in dp_group_seq_ids.iter().enumerate() {
            let dp_idx = group_idx / self.group();
            if dp_idx >= scheduled_ids.len() {
                continue;
            }
            for seq_id in seq_ids {
                scheduled_ids[dp_idx].insert(*seq_id);
            }
        }

        let mut next_running: Vec<Vec<u64>> = self
            .running
            .iter()
            .enumerate()
            .map(|(dp_idx, seqs)| {
                seqs.iter()
                    .copied()
                    .filter(|seq_id| !scheduled_ids[dp_idx].contains(seq_id))
                    .collect()
            })
            .collect();

        for (group_idx, seq_ids) in dp_group_seq_ids.iter().enumerate() {
            let dp_idx = group_idx / self.group();
            let Some(group_tokens) = dp_group_token_ids.get(group_idx) else {
                continue;
            };

            for (seq_idx, seq_id) in seq_ids.iter().copied().enumerate() {
                if self.dummy_seq_ids.contains(&seq_id) {
                    continue;
                }
                let Some(tokens) = group_tokens.get(seq_idx) else {
                    next_running[dp_idx].push(seq_id);
                    continue;
                };

                let (prefill_target, cur_tokens) = {
                    let Some(seq) = self.seq_table.get(&seq_id) else {
                        continue;
                    };
                    (
                        seq.num_prompt_tokens.max(seq.num_checkpointed_tokens),
                        seq.num_tokens,
                    )
                };

                if cur_tokens < prefill_target {
                    self.commit_hbm_blocks(seq_id, cur_tokens);
                    let metric = {
                        let Some(seq) = self.seq_table.get_mut(&seq_id) else {
                            continue;
                        };
                        seq.status = 4;
                        self.sequence_metrics.get(&seq_id).map(|m| m.clone_ref(py))
                    };
                    if let Some(metric) = metric {
                        metric.borrow_mut(py).record_prefill_chunk();
                    }
                    self.prefilling[dp_idx].push(seq_id);
                    continue;
                }

                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    if seq.status == 4 {
                        seq.status = 1;
                    }
                }
                if cur_tokens == prefill_target {
                    self.commit_hbm_blocks(seq_id, prefill_target);
                }

                let mut applied_tokens = Vec::with_capacity(tokens.len());
                let mut hit_terminal = false;
                for (token_offset, token) in tokens.iter().copied().enumerate() {
                    let can_append = self
                        .seq_table
                        .get(&seq_id)
                        .map(|seq| {
                            let generated = seq.num_tokens - seq.num_prompt_tokens;
                            generated < seq.sampling_params.max_tokens
                                && seq.num_tokens < self.config.max_model_len
                        })
                        .unwrap_or(false);
                    if !can_append {
                        hit_terminal = true;
                        break;
                    }
                    let logprob = dp_group_token_logprobs
                        .as_ref()
                        .and_then(|groups| groups.get(group_idx))
                        .and_then(|group| group.get(seq_idx))
                        .and_then(|values| values.get(token_offset))
                        .copied();
                    let (generated, metric) = {
                        let Some(seq) = self.seq_table.get_mut(&seq_id) else {
                            continue;
                        };
                        seq.token_ids.push(token);
                        seq.last_token = token;
                        if let Some(logprob) = logprob {
                            seq.completion_logprobs.push(logprob);
                        }
                        seq.num_tokens = seq.token_ids.len() as i32;
                        (
                            seq.num_tokens - seq.num_prompt_tokens,
                            self.sequence_metrics.get(&seq_id).map(|m| m.clone_ref(py)),
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
                    applied_tokens.push(token);

                    let terminal_after_append = self
                        .seq_table
                        .get(&seq_id)
                        .map(|seq| {
                            let generated = seq.num_tokens - seq.num_prompt_tokens;
                            (!seq.sampling_params.ignore_eos
                                && self.config.eos_ids.contains(&token))
                                || generated >= seq.sampling_params.max_tokens
                                || seq.num_tokens >= self.config.max_model_len
                        })
                        .unwrap_or(true);
                    if terminal_after_append {
                        hit_terminal = true;
                        break;
                    }
                }

                self.last_step_token_ids.insert(seq_id, applied_tokens);
                if !self.last_step_token_ids[&seq_id].is_empty() {
                    if let Some(metric) = self.sequence_metrics.get(&seq_id) {
                        metric.borrow_mut(py).record_output_step();
                    }
                }

                let (max_tokens, ignore_eos, num_tokens, generated, last_token) = {
                    let Some(seq) = self.seq_table.get(&seq_id) else {
                        continue;
                    };
                    (
                        seq.sampling_params.max_tokens,
                        seq.sampling_params.ignore_eos,
                        seq.num_tokens,
                        seq.num_tokens - seq.num_prompt_tokens,
                        seq.last_token,
                    )
                };
                let hit_eos = !ignore_eos && self.config.eos_ids.contains(&last_token);
                let hit_limit = generated >= max_tokens || num_tokens >= self.config.max_model_len;

                if hit_terminal || hit_eos || hit_limit {
                    if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                        seq.status = 2;
                    }
                    self.release_seq(seq_id);
                } else if self.config.mode == "prefill" {
                    if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                        seq.status = 3;
                    }
                    self.to_be_migrated.insert(seq_id, dp_idx);
                } else {
                    let _ = self.cache_ensure_blocks_for_seq(py, seq_id, false);
                    if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                        seq.status = 1;
                    }
                    next_running[dp_idx].push(seq_id);
                }
            }
        }

        self.running = next_running;
    }

    pub(super) fn prefix_cached_tokens_impl(&self, seq_id: u64) -> i32 {
        self.cache
            .prefix_cached_tokens_by_seq
            .get(&seq_id)
            .copied()
            .unwrap_or(0)
    }

    pub(super) fn clear_finished_metric_state_impl(&mut self, seq_id: u64) {
        self.cache.prefix_counted_seq_ids.remove(&seq_id);
        self.cache.prefix_cached_tokens_by_seq.remove(&seq_id);
    }

    pub(super) fn abort_impl(&mut self, seq_id: u64) -> bool {
        let found = self.seq_table.contains_key(&seq_id)
            && (self.cache.seq_assignment.contains_key(&seq_id)
                || self.to_be_migrated.contains_key(&seq_id)
                || self.cache.session_wait.contains_key(&seq_id)
                || self.waiting.contains(&seq_id)
                || self.waiting_migration.contains(&seq_id)
                || self.running.iter().any(|queue| queue.contains(&seq_id))
                || self.prefilling.iter().any(|queue| queue.contains(&seq_id)));
        if !found {
            return false;
        }

        self.release_seq(seq_id);
        self.waiting.retain(|id| *id != seq_id);
        self.waiting_migration.retain(|id| *id != seq_id);
        for queue in &mut self.running {
            queue.retain(|id| *id != seq_id);
        }
        for queue in &mut self.prefilling {
            queue.retain(|id| *id != seq_id);
        }
        self.to_be_migrated.remove(&seq_id);
        self.cache.session_wait.remove(&seq_id);
        self.clear_finished_metric_state_impl(seq_id);
        true
    }

    pub(super) fn abort_many_impl(&mut self, seq_ids: Vec<u64>) -> Vec<u64> {
        seq_ids
            .into_iter()
            .filter(|seq_id| self.abort_impl(*seq_id))
            .collect()
    }

    pub(super) fn free_to_be_migrated_ids_impl(&mut self, _py: Python<'_>, seq_ids: Vec<u64>) {
        for seq_id in seq_ids {
            if self.to_be_migrated.remove(&seq_id).is_some() {
                self.release_seq(seq_id);
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.status = 2;
                }
            } else {
                self.release_seq(seq_id);
            }
        }
    }

    pub(super) fn set_sequence_metric_impl(
        &mut self,
        _py: Python<'_>,
        seq_id: u64,
        metric: Py<SequenceMetric>,
    ) -> bool {
        if !self.seq_table.contains_key(&seq_id) {
            return false;
        }
        self.sequence_metrics.insert(seq_id, metric);
        true
    }

    fn commit_hbm_blocks(&mut self, seq_id: u64, committed_tokens: i32) {
        let (dp_idx, group_id, token_ids) = {
            let Some(seq) = self.seq_table.get(&seq_id) else {
                return;
            };
            (
                seq.active_dp_idx.max(0) as usize,
                seq.active_group_id.max(0) as usize,
                seq.token_ids.clone(),
            )
        };
        if dp_idx >= self.dp() || group_id >= self.group() {
            return;
        }
        let flat = self.flat_idx(dp_idx, group_id);
        self.cache.hbm_pools[flat].commit_ready(seq_id, &token_ids, committed_tokens);
    }
}
