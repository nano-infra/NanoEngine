use super::{ScheduleResult, Scheduler};
use crate::sequence::Sequence;
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn prompt_target(&self, seq: &Sequence) -> i32 {
        seq.num_prompt_tokens.max(seq.num_checkpointed_tokens)
    }

    pub(super) fn schedule_prefill(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<Py<Sequence>>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<Py<Sequence>>> = (0..dp).map(|_| Vec::new()).collect();
        let mut num_seqs = vec![vec![0i32; group]; dp];
        let mut num_tokens = vec![vec![0i32; group]; dp];

        for dp_idx in 0..dp {
            let mut rest = Vec::new();
            let continuations = std::mem::take(&mut self.prefilling[dp_idx]);
            for seq in continuations {
                let (group_id, target, cur) = {
                    let s = seq.borrow(py);
                    (
                        (s.active_group_id.max(0) as usize).min(group - 1),
                        self.prompt_target(&s),
                        s.num_tokens,
                    )
                };
                let budget =
                    self.config.max_num_batched_tokens.max(1) - num_tokens[dp_idx][group_id];
                let new_tokens = (target - cur).min(budget).max(0);
                if new_tokens <= 0 {
                    rest.push(seq);
                    continue;
                }
                {
                    let mut s = seq.borrow_mut(py);
                    s.num_tokens = cur + new_tokens;
                    s.active_dispatched_tokens =
                        self.dispatch_for_master(group_id, cur + new_tokens);
                }
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
                    let group_id = (seq.borrow(py).active_group_id.max(0) as usize).min(group - 1);
                    num_seqs[dp_idx][group_id] += 1;
                    num_tokens[dp_idx][group_id] += new_tokens;
                    if use_migration_queue {
                        self.waiting_migration.remove(0);
                    } else {
                        self.waiting.remove(0);
                    }
                    self.running[dp_idx].push(seq.clone_ref(py));
                    scheduled[dp_idx].push(seq.clone_ref(py));
                    let affinity = seq.borrow(py).affinity_key;
                    if affinity != 0 {
                        self.session_affinity.insert(affinity, dp_idx);
                    }
                    let seq_id = seq.borrow(py).seq_id;
                    self.session_wait.remove(&seq_id);
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

    pub(super) fn schedule_decode(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<Py<Sequence>>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<Py<Sequence>>> = (0..dp).map(|_| Vec::new()).collect();
        for dp_idx in 0..dp {
            let mut skipped = Vec::new();
            let mut per_group = vec![0i32; group];
            let mut group_lens = vec![0i32; group];
            let queue = std::mem::take(&mut self.running[dp_idx]);
            for seq in queue {
                let group_id = (seq.borrow(py).active_group_id.max(0) as usize).min(group - 1);
                if per_group[group_id] >= self.config.max_num_seqs.max(1) {
                    skipped.push(seq);
                    continue;
                }
                if let Err(_e) = self.ensure_blocks_for_seq(py, &seq, false) {
                    self.preempt(py, dp_idx, seq)?;
                    continue;
                }
                per_group[group_id] += 1;
                group_lens[group_id] += seq.borrow(py).num_tokens;
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

    pub(super) fn make_schedule_result(
        &self,
        py: Python<'_>,
        dp_seqs: Vec<Vec<Py<Sequence>>>,
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
                        (seq.borrow(py).active_group_id.max(0) as usize).min(group - 1) == group_id
                    })
                    .map(|seq| seq.clone_ref(py))
                    .collect::<Vec<_>>();
                result.filtered_dp_group_seqs.push(filtered);
            }
        }

        for dp_idx in 0..dp {
            for seq in &dp_seqs[dp_idx] {
                let s = seq.borrow(py);
                let master = (s.active_group_id.max(0) as usize).min(group - 1);
                let tokens = s.active_dispatched_tokens.clone();
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
            let n = head.borrow(py).num_tokens;
            result.waiting_head_blocks = self.blocks_needed_for_tokens(n) as i32;
        }
        result.waiting_total_blocks = wait_queue
            .iter()
            .map(|seq| {
                let n = seq.borrow(py).num_tokens;
                self.blocks_needed_for_tokens(n) as i32
            })
            .sum();
        Ok(result)
    }

    fn try_allocate_prefill(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        dp_idx: usize,
        batch_seqs: &[i32],
        batch_tokens: &[i32],
    ) -> PyResult<Option<i32>> {
        let (seq_id, full_len, cached) = {
            let s = seq.borrow(py);
            if self.config.mode == "decode" {
                (s.seq_id, s.num_tokens, s.num_tokens)
            } else {
                (s.seq_id, self.prompt_target(&s), s.num_cached_tokens)
            }
        };
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
        let new_tokens = if self.config.mode == "decode" {
            full_len.max(1)
        } else {
            (full_len - cached).min(budget).max(0)
        };
        if new_tokens <= 0 {
            return Ok(None);
        }
        let chunk_end = cached + new_tokens;
        self.seq_assignment.insert(seq_id, (dp_idx, master));
        let dispatch = self.compute_dispatch(py, dp_idx, master, full_len);
        {
            let mut s = seq.borrow_mut(py);
            s.active_dp_idx = dp_idx as i32;
            s.active_group_id = master as i32;
            if self.config.mode != "decode" {
                s.migrate_group_id = master as i32;
            }
            if self.config.mode != "decode" {
                s.num_tokens = chunk_end;
            }
            s.status = 1;
            s.active_dispatched_tokens = dispatch.clone();
        }

        for (gid, count) in dispatch.iter().copied().enumerate() {
            if count > 0 {
                self.ensure_group_blocks(
                    py,
                    seq,
                    seq_id,
                    dp_idx,
                    gid,
                    count,
                    self.config.mode != "decode",
                )?;
            }
        }
        {
            let mut s = seq.borrow_mut(py);
            s.active_group_id = master as i32;
            if self.config.mode != "decode" {
                s.migrate_group_id = master as i32;
            }
        }
        self.ensure_state_slot(py, seq, seq_id)?;
        self.ensure_hisparse_slot(py, seq, seq_id)?;
        self.ensure_compressed_pages(py, seq, seq_id, full_len)?;
        if self.config.mode != "decode" {
            let mut s = seq.borrow_mut(py);
            if let Some(slot) = self.seq_state_slots.get(&seq_id).copied() {
                s.migrate_state_slot = slot;
            }
            if let Some(slot) = self.seq_hisparse_slots.get(&seq_id).copied() {
                s.migrate_hisparse_slot = slot;
            }
            for (ratio, pool) in &self.compressed_pools {
                if let Some(pages) = pool.seq_pages.get(&seq_id) {
                    s.migrate_compressed_block_tables
                        .insert(*ratio, pages.clone());
                }
            }
        }
        self.prefix_cached_tokens_by_seq.insert(seq_id, cached);
        Ok(Some(new_tokens))
    }
}
