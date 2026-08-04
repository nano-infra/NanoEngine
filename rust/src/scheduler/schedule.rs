use super::{ScheduleResult, Scheduler};
use pyo3::prelude::*;

impl Scheduler {
    fn mark_scheduled(&mut self, seq_id: u64) {
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.last_scheduled_step = self.current_step;
        }
    }

    pub(super) fn prompt_target(&self, seq: &crate::sequence::Sequence) -> i32 {
        seq.num_prompt_tokens.max(seq.num_checkpointed_tokens)
    }

    pub(super) fn schedule_prefill(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<u64>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<u64>> = (0..dp).map(|_| Vec::new()).collect();
        let mut num_seqs = vec![vec![0i32; group]; dp];
        let mut num_tokens = vec![vec![0i32; group]; dp];

        for dp_idx in 0..dp {
            let mut rest = Vec::new();
            let continuations = std::mem::take(&mut self.prefilling[dp_idx]);
            for seq_id in continuations {
                let (group_id, target, cur) = {
                    let Some(s) = self.seq_table.get(&seq_id) else {
                        continue;
                    };
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
                    rest.push(seq_id);
                    continue;
                }
                {
                    let dispatch = self.dispatch_for_master(group_id, cur + new_tokens);
                    let Some(s) = self.seq_table.get_mut(&seq_id) else {
                        continue;
                    };
                    s.prefill_start_offset = cur;
                    s.num_tokens = cur + new_tokens;
                    s.active_dispatched_tokens = dispatch;
                }
                num_seqs[dp_idx][group_id] += 1;
                num_tokens[dp_idx][group_id] += new_tokens;
                self.running[dp_idx].push(seq_id);
                scheduled[dp_idx].push(seq_id);
                self.mark_scheduled(seq_id);
            }
            self.prefilling[dp_idx] = rest;
        }

        let use_migration_queue = self.config.mode == "decode";
        loop {
            let seq_id = if use_migration_queue {
                self.waiting_migration.first().copied()
            } else {
                self.waiting.first().copied()
            };
            let Some(seq_id) = seq_id else {
                break;
            };
            let mut placed = false;
            for dp_idx in self.route_candidates(py, seq_id) {
                if self.cache_seq_has_host_blocks(seq_id) {
                    if self.cache_try_restore_host_blocks(seq_id, dp_idx)? {
                        if use_migration_queue {
                            self.waiting_migration.remove(0);
                        } else {
                            self.waiting.remove(0);
                        }
                        self.running[dp_idx].push(seq_id);
                        let affinity = self
                            .seq_table
                            .get(&seq_id)
                            .map(|s| s.affinity_key)
                            .unwrap_or(0);
                        if affinity != 0 {
                            self.cache.session_affinity.insert(affinity, dp_idx);
                        }
                        self.cache.session_wait.remove(&seq_id);
                        placed = true;
                        break;
                    }
                    continue;
                }
                if let Some(new_tokens) = self.try_allocate_prefill(
                    py,
                    seq_id,
                    dp_idx,
                    &num_seqs[dp_idx],
                    &num_tokens[dp_idx],
                )? {
                    let group_id = self
                        .seq_table
                        .get(&seq_id)
                        .map(|s| (s.active_group_id.max(0) as usize).min(group - 1))
                        .unwrap_or(0);
                    num_seqs[dp_idx][group_id] += 1;
                    num_tokens[dp_idx][group_id] += new_tokens;
                    if use_migration_queue {
                        self.waiting_migration.remove(0);
                    } else {
                        self.waiting.remove(0);
                    }
                    self.running[dp_idx].push(seq_id);
                    scheduled[dp_idx].push(seq_id);
                    self.mark_scheduled(seq_id);
                    let affinity = self
                        .seq_table
                        .get(&seq_id)
                        .map(|s| s.affinity_key)
                        .unwrap_or(0);
                    if affinity != 0 {
                        self.cache.session_affinity.insert(affinity, dp_idx);
                    }
                    self.cache.session_wait.remove(&seq_id);
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

    pub(super) fn schedule_decode(&mut self, py: Python<'_>) -> PyResult<Vec<Vec<u64>>> {
        let dp = self.dp();
        let group = self.group();
        let mut scheduled: Vec<Vec<u64>> = (0..dp).map(|_| Vec::new()).collect();
        for dp_idx in 0..dp {
            let mut skipped = Vec::new();
            let mut per_group = vec![0i32; group];
            let mut group_lens = vec![0i32; group];
            let queue = std::mem::take(&mut self.running[dp_idx]);
            for seq_id in queue {
                let group_id = self
                    .seq_table
                    .get(&seq_id)
                    .map(|s| (s.active_group_id.max(0) as usize).min(group - 1))
                    .unwrap_or(0);
                if per_group[group_id] >= self.config.max_num_seqs.max(1) {
                    skipped.push(seq_id);
                    continue;
                }
                if let Err(_e) = self.cache_ensure_blocks_for_seq(py, seq_id, false) {
                    self.preempt_impl(py, dp_idx, seq_id)?;
                    continue;
                }
                per_group[group_id] += 1;
                group_lens[group_id] += self
                    .seq_table
                    .get(&seq_id)
                    .map(|s| s.num_tokens)
                    .unwrap_or(0);
                scheduled[dp_idx].push(seq_id);
                self.mark_scheduled(seq_id);
                skipped.push(seq_id);
            }
            self.running[dp_idx] = skipped;
            for group_id in 0..group {
                if group_lens[group_id] == 0 && group > 1 {
                    scheduled[dp_idx].push(self.make_dummy_seq_id(dp_idx, group_id));
                }
            }
        }
        Ok(scheduled)
    }

    pub(super) fn make_schedule_result(
        &self,
        _py: Python<'_>,
        dp_seq_ids: Vec<Vec<u64>>,
        is_prefill: bool,
    ) -> PyResult<ScheduleResult> {
        let dp = self.dp();
        let group = self.group();
        let mut result = ScheduleResult::default();
        result.is_prefill = is_prefill;
        result.dp_seq_ids = dp_seq_ids.clone();
        result.swap_out_tasks = self.cache.pending_swap_out_tasks.clone();
        result.swap_in_tasks = self.cache.pending_swap_in_tasks.clone();
        result.dp_group_seq_ids = Vec::with_capacity(dp * group);
        result.filtered_dp_group_seq_ids = Vec::with_capacity(dp * group);
        result.group_send_counts = vec![vec![0; group]; dp];
        result.group_recv_counts = vec![vec![0; group]; dp];
        result.group_q_matrix = vec![vec![vec![0; group]; group]; dp];

        for dp_idx in 0..dp {
            for group_id in 0..group {
                result.dp_group_seq_ids.push(dp_seq_ids[dp_idx].clone());
                let filtered = dp_seq_ids[dp_idx]
                    .iter()
                    .filter(|seq_id| {
                        self.seq_table
                            .get(seq_id)
                            .map(|seq| {
                                (seq.active_group_id.max(0) as usize).min(group - 1) == group_id
                            })
                            .unwrap_or(false)
                    })
                    .copied()
                    .collect::<Vec<_>>();
                result.filtered_dp_group_seq_ids.push(filtered);
            }
        }

        for dp_idx in 0..dp {
            for seq in &dp_seq_ids[dp_idx] {
                let Some(s) = self.seq_table.get(seq) else {
                    continue;
                };
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
            let n = self.seq_table.get(head).map(|s| s.num_tokens).unwrap_or(0);
            result.waiting_head_blocks = self.blocks_needed_for_tokens(n) as i32;
        }
        result.waiting_total_blocks = wait_queue
            .iter()
            .map(|seq_id| {
                let n = self
                    .seq_table
                    .get(seq_id)
                    .map(|s| s.num_tokens)
                    .unwrap_or(0);
                self.blocks_needed_for_tokens(n) as i32
            })
            .sum();
        Ok(result)
    }

    fn try_allocate_prefill(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        batch_seqs: &[i32],
        batch_tokens: &[i32],
    ) -> PyResult<Option<i32>> {
        let (seq_id, full_len, mut cached) = {
            let Some(s) = self.seq_table.get(&seq_id) else {
                return Ok(None);
            };
            if self.config.mode == "decode" {
                (s.seq_id, s.num_tokens, s.num_tokens)
            } else {
                (s.seq_id, self.prompt_target(&s), s.num_cached_tokens)
            }
        };
        if self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) != 0
            && !self.cache.can_ensure_state_slot(seq_id)
        {
            // Linear-attention state is a per-active-sequence resource. When
            // all slots are occupied, retain this request at the head of the
            // waiting queue until a running sequence completes instead of
            // turning normal admission pressure into a fatal scheduler error.
            return Ok(None);
        }
        let Some(master) = self.choose_master_group(dp_idx, batch_seqs, batch_tokens) else {
            return Ok(None);
        };
        if self.config.mode != "decode" && self.group() == 1 && self.cache.prefix_caching_enabled {
            let token_ids = self
                .seq_table
                .get(&seq_id)
                .map(|s| s.token_ids.clone())
                .unwrap_or_default();
            let flat = self.flat_idx(dp_idx, master);
            cached = self.cache_cached_tokens_for_prefix(flat, &token_ids, full_len);
            if let Some(s) = self.seq_table.get_mut(&seq_id) {
                s.num_cached_tokens = cached;
                s.prefill_start_offset = cached;
            }
        }
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
        self.cache.assign_seq(seq_id, dp_idx, master);
        let dispatch = self.compute_dispatch(dp_idx, master, full_len);
        {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(None);
            };
            s.active_dp_idx = dp_idx as i32;
            s.active_group_id = master as i32;
            if self.config.mode != "decode" {
                s.migrate_group_id = master as i32;
            }
            if self.config.mode != "decode" {
                s.prefill_start_offset = cached;
                s.num_tokens = chunk_end;
            }
            s.status = 1;
            s.active_dispatched_tokens = dispatch.clone();
        }

        for (gid, count) in dispatch.iter().copied().enumerate() {
            if count > 0 {
                if let Err(err) = self.cache_ensure_group_blocks(
                    py,
                    seq_id,
                    dp_idx,
                    gid,
                    count,
                    self.config.mode != "decode",
                ) {
                    self.cache.remove_assignment(seq_id);
                    if let Some(s) = self.seq_table.get_mut(&seq_id) {
                        s.status = 0;
                        s.active_block_table.clear();
                        s.active_block_tables.clear();
                    }
                    if self.preempt_one_for_allocation(py, dp_idx, seq_id)? {
                        return Ok(None);
                    }
                    return Err(err);
                }
            }
        }
        {
            if let Some(s) = self.seq_table.get_mut(&seq_id) {
                s.active_group_id = master as i32;
                if self.config.mode != "decode" {
                    s.migrate_group_id = master as i32;
                }
            }
        }
        self.cache_ensure_state_slot(py, seq_id)?;
        self.cache_ensure_hisparse_slot(py, seq_id)?;
        self.cache_ensure_compressed_pages(py, seq_id, full_len)?;
        if self.config.mode != "decode" {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(None);
            };
            if let Some(slot) = self.cache.state_slot(seq_id) {
                s.migrate_state_slot = slot;
            }
            if let Some(slot) = self.cache.hisparse_slot(seq_id) {
                s.migrate_hisparse_slot = slot;
            }
            for (ratio, pages) in self.cache.compressed_pages(seq_id) {
                s.migrate_compressed_block_tables.insert(ratio, pages);
            }
        }
        self.cache.set_prefix_cached_tokens(seq_id, cached);
        Ok(Some(new_tokens))
    }
}
