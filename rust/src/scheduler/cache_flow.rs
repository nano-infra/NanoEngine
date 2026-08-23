use super::{Scheduler, HOST_PREFIX_CACHE_SEQ_ID, HOST_SWAP_IN_COOLDOWN_STEPS};
use crate::cache::table::block::{compute_block_hash, EvictedBlock};
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn cache_seq_has_host_blocks(&self, seq_id: u64) -> bool {
        self.seq_table
            .get(&seq_id)
            .map(|seq| !seq.host_block_table.is_empty())
            .unwrap_or(false)
    }

    pub(super) fn cache_try_restore_host_blocks(
        &mut self,
        seq_id: u64,
        dp_idx: usize,
    ) -> PyResult<bool> {
        let (original_dp, group_id, host_blocks, last_swapped_out_step) = {
            let Some(seq) = self.seq_table.get(&seq_id) else {
                return Ok(false);
            };
            let original_dp = seq.active_dp_idx.max(0) as usize;
            let group_id = (seq.active_group_id.max(0) as usize).min(self.group() - 1);
            let host_blocks = seq
                .host_block_tables
                .get(&(group_id as i32))
                .cloned()
                .filter(|blocks| !blocks.is_empty())
                .unwrap_or_else(|| seq.host_block_table.clone());
            (
                original_dp,
                group_id,
                host_blocks,
                seq.last_swapped_out_step,
            )
        };
        if host_blocks.is_empty() || dp_idx != original_dp {
            return Ok(false);
        }
        if self.current_step <= last_swapped_out_step.saturating_add(HOST_SWAP_IN_COOLDOWN_STEPS) {
            return Ok(false);
        }

        let flat = self.flat_idx(dp_idx, group_id);
        if self.cache.hbm_num_free_blocks(flat) < host_blocks.len() {
            return Ok(false);
        }
        let tokens = (host_blocks.len() as i32) * self.config.kvcache_block_size.max(1);
        let gpu_blocks = self.cache
            .ensure_hbm_blocks(flat, seq_id, &[], tokens, false)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks while restoring host KV: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id}"
                ))
            })?;
        self.cache.assign_seq(seq_id, dp_idx, group_id);
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_dp_idx = dp_idx as i32;
            seq.active_group_id = group_id as i32;
            seq.active_block_table = gpu_blocks.clone();
            seq.active_block_tables
                .insert(group_id as i32, gpu_blocks.clone());
            seq.status = 1;
        }
        self.cache
            .push_pending_swap_in(flat, seq_id, host_blocks, gpu_blocks);
        Ok(true)
    }

    pub(super) fn cache_prepare_host_swap_out(&mut self, seq_id: u64) -> bool {
        if self.config.num_host_kvcache_blocks <= 0 || self.cache.has_pending_host_swap(seq_id) {
            return false;
        }
        let Some((dp_idx, group_id)) = self.cache.assignment(seq_id) else {
            return false;
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let gpu_blocks = self
            .seq_table
            .get(&seq_id)
            .map(|seq| seq.active_block_table.clone())
            .unwrap_or_default();
        if gpu_blocks.is_empty() {
            return false;
        }
        let tokens = (gpu_blocks.len() as i32) * self.config.kvcache_block_size.max(1);
        let Ok(host_blocks) = self
            .cache
            .ensure_host_blocks(flat, seq_id, &[], tokens, false)
        else {
            return false;
        };
        self.cache
            .insert_pending_host_swap(seq_id, dp_idx, group_id);
        self.cache
            .push_pending_swap_out(flat, seq_id, gpu_blocks, host_blocks.clone());
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.host_block_table = host_blocks.clone();
            seq.host_block_tables.insert(group_id as i32, host_blocks);
            seq.last_swapped_out_step = self.current_step;
        }
        true
    }

    pub(super) fn cache_complete_host_swap_outs(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        for group_tasks in tasks {
            for (seq_id, _gpu_blocks, _host_blocks) in group_tasks {
                let Some(pending) = self.cache.take_pending_host_swap(seq_id) else {
                    continue;
                };
                let flat = self.flat_idx(pending.dp_idx, pending.group_id);
                self.cache.remove_hbm_seq(flat, seq_id);
                self.cache.remove_assignment(seq_id);
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.active_block_table.clear();
                    seq.active_block_tables.clear();
                }
            }
        }
        self.cache.clear_pending_swap_out_tasks();
    }

    pub(super) fn cache_complete_host_swap_ins(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        for (flat, group_tasks) in tasks.into_iter().enumerate() {
            for (seq_id, _host_blocks, _gpu_blocks) in group_tasks {
                self.cache.remove_host_seq(flat, seq_id);
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.host_block_table.clear();
                    seq.host_block_tables.clear();
                }
            }
        }
        self.cache.clear_pending_swap_in_tasks();
    }

    pub(super) fn cache_write_back_evicted_prefix(
        &mut self,
        flat: usize,
        evicted: Option<EvictedBlock>,
    ) -> PyResult<()> {
        self.cache.write_back_evicted_prefix(
            flat,
            evicted,
            self.config.num_host_kvcache_blocks > 0,
            HOST_PREFIX_CACHE_SEQ_ID,
        )
    }

    pub(super) fn cache_cached_tokens_for_prefix(
        &mut self,
        flat: usize,
        token_ids: &[i32],
        tokens: i32,
    ) -> i32 {
        let cached = self.cache.cached_tokens_for_prefix(
            flat,
            token_ids,
            tokens,
            self.config.kvcache_block_size,
            self.blocks_needed_for_tokens(tokens),
        );
        let gqa_hisparse = self.config.cache_plan.flags & (1 << 0) != 0
            && self.config.cache_plan.flags & (1 << 6) != 0;
        if gqa_hisparse {
            let tail = self.config.cache_plan.hisparse.swap_in_block_size.max(1);
            cached.min((tokens - tail).max(0))
        } else {
            cached
        }
    }

    pub(super) fn cache_ensure_group_blocks(
        &mut self,
        _py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        tokens: i32,
        set_migrate: bool,
    ) -> PyResult<Vec<i32>> {
        // Hybrid prefill allocates through this group-level path rather than
        // cache_ensure_blocks_for_seq. The predictor seeds recurrent drafts
        // before postprocess, so its lookahead pages must be present in the
        // first RunnerIn block table.
        let allocation_tokens = if set_migrate && self.config.mode != "decode" {
            tokens + self.config.num_speculative_tokens.max(0)
        } else {
            tokens
        };
        let needed_blocks = self.blocks_needed_for_tokens(allocation_tokens);
        let flat = self.flat_idx(dp_idx, group_id);
        let use_prefix_cache = set_migrate
            && self.group() == 1
            && self.config.mode != "decode"
            && self.cache.prefix_caching_enabled;
        if use_prefix_cache && !self.cache.hbm_has_seq(flat, seq_id) {
            return self.cache_ensure_group_blocks_with_host_prefix(
                seq_id,
                dp_idx,
                group_id,
                allocation_tokens,
                set_migrate,
            );
        }
        let token_ids = if use_prefix_cache {
            self.seq_table
                .get(&seq_id)
                .map(|seq| seq.token_ids.as_slice())
                .unwrap_or(&[])
        } else {
            &[]
        };
        let blocks = self
            .cache
            .ensure_hbm_blocks(
                flat,
                seq_id,
                token_ids,
                allocation_tokens,
                use_prefix_cache,
            )
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
        })?;
        self.cache_attach_active_blocks(seq_id, dp_idx, group_id, &blocks);
        if set_migrate {
            self.cache_attach_migrate_blocks(seq_id, dp_idx, group_id, &blocks);
        }
        Ok(blocks)
    }

    pub(super) fn cache_ensure_blocks_for_seq(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        is_prefill: bool,
    ) -> PyResult<()> {
        let (seq_id, num_tokens) = {
            let Some(s) = self.seq_table.get(&seq_id) else {
                return Ok(());
            };
            (s.seq_id, s.num_tokens)
        };
        let speculative_lookahead = self.config.num_speculative_tokens.max(0);
        let needed_tokens = if is_prefill && self.config.mode != "decode" {
            // Recurrent MTP seeds its predictor cache during the target
            // prefill, before postprocess has a chance to allocate decode
            // blocks. Reserve the draft lookahead here as well, especially
            // when the prompt ends near a cache-page boundary.
            num_tokens + speculative_lookahead
        } else {
            // A normal decode writes one new cache position. Linear MTP first
            // verifies N drafts, then immediately builds the next N-draft
            // line. Reserve both windows before the accepted length is known.
            num_tokens + speculative_lookahead.saturating_mul(2).max(1)
        };
        let needed_blocks = self.blocks_needed_for_tokens(needed_tokens);
        if !is_prefill
            && self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4) | (1 << 6)) == 0
            && self.cache.compressed_pools_empty()
        {
            let has_blocks = self
                .seq_table
                .get(&seq_id)
                .map(|seq| seq.active_block_table.len() >= needed_blocks)
                .unwrap_or(false);
            if has_blocks {
                return Ok(());
            }
        }
        let (dp_idx, group_id) = match self.cache.assignment(seq_id) {
            Some(assignment) => assignment,
            None => {
                let assignment = self.choose_assignment();
                self.cache.assign_seq(seq_id, assignment.0, assignment.1);
                assignment
            }
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let use_prefix_cache = is_prefill
            && self.group() == 1
            && self.config.mode != "decode"
            && self.cache.prefix_caching_enabled;
        let token_ids = if use_prefix_cache {
            self.seq_table
                .get(&seq_id)
                .map(|seq| seq.token_ids.as_slice())
                .unwrap_or(&[])
        } else {
            &[]
        };
        let blocks = self.cache
            .ensure_hbm_blocks(flat, seq_id, token_ids, needed_tokens, use_prefix_cache)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
        })?;
        self.cache_attach_active_blocks(seq_id, dp_idx, group_id, &blocks);

        self.cache_ensure_state_slot(py, seq_id)?;
        self.cache_ensure_hisparse_slot(py, seq_id)?;
        self.cache_ensure_compressed_pages(py, seq_id, needed_tokens)?;

        if is_prefill && self.config.mode != "decode" {
            self.cache_attach_migrate_blocks(seq_id, dp_idx, group_id, &blocks);
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(());
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
        Ok(())
    }

    pub(super) fn cache_ensure_state_slot(&mut self, _py: Python<'_>, seq_id: u64) -> PyResult<()> {
        let needs_mtp_handoff =
            self.config.num_speculative_tokens > 1 && self.config.mode != "hybrid";
        if !needs_mtp_handoff
            && self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) == 0
        {
            return Ok(());
        }
        let Some(slot) = self.cache.ensure_state_slot(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of state slots: seq_id={seq_id}"
            )));
        };
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_state_slot = slot;
        }
        Ok(())
    }

    pub(super) fn cache_ensure_hisparse_slot(
        &mut self,
        _py: Python<'_>,
        seq_id: u64,
    ) -> PyResult<()> {
        if self.config.cache_plan.flags & (1 << 6) == 0 {
            return Ok(());
        }
        let Some(slot) = self.cache.ensure_hisparse_slot(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of HiSparse slots: seq_id={seq_id}"
            )));
        };
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_hisparse_slot = slot;
        }
        Ok(())
    }

    pub(super) fn cache_ensure_compressed_pages(
        &mut self,
        _py: Python<'_>,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<()> {
        if self.cache.compressed_pools_empty() {
            return Ok(());
        }
        let active_tables = self.cache.ensure_compressed_pages(seq_id, needed_tokens)?;
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_compressed_block_tables.clear();
            seq.active_compressed_block_tables = active_tables;
        }
        Ok(())
    }

    fn cache_ensure_group_blocks_with_host_prefix(
        &mut self,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        tokens: i32,
        set_migrate: bool,
    ) -> PyResult<Vec<i32>> {
        let flat = self.flat_idx(dp_idx, group_id);
        let token_ids = self
            .seq_table
            .get(&seq_id)
            .map(|seq| seq.token_ids.clone())
            .unwrap_or_default();
        let needed = self.blocks_needed_for_tokens(tokens);
        let mut blocks = Vec::with_capacity(needed);
        let mut hash = -1i64;
        let mut cache_miss = false;
        for block_idx in 0..needed {
            let view = self.cache_block_token_view(&token_ids, block_idx);
            let full = view.len() == self.config.kvcache_block_size.max(1) as usize;
            if full && !cache_miss {
                hash = compute_block_hash(view, hash);
                if let Some(block_id) = self.cache.lookup_hbm_ready_prefix(flat, hash, view) {
                    self.cache.retain_hbm_ready_prefix(flat, block_id)?;
                    blocks.push(block_id);
                    continue;
                }
                if let Some(host_block) = self.cache.lookup_host_ready_prefix(flat, hash, view) {
                    let (gpu_block, evicted) =
                        self.cache.allocate_hbm_promoted_ready(flat, hash, view)?;
                    self.cache_write_back_evicted_prefix(flat, evicted)?;
                    self.cache.push_pending_swap_in(
                        flat,
                        seq_id,
                        vec![host_block],
                        vec![gpu_block],
                    );
                    blocks.push(gpu_block);
                    continue;
                }
            }
            cache_miss = true;
            let (block_id, evicted) = self.cache.allocate_hbm_pending_fresh(flat)?;
            self.cache_write_back_evicted_prefix(flat, evicted)?;
            blocks.push(block_id);
        }
        self.cache
            .insert_hbm_seq_blocks(flat, seq_id, blocks.clone());
        self.cache_attach_active_blocks(seq_id, dp_idx, group_id, &blocks);
        if set_migrate {
            self.cache_attach_migrate_blocks(seq_id, dp_idx, group_id, &blocks);
        }
        Ok(blocks)
    }

    fn cache_attach_active_blocks(
        &mut self,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        blocks: &[i32],
    ) {
        let Some(s) = self.seq_table.get_mut(&seq_id) else {
            return;
        };
        let group_id = group_id as i32;
        s.active_group_id = group_id.max(0);
        s.active_block_table = blocks.to_vec();
        s.active_block_tables.insert(group_id, blocks.to_vec());
        s.active_dp_idx = dp_idx as i32;
    }

    fn cache_attach_migrate_blocks(
        &mut self,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        blocks: &[i32],
    ) {
        let Some(s) = self.seq_table.get_mut(&seq_id) else {
            return;
        };
        let group_id = group_id as i32;
        s.migrate_group_id = group_id.max(0);
        s.migrate_block_table = blocks.to_vec();
        s.migrate_block_tables.insert(group_id, blocks.to_vec());
        s.migrate_engine_id = self.engine_id_.clone();
        s.migrate_num_kvcache_blocks = self.config.num_kvcache_blocks;
        s.migrate_group_size = self.config.group_size.max(1);
        s.migrate_dp_idx = dp_idx as i32;
    }

    fn cache_block_token_view<'a>(&self, token_ids: &'a [i32], block_idx: usize) -> &'a [i32] {
        let block_size = self.config.kvcache_block_size.max(1) as usize;
        let start = block_idx * block_size;
        let end = (start + block_size).min(token_ids.len());
        if start >= end {
            &[]
        } else {
            &token_ids[start..end]
        }
    }
}

impl Scheduler {
    pub(super) fn release_seq(&mut self, seq_id: u64) {
        let group = self.group();
        self.cache.release_seq(seq_id, group);
    }
}
