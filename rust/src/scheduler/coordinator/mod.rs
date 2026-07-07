use super::{Scheduler, HOST_PREFIX_CACHE_SEQ_ID, HOST_SWAP_IN_COOLDOWN_STEPS};
use crate::cache::table::block::{compute_block_hash, CompressedPool, EvictedBlock};
use crate::cache::{CacheHit, PendingHostSwap};
use pyo3::prelude::*;

mod default;
mod hisparse;
mod maybe_hybrid;
mod snapshot;
mod strategy;

use default::DefaultCoordinator;
use hisparse::HisparseCoordinator;
use strategy::{CacheCoordinator, CoordinatorKind};

impl Scheduler {
    fn use_hisparse_cache_coordinator(&self) -> bool {
        self.config.cache_plan.flags & (1 << 6) != 0
    }

    fn active_cache_coordinator(&self) -> CoordinatorKind {
        if self.use_hisparse_cache_coordinator() {
            CoordinatorKind::Hisparse
        } else {
            CoordinatorKind::Default
        }
    }

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
        if self.cache.hbm_pools[flat].num_free_blocks() < host_blocks.len() {
            return Ok(false);
        }
        let tokens = (host_blocks.len() as i32) * self.config.kvcache_block_size.max(1);
        let gpu_blocks = self.cache.hbm_pools[flat]
            .ensure_blocks(seq_id, &[], tokens, false)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks while restoring host KV: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id}"
                ))
            })?;
        self.cache.seq_assignment.insert(seq_id, (dp_idx, group_id));
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_dp_idx = dp_idx as i32;
            seq.active_group_id = group_id as i32;
            seq.active_block_table = gpu_blocks.clone();
            seq.active_block_tables
                .insert(group_id as i32, gpu_blocks.clone());
            seq.status = 1;
        }
        if let Some(tasks) = self.cache.pending_swap_in_tasks.get_mut(flat) {
            tasks.push((seq_id, host_blocks, gpu_blocks));
        }
        Ok(true)
    }

    pub(super) fn cache_prepare_host_swap_out(&mut self, seq_id: u64) -> bool {
        if self.config.num_host_kvcache_blocks <= 0
            || self.cache.pending_host_swaps.contains_key(&seq_id)
        {
            return false;
        }
        let Some((dp_idx, group_id)) = self.cache.seq_assignment.get(&seq_id).copied() else {
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
        let Ok(host_blocks) = self.cache.host_pools[flat].ensure_blocks(seq_id, &[], tokens, false)
        else {
            return false;
        };
        self.cache
            .pending_host_swaps
            .insert(seq_id, PendingHostSwap { dp_idx, group_id });
        if let Some(tasks) = self.cache.pending_swap_out_tasks.get_mut(flat) {
            tasks.push((seq_id, gpu_blocks, host_blocks.clone()));
        }
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
                let Some(pending) = self.cache.pending_host_swaps.remove(&seq_id) else {
                    continue;
                };
                let flat = self.flat_idx(pending.dp_idx, pending.group_id);
                self.cache.hbm_pools[flat].remove_seq(seq_id);
                self.cache.seq_assignment.remove(&seq_id);
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.active_block_table.clear();
                    seq.active_block_tables.clear();
                }
            }
        }
        for tasks in &mut self.cache.pending_swap_out_tasks {
            tasks.clear();
        }
    }

    pub(super) fn cache_complete_host_swap_ins(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        for (flat, group_tasks) in tasks.into_iter().enumerate() {
            for (seq_id, _host_blocks, _gpu_blocks) in group_tasks {
                if let Some(pool) = self.cache.host_pools.get_mut(flat) {
                    pool.remove_seq(seq_id);
                }
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.host_block_table.clear();
                    seq.host_block_tables.clear();
                }
            }
        }
        for tasks in &mut self.cache.pending_swap_in_tasks {
            tasks.clear();
        }
    }

    pub(super) fn cache_write_back_evicted_prefix(
        &mut self,
        flat: usize,
        evicted: Option<EvictedBlock>,
    ) -> PyResult<()> {
        let Some(evicted) = evicted else {
            return Ok(());
        };
        if self.config.num_host_kvcache_blocks <= 0 {
            return Ok(());
        }
        let host_block = self.cache.host_pools[flat]
            .store_ready_cache_block(evicted.hash, &evicted.token_ids)?;
        self.cache.pending_swap_out_tasks[flat].push((
            HOST_PREFIX_CACHE_SEQ_ID,
            vec![evicted.block_id],
            vec![host_block],
        ));
        Ok(())
    }

    pub(super) fn cache_cached_tokens_for_prefix(
        &mut self,
        flat: usize,
        token_ids: &[i32],
        tokens: i32,
    ) -> i32 {
        let block_size = self.config.kvcache_block_size.max(1) as usize;
        let mut hash = -1i64;
        let mut matched = 0i32;
        for block_idx in 0..self.blocks_needed_for_tokens(tokens) {
            let view = self.cache_block_token_view(token_ids, block_idx);
            if view.len() != block_size {
                break;
            }
            hash = compute_block_hash(view, hash);
            let hbm_hit = self.cache.hbm_pools[flat]
                .lookup_ready_prefix_block(hash, view)
                .is_some();
            let host_hit = if hbm_hit {
                false
            } else {
                self.cache.host_pools[flat]
                    .lookup_ready_prefix_block(hash, view)
                    .is_some()
            };
            if !hbm_hit && !host_hit {
                break;
            }
            matched += 1;
        }
        let cached = (matched * self.config.kvcache_block_size.max(1)).min((tokens - 1).max(0));
        match self.active_cache_coordinator() {
            CoordinatorKind::Default => {
                DefaultCoordinator::adjust_prefix_cached_tokens(self, tokens, cached)
            }
            CoordinatorKind::Hisparse => {
                HisparseCoordinator::adjust_prefix_cached_tokens(self, tokens, cached)
            }
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
        let needed_blocks = self.blocks_needed_for_tokens(tokens);
        let flat = self.flat_idx(dp_idx, group_id);
        let use_prefix_cache = set_migrate
            && self.group() == 1
            && self.config.mode != "decode"
            && self.cache.prefix_caching_enabled;
        if use_prefix_cache && !self.cache.hbm_pools[flat].has_seq(seq_id) {
            return self.cache_ensure_group_blocks_with_host_prefix(
                seq_id,
                dp_idx,
                group_id,
                tokens,
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
        let blocks = self.cache.hbm_pools[flat]
            .ensure_blocks(seq_id, token_ids, tokens, use_prefix_cache)
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
        let needed_tokens = if is_prefill && self.config.mode != "decode" {
            num_tokens
        } else {
            num_tokens + 1
        };
        let needed_blocks = self.blocks_needed_for_tokens(needed_tokens);
        if !is_prefill
            && self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4) | (1 << 6)) == 0
            && self.cache.compressed_pools.is_empty()
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
        let (dp_idx, group_id) = match self.cache.seq_assignment.get(&seq_id).copied() {
            Some(assignment) => assignment,
            None => {
                let assignment = self.choose_assignment();
                self.cache.seq_assignment.insert(seq_id, assignment);
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
        let blocks = self.cache.hbm_pools[flat]
            .ensure_blocks(seq_id, token_ids, needed_tokens, use_prefix_cache)
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
            if let Some(slot) = self.cache.state_slots.get(seq_id) {
                s.migrate_state_slot = slot;
            }
            if let Some(slot) = self.cache.hisparse_slots.get(seq_id) {
                s.migrate_hisparse_slot = slot;
            }
            for (ratio, pool) in &self.cache.compressed_pools {
                if let Some(pages) = pool.seq_pages.get(&seq_id) {
                    s.migrate_compressed_block_tables
                        .insert(*ratio, pages.clone());
                }
            }
        }
        Ok(())
    }

    pub(super) fn cache_ensure_state_slot(&mut self, _py: Python<'_>, seq_id: u64) -> PyResult<()> {
        if self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) == 0 {
            return Ok(());
        }
        let Some(slot) = self.cache.state_slots.ensure(seq_id) else {
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
        let Some(slot) = self.cache.hisparse_slots.ensure(seq_id) else {
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
        if self.cache.compressed_pools.is_empty() {
            return Ok(());
        }
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_compressed_block_tables.clear();
        }
        for (ratio, pool) in self.cache.compressed_pools.iter_mut() {
            let needed_pages = Self::cache_compressed_pages_needed_for_tokens(pool, needed_tokens);
            let pages = pool.seq_pages.entry(seq_id).or_default();
            while pages.len() < needed_pages {
                let Some(page) = pool.free_pages.pop() else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "out of compressed cache pages: seq_id={seq_id} ratio={ratio} need={} have={}",
                        needed_pages,
                        pages.len()
                    )));
                };
                pages.push(page);
            }
            if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                seq.active_compressed_block_tables
                    .insert(*ratio, pages.clone());
            }
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
                if let Some(block_id) =
                    self.cache.hbm_pools[flat].lookup_ready_prefix_block(hash, view)
                {
                    self.cache.hbm_pools[flat].retain_ready_prefix_block(block_id)?;
                    blocks.push(block_id);
                    continue;
                }
                if let Some(host_block) =
                    self.cache.host_pools[flat].lookup_ready_prefix_block(hash, view)
                {
                    let (gpu_block, evicted) =
                        self.cache.hbm_pools[flat].allocate_promoted_ready(hash, view)?;
                    self.cache_write_back_evicted_prefix(flat, evicted)?;
                    self.cache.pending_swap_in_tasks[flat].push((
                        seq_id,
                        vec![host_block],
                        vec![gpu_block],
                    ));
                    blocks.push(gpu_block);
                    continue;
                }
            }
            cache_miss = true;
            let (block_id, evicted) = self.cache.hbm_pools[flat].allocate_pending_fresh()?;
            self.cache_write_back_evicted_prefix(flat, evicted)?;
            blocks.push(block_id);
        }
        self.cache.hbm_pools[flat].insert_seq_blocks(seq_id, blocks.clone());
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

    fn cache_compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
    }
}

impl Scheduler {
    pub(super) fn try_adopt_session(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        batch_tokens: &[i32],
    ) -> PyResult<CacheHit> {
        match self.active_cache_coordinator() {
            CoordinatorKind::Default => {
                DefaultCoordinator::try_adopt_session(self, py, seq_id, dp_idx, batch_tokens)
            }
            CoordinatorKind::Hisparse => {
                HisparseCoordinator::try_adopt_session(self, py, seq_id, dp_idx, batch_tokens)
            }
        }
    }

    pub(super) fn release_seq(&mut self, seq_id: u64) {
        self.cache.release_seq(seq_id, self.group());
    }

    pub(super) fn park_or_release(&mut self, py: Python<'_>, seq_id: u64) {
        match self.active_cache_coordinator() {
            CoordinatorKind::Default => DefaultCoordinator::park_or_release(self, py, seq_id),
            CoordinatorKind::Hisparse => HisparseCoordinator::park_or_release(self, py, seq_id),
        }
    }
}
