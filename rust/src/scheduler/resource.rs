use super::Scheduler;
use crate::table::block::{compute_block_hash, CompressedPool, EvictedBlock};
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn ensure_group_blocks(
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
            && self.prefix_caching_enabled;
        if use_prefix_cache && !self.hbm_pools[flat].has_seq(seq_id) {
            return self.ensure_group_blocks_with_host_prefix(
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
        let blocks = self.hbm_pools[flat]
            .ensure_blocks(seq_id, token_ids, tokens, use_prefix_cache)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
        })?;
        {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(blocks);
            };
            let group_id = group_id as i32;
            s.active_group_id = group_id.max(0);
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
        }
        if set_migrate {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(blocks);
            };
            let group_id = group_id as i32;
            s.migrate_group_id = group_id.max(0);
            s.migrate_block_table = blocks.clone();
            s.migrate_block_tables.insert(group_id, blocks.clone());
            s.migrate_engine_id = self.engine_id_.clone();
            s.migrate_num_kvcache_blocks = self.config.num_kvcache_blocks;
            s.migrate_group_size = self.config.group_size.max(1);
            s.migrate_dp_idx = dp_idx as i32;
        }
        Ok(blocks)
    }

    fn ensure_group_blocks_with_host_prefix(
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
            let view = self.block_token_view(&token_ids, block_idx);
            let full = view.len() == self.config.kvcache_block_size.max(1) as usize;
            if full && !cache_miss {
                hash = compute_block_hash(view, hash);
                if let Some(block_id) = self.hbm_pools[flat].lookup_ready_prefix_block(hash, view) {
                    self.hbm_pools[flat].retain_ready_prefix_block(block_id)?;
                    blocks.push(block_id);
                    continue;
                }
                if let Some(host_block) =
                    self.host_pools[flat].lookup_ready_prefix_block(hash, view)
                {
                    let (gpu_block, evicted) =
                        self.hbm_pools[flat].allocate_promoted_ready(hash, view)?;
                    self.write_back_evicted_prefix(flat, evicted)?;
                    self.pending_swap_in_tasks[flat].push((
                        seq_id,
                        vec![host_block],
                        vec![gpu_block],
                    ));
                    blocks.push(gpu_block);
                    continue;
                }
            }
            cache_miss = true;
            let (block_id, evicted) = self.hbm_pools[flat].allocate_pending_fresh()?;
            self.write_back_evicted_prefix(flat, evicted)?;
            blocks.push(block_id);
        }
        self.hbm_pools[flat].insert_seq_blocks(seq_id, blocks.clone());
        {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(blocks);
            };
            let group_id = group_id as i32;
            s.active_group_id = group_id.max(0);
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
        }
        if set_migrate {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(blocks);
            };
            let group_id = group_id as i32;
            s.migrate_group_id = group_id.max(0);
            s.migrate_block_table = blocks.clone();
            s.migrate_block_tables.insert(group_id, blocks.clone());
            s.migrate_engine_id = self.engine_id_.clone();
            s.migrate_num_kvcache_blocks = self.config.num_kvcache_blocks;
            s.migrate_group_size = self.config.group_size.max(1);
            s.migrate_dp_idx = dp_idx as i32;
        }
        Ok(blocks)
    }

    fn write_back_evicted_prefix(
        &mut self,
        flat: usize,
        evicted: Option<EvictedBlock>,
    ) -> PyResult<()> {
        self.cache_write_back_evicted_prefix(flat, evicted)
    }

    pub(super) fn cached_tokens_for_with_host_prefix(
        &mut self,
        flat: usize,
        token_ids: &[i32],
        tokens: i32,
    ) -> i32 {
        let block_size = self.config.kvcache_block_size.max(1) as usize;
        let mut hash = -1i64;
        let mut matched = 0i32;
        for block_idx in 0..self.blocks_needed_for_tokens(tokens) {
            let view = self.block_token_view(token_ids, block_idx);
            if view.len() != block_size {
                break;
            }
            hash = compute_block_hash(view, hash);
            let hbm_hit = self.hbm_pools[flat]
                .lookup_ready_prefix_block(hash, view)
                .is_some();
            let host_hit = if hbm_hit {
                false
            } else {
                self.host_pools[flat]
                    .lookup_ready_prefix_block(hash, view)
                    .is_some()
            };
            if !hbm_hit && !host_hit {
                break;
            }
            matched += 1;
        }
        let cached = (matched * self.config.kvcache_block_size.max(1)).min((tokens - 1).max(0));
        let gqa_hisparse = self.config.cache_plan.flags & (1 << 0) != 0
            && self.config.cache_plan.flags & (1 << 6) != 0;
        if gqa_hisparse {
            let tail = self.config.cache_plan.hisparse.swap_in_block_size.max(1);
            cached.min((tokens - tail).max(0))
        } else {
            cached
        }
    }

    fn block_token_view<'a>(&self, token_ids: &'a [i32], block_idx: usize) -> &'a [i32] {
        let block_size = self.config.kvcache_block_size.max(1) as usize;
        let start = block_idx * block_size;
        let end = (start + block_size).min(token_ids.len());
        if start >= end {
            &[]
        } else {
            &token_ids[start..end]
        }
    }

    fn compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
    }

    pub(super) fn ensure_blocks_for_seq(
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
            && self.compressed_pools.is_empty()
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
        let (dp_idx, group_id) = match self.seq_assignment.get(&seq_id).copied() {
            Some(assignment) => assignment,
            None => {
                let assignment = self.choose_assignment();
                self.seq_assignment.insert(seq_id, assignment);
                assignment
            }
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let use_prefix_cache = is_prefill
            && self.group() == 1
            && self.config.mode != "decode"
            && self.prefix_caching_enabled;
        let token_ids = if use_prefix_cache {
            self.seq_table
                .get(&seq_id)
                .map(|seq| seq.token_ids.as_slice())
                .unwrap_or(&[])
        } else {
            &[]
        };
        let blocks = self.hbm_pools[flat]
            .ensure_blocks(seq_id, token_ids, needed_tokens, use_prefix_cache)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
        })?;
        {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(());
            };
            let group_id = group_id as i32;
            s.active_group_id = group_id.max(0);
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
            s.active_dp_idx = dp_idx as i32;
        }

        self.ensure_state_slot(py, seq_id)?;
        self.ensure_hisparse_slot(py, seq_id)?;
        self.ensure_compressed_pages(py, seq_id, needed_tokens)?;

        if is_prefill && self.config.mode != "decode" {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(());
            };
            let group_id = group_id as i32;
            s.migrate_group_id = group_id.max(0);
            s.migrate_block_table = blocks.clone();
            s.migrate_block_tables.insert(group_id, blocks.clone());
            s.migrate_engine_id = self.engine_id_.clone();
            s.migrate_num_kvcache_blocks = self.config.num_kvcache_blocks;
            s.migrate_group_size = self.config.group_size.max(1);
            s.migrate_dp_idx = dp_idx as i32;
            if let Some(slot) = self.state_slots.get(seq_id) {
                s.migrate_state_slot = slot;
            }
            if let Some(slot) = self.hisparse_slots.get(seq_id) {
                s.migrate_hisparse_slot = slot;
            }
            for (ratio, pool) in &self.compressed_pools {
                if let Some(pages) = pool.seq_pages.get(&seq_id) {
                    s.migrate_compressed_block_tables
                        .insert(*ratio, pages.clone());
                }
            }
        }
        Ok(())
    }

    pub(super) fn ensure_state_slot(&mut self, _py: Python<'_>, seq_id: u64) -> PyResult<()> {
        if self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) == 0 {
            return Ok(());
        }
        let Some(slot) = self.state_slots.ensure(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of state slots: seq_id={seq_id}"
            )));
        };
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_state_slot = slot;
        }
        Ok(())
    }

    pub(super) fn ensure_hisparse_slot(&mut self, _py: Python<'_>, seq_id: u64) -> PyResult<()> {
        if self.config.cache_plan.flags & (1 << 6) == 0 {
            return Ok(());
        }
        let Some(slot) = self.hisparse_slots.ensure(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of HiSparse slots: seq_id={seq_id}"
            )));
        };
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_hisparse_slot = slot;
        }
        Ok(())
    }

    pub(super) fn ensure_compressed_pages(
        &mut self,
        _py: Python<'_>,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<()> {
        if self.compressed_pools.is_empty() {
            return Ok(());
        }
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_compressed_block_tables.clear();
        }
        for (ratio, pool) in self.compressed_pools.iter_mut() {
            let needed_pages = Self::compressed_pages_needed_for_tokens(pool, needed_tokens);
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
}
