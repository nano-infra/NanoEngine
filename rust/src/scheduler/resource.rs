use super::Scheduler;
use crate::sequence::Sequence;
use crate::table::block::CompressedPool;
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn ensure_group_blocks(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        tokens: i32,
        set_migrate: bool,
    ) -> PyResult<Vec<i32>> {
        let needed_blocks = self.blocks_needed_for_tokens(tokens);
        let flat = self.flat_idx(dp_idx, group_id);
        let token_ids = seq.borrow(py).token_ids.clone();
        let use_prefix_cache = set_migrate && self.group() == 1 && self.config.mode != "decode";
        let blocks = self.hbm_pools[flat]
            .ensure_blocks(seq_id, &token_ids, tokens, use_prefix_cache)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
            })?;
        {
            let mut s = seq.borrow_mut(py);
            let group_id = group_id as i32;
            s.active_group_id = group_id.max(0);
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
        }
        if set_migrate {
            let mut s = seq.borrow_mut(py);
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

    fn compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
    }

    pub(super) fn ensure_blocks_for_seq(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        is_prefill: bool,
    ) -> PyResult<()> {
        let (seq_id, num_tokens) = {
            let s = seq.borrow(py);
            (s.seq_id, s.num_tokens)
        };
        let needed_tokens = if is_prefill && self.config.mode != "decode" {
            num_tokens
        } else {
            num_tokens + 1
        };
        let needed_blocks = self.blocks_needed_for_tokens(needed_tokens);
        let (dp_idx, group_id) = match self.seq_assignment.get(&seq_id).copied() {
            Some(assignment) => assignment,
            None => {
                let assignment = self.choose_assignment();
                self.seq_assignment.insert(seq_id, assignment);
                assignment
            }
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let token_ids = seq.borrow(py).token_ids.clone();
        let use_prefix_cache = is_prefill && self.group() == 1 && self.config.mode != "decode";
        let blocks = self.hbm_pools[flat]
            .ensure_blocks(seq_id, &token_ids, needed_tokens, use_prefix_cache)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id} need={needed_blocks}"
                ))
            })?;
        {
            let mut s = seq.borrow_mut(py);
            let group_id = group_id as i32;
            s.active_group_id = group_id.max(0);
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
            s.active_dp_idx = dp_idx as i32;
        }

        self.ensure_state_slot(py, seq, seq_id)?;
        self.ensure_hisparse_slot(py, seq, seq_id)?;
        self.ensure_compressed_pages(py, seq, seq_id, needed_tokens)?;

        if is_prefill && self.config.mode != "decode" {
            let mut s = seq.borrow_mut(py);
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

    pub(super) fn ensure_state_slot(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        seq_id: u64,
    ) -> PyResult<()> {
        if self.config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) == 0 {
            return Ok(());
        }
        let Some(slot) = self.state_slots.ensure(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of state slots: seq_id={seq_id}"
            )));
        };
        seq.borrow_mut(py).active_state_slot = slot;
        Ok(())
    }

    pub(super) fn ensure_hisparse_slot(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        seq_id: u64,
    ) -> PyResult<()> {
        if self.config.cache_plan.flags & (1 << 6) == 0 {
            return Ok(());
        }
        let Some(slot) = self.hisparse_slots.ensure(seq_id) else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "out of HiSparse slots: seq_id={seq_id}"
            )));
        };
        seq.borrow_mut(py).active_hisparse_slot = slot;
        Ok(())
    }

    pub(super) fn ensure_compressed_pages(
        &mut self,
        py: Python<'_>,
        seq: &Py<Sequence>,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<()> {
        if self.compressed_pools.is_empty() {
            return Ok(());
        }
        seq.borrow_mut(py).active_compressed_block_tables.clear();
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
            seq.borrow_mut(py)
                .active_compressed_block_tables
                .insert(*ratio, pages.clone());
        }
        Ok(())
    }
}
