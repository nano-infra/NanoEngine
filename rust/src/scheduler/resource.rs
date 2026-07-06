use super::Scheduler;
use crate::table::block::CompressedPool;
use pyo3::prelude::*;

impl Scheduler {
    fn compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
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
