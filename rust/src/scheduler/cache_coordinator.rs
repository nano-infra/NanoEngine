use super::Scheduler;
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn cache_seq_has_host_blocks(&self, seq_id: u64) -> bool {
        self.seq_has_host_blocks(seq_id)
    }

    pub(super) fn cache_try_restore_host_blocks(
        &mut self,
        seq_id: u64,
        dp_idx: usize,
    ) -> PyResult<bool> {
        self.try_restore_host_blocks(seq_id, dp_idx)
    }

    pub(super) fn cache_cached_tokens_for_prefix(
        &mut self,
        flat: usize,
        token_ids: &[i32],
        tokens: i32,
    ) -> i32 {
        self.cached_tokens_for_with_host_prefix(flat, token_ids, tokens)
    }

    pub(super) fn cache_ensure_group_blocks(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        group_id: usize,
        tokens: i32,
        set_migrate: bool,
    ) -> PyResult<Vec<i32>> {
        self.ensure_group_blocks(py, seq_id, dp_idx, group_id, tokens, set_migrate)
    }

    pub(super) fn cache_ensure_blocks_for_seq(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        is_prefill: bool,
    ) -> PyResult<()> {
        self.ensure_blocks_for_seq(py, seq_id, is_prefill)
    }

    pub(super) fn cache_ensure_state_slot(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
    ) -> PyResult<()> {
        self.ensure_state_slot(py, seq_id)
    }

    pub(super) fn cache_ensure_hisparse_slot(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
    ) -> PyResult<()> {
        self.ensure_hisparse_slot(py, seq_id)
    }

    pub(super) fn cache_ensure_compressed_pages(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<()> {
        self.ensure_compressed_pages(py, seq_id, needed_tokens)
    }
}
