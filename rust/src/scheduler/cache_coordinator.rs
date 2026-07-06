use super::{PendingHostSwap, Scheduler, HOST_PREFIX_CACHE_SEQ_ID, HOST_SWAP_IN_COOLDOWN_STEPS};
use crate::table::block::EvictedBlock;
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
        if self.hbm_pools[flat].num_free_blocks() < host_blocks.len() {
            return Ok(false);
        }
        let tokens = (host_blocks.len() as i32) * self.config.kvcache_block_size.max(1);
        let gpu_blocks = self.hbm_pools[flat]
            .ensure_blocks(seq_id, &[], tokens, false)
            .map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "out of KV cache blocks while restoring host KV: seq_id={seq_id} dp_idx={dp_idx} group_id={group_id}"
                ))
            })?;
        self.seq_assignment.insert(seq_id, (dp_idx, group_id));
        if let Some(seq) = self.seq_table.get_mut(&seq_id) {
            seq.active_dp_idx = dp_idx as i32;
            seq.active_group_id = group_id as i32;
            seq.active_block_table = gpu_blocks.clone();
            seq.active_block_tables
                .insert(group_id as i32, gpu_blocks.clone());
            seq.status = 1;
        }
        if let Some(tasks) = self.pending_swap_in_tasks.get_mut(flat) {
            tasks.push((seq_id, host_blocks, gpu_blocks));
        }
        Ok(true)
    }

    pub(super) fn cache_prepare_host_swap_out(&mut self, seq_id: u64) -> bool {
        if self.config.num_host_kvcache_blocks <= 0 || self.pending_host_swaps.contains_key(&seq_id)
        {
            return false;
        }
        let Some((dp_idx, group_id)) = self.seq_assignment.get(&seq_id).copied() else {
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
        let Ok(host_blocks) = self.host_pools[flat].ensure_blocks(seq_id, &[], tokens, false)
        else {
            return false;
        };
        self.pending_host_swaps
            .insert(seq_id, PendingHostSwap { dp_idx, group_id });
        if let Some(tasks) = self.pending_swap_out_tasks.get_mut(flat) {
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
                let Some(pending) = self.pending_host_swaps.remove(&seq_id) else {
                    continue;
                };
                let flat = self.flat_idx(pending.dp_idx, pending.group_id);
                self.hbm_pools[flat].remove_seq(seq_id);
                self.seq_assignment.remove(&seq_id);
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.active_block_table.clear();
                    seq.active_block_tables.clear();
                }
            }
        }
        for tasks in &mut self.pending_swap_out_tasks {
            tasks.clear();
        }
    }

    pub(super) fn cache_complete_host_swap_ins(
        &mut self,
        tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    ) {
        for (flat, group_tasks) in tasks.into_iter().enumerate() {
            for (seq_id, _host_blocks, _gpu_blocks) in group_tasks {
                if let Some(pool) = self.host_pools.get_mut(flat) {
                    pool.remove_seq(seq_id);
                }
                if let Some(seq) = self.seq_table.get_mut(&seq_id) {
                    seq.host_block_table.clear();
                    seq.host_block_tables.clear();
                }
            }
        }
        for tasks in &mut self.pending_swap_in_tasks {
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
        let host_block =
            self.host_pools[flat].store_ready_cache_block(evicted.hash, &evicted.token_ids)?;
        self.pending_swap_out_tasks[flat].push((
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

    pub(super) fn cache_ensure_state_slot(&mut self, py: Python<'_>, seq_id: u64) -> PyResult<()> {
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
