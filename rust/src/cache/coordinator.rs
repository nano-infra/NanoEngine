use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use pyo3::prelude::*;

use crate::cache::table::block::{compute_block_hash, CompressedPool, EvictedBlock};
use crate::scheduler::SchedulerConfig;

use super::{CacheState, PendingHostSwap};

/// Prefix-cache coordination boundary between scheduling and cache ownership.
///
/// `PrefixCacheCoordinator` is the only cache-facing object the scheduler should
/// own. It coordinates real block ownership (`BlockPool`/future block managers),
/// prefix lookup visibility (future `HashTree`/`PrefixIndex`), host spill state,
/// and request/session cache lifecycle.
///
/// The ordering rules are intentionally part of the type contract:
///
/// - publish: block owner commits or seals data, then the prefix index may expose
///   a location snapshot;
/// - revoke: block owner releases or invalidates data, then the prefix index
///   marks the corresponding location stale;
/// - lookup: prefix index hits are candidates only, and must be resolved against
///   the owning block manager before data is read.
///
/// Today this type wraps `CacheState` while existing scheduler cache paths are
/// being moved behind coordinator methods. The `Deref` compatibility is a
/// transitional shim: new cache mutations should prefer explicit coordinator
/// methods rather than reaching into the state directly.
pub(crate) struct PrefixCacheCoordinator {
    state: CacheState,
}

impl PrefixCacheCoordinator {
    pub(crate) fn new(config: &SchedulerConfig, dp: usize, group: usize) -> Self {
        Self {
            state: CacheState::new(config, dp, group),
        }
    }

    pub(crate) fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        self.state.set_prefix_caching_enabled(enabled);
    }

    pub(crate) fn release_seq(&mut self, seq_id: u64, group: usize) {
        self.state.release_seq(seq_id, group);
    }

    pub(crate) fn assignment(&self, seq_id: u64) -> Option<(usize, usize)> {
        self.state.seq_assignment.get(&seq_id).copied()
    }

    pub(crate) fn assign_seq(&mut self, seq_id: u64, dp_idx: usize, group_id: usize) {
        self.state.seq_assignment.insert(seq_id, (dp_idx, group_id));
    }

    pub(crate) fn remove_assignment(&mut self, seq_id: u64) -> Option<(usize, usize)> {
        self.state.seq_assignment.remove(&seq_id)
    }

    pub(crate) fn has_pending_host_swap(&self, seq_id: u64) -> bool {
        self.state.pending_host_swaps.contains_key(&seq_id)
    }

    pub(crate) fn insert_pending_host_swap(&mut self, seq_id: u64, dp_idx: usize, group_id: usize) {
        self.state
            .pending_host_swaps
            .insert(seq_id, PendingHostSwap { dp_idx, group_id });
    }

    pub(crate) fn take_pending_host_swap(&mut self, seq_id: u64) -> Option<PendingHostSwap> {
        self.state.pending_host_swaps.remove(&seq_id)
    }

    pub(crate) fn clear_pending_swap_out_tasks(&mut self) {
        for tasks in &mut self.state.pending_swap_out_tasks {
            tasks.clear();
        }
    }

    pub(crate) fn clear_pending_swap_in_tasks(&mut self) {
        for tasks in &mut self.state.pending_swap_in_tasks {
            tasks.clear();
        }
    }

    pub(crate) fn push_pending_swap_out(
        &mut self,
        flat: usize,
        seq_id: u64,
        from_blocks: Vec<i32>,
        to_blocks: Vec<i32>,
    ) {
        if let Some(tasks) = self.state.pending_swap_out_tasks.get_mut(flat) {
            tasks.push((seq_id, from_blocks, to_blocks));
        }
    }

    pub(crate) fn push_pending_swap_in(
        &mut self,
        flat: usize,
        seq_id: u64,
        from_blocks: Vec<i32>,
        to_blocks: Vec<i32>,
    ) {
        if let Some(tasks) = self.state.pending_swap_in_tasks.get_mut(flat) {
            tasks.push((seq_id, from_blocks, to_blocks));
        }
    }

    pub(crate) fn hbm_num_free_blocks(&self, flat: usize) -> usize {
        self.state
            .hbm_pools
            .get(flat)
            .map(|pool| pool.num_free_blocks())
            .unwrap_or(0)
    }

    pub(crate) fn hbm_has_seq(&self, flat: usize, seq_id: u64) -> bool {
        self.state
            .hbm_pools
            .get(flat)
            .map(|pool| pool.has_seq(seq_id))
            .unwrap_or(false)
    }

    pub(crate) fn ensure_hbm_blocks(
        &mut self,
        flat: usize,
        seq_id: u64,
        token_ids: &[i32],
        tokens: i32,
        use_prefix_cache: bool,
    ) -> PyResult<Vec<i32>> {
        self.state.hbm_pools[flat].ensure_blocks(seq_id, token_ids, tokens, use_prefix_cache)
    }

    pub(crate) fn ensure_host_blocks(
        &mut self,
        flat: usize,
        seq_id: u64,
        token_ids: &[i32],
        tokens: i32,
        use_prefix_cache: bool,
    ) -> PyResult<Vec<i32>> {
        self.state.host_pools[flat].ensure_blocks(seq_id, token_ids, tokens, use_prefix_cache)
    }

    pub(crate) fn remove_hbm_seq(&mut self, flat: usize, seq_id: u64) {
        self.state.hbm_pools[flat].remove_seq(seq_id);
    }

    pub(crate) fn remove_host_seq(&mut self, flat: usize, seq_id: u64) {
        if let Some(pool) = self.state.host_pools.get_mut(flat) {
            pool.remove_seq(seq_id);
        }
    }

    pub(crate) fn write_back_evicted_prefix(
        &mut self,
        flat: usize,
        evicted: Option<EvictedBlock>,
        host_enabled: bool,
        host_prefix_cache_seq_id: u64,
    ) -> PyResult<()> {
        let Some(evicted) = evicted else {
            return Ok(());
        };
        if !host_enabled {
            return Ok(());
        }
        let host_block = self.state.host_pools[flat]
            .store_ready_cache_block(evicted.hash, &evicted.token_ids)?;
        self.push_pending_swap_out(
            flat,
            host_prefix_cache_seq_id,
            vec![evicted.block_id],
            vec![host_block],
        );
        Ok(())
    }

    pub(crate) fn cached_tokens_for_prefix(
        &mut self,
        flat: usize,
        token_ids: &[i32],
        tokens: i32,
        block_size: i32,
        blocks_needed: usize,
    ) -> i32 {
        let block_size = block_size.max(1) as usize;
        let mut hash = -1i64;
        let mut matched = 0i32;
        for block_idx in 0..blocks_needed {
            let start = block_idx * block_size;
            let end = (start + block_size).min(token_ids.len());
            if start >= end || end - start != block_size {
                break;
            }
            let view = &token_ids[start..end];
            hash = compute_block_hash(view, hash);
            let hbm_hit = self.state.hbm_pools[flat]
                .lookup_ready_prefix_block(hash, view)
                .is_some();
            let host_hit = if hbm_hit {
                false
            } else {
                self.state.host_pools[flat]
                    .lookup_ready_prefix_block(hash, view)
                    .is_some()
            };
            if !hbm_hit && !host_hit {
                break;
            }
            matched += 1;
        }
        (matched * block_size as i32).min((tokens - 1).max(0))
    }

    pub(crate) fn lookup_hbm_ready_prefix(
        &mut self,
        flat: usize,
        hash: i64,
        tokens: &[i32],
    ) -> Option<i32> {
        self.state.hbm_pools[flat].lookup_ready_prefix_block(hash, tokens)
    }

    pub(crate) fn retain_hbm_ready_prefix(&mut self, flat: usize, block_id: i32) -> PyResult<()> {
        self.state.hbm_pools[flat].retain_ready_prefix_block(block_id)
    }

    pub(crate) fn lookup_host_ready_prefix(
        &mut self,
        flat: usize,
        hash: i64,
        tokens: &[i32],
    ) -> Option<i32> {
        self.state.host_pools[flat].lookup_ready_prefix_block(hash, tokens)
    }

    pub(crate) fn allocate_hbm_promoted_ready(
        &mut self,
        flat: usize,
        hash: i64,
        tokens: &[i32],
    ) -> PyResult<(i32, Option<EvictedBlock>)> {
        self.state.hbm_pools[flat].allocate_promoted_ready(hash, tokens)
    }

    pub(crate) fn allocate_hbm_pending_fresh(
        &mut self,
        flat: usize,
    ) -> PyResult<(i32, Option<EvictedBlock>)> {
        self.state.hbm_pools[flat].allocate_pending_fresh()
    }

    pub(crate) fn insert_hbm_seq_blocks(&mut self, flat: usize, seq_id: u64, blocks: Vec<i32>) {
        self.state.hbm_pools[flat].insert_seq_blocks(seq_id, blocks);
    }

    pub(crate) fn compressed_pools_empty(&self) -> bool {
        self.state.compressed_pools.is_empty()
    }

    pub(crate) fn state_slot(&self, seq_id: u64) -> Option<i32> {
        self.state.state_slots.get(seq_id)
    }

    pub(crate) fn hisparse_slot(&self, seq_id: u64) -> Option<i32> {
        self.state.hisparse_slots.get(seq_id)
    }

    pub(crate) fn ensure_state_slot(&mut self, seq_id: u64) -> Option<i32> {
        self.state.state_slots.ensure(seq_id)
    }

    pub(crate) fn can_ensure_state_slot(&self, seq_id: u64) -> bool {
        self.state.state_slots.can_ensure(seq_id)
    }

    pub(crate) fn ensure_hisparse_slot(&mut self, seq_id: u64) -> Option<i32> {
        self.state.hisparse_slots.ensure(seq_id)
    }

    pub(crate) fn compressed_pages(&self, seq_id: u64) -> HashMap<i32, Vec<i32>> {
        self.state
            .compressed_pools
            .iter()
            .filter_map(|(ratio, pool)| {
                pool.seq_pages
                    .get(&seq_id)
                    .map(|pages| (*ratio, pages.clone()))
            })
            .collect()
    }

    pub(crate) fn ensure_compressed_pages(
        &mut self,
        seq_id: u64,
        needed_tokens: i32,
    ) -> PyResult<HashMap<i32, Vec<i32>>> {
        let mut active_tables = HashMap::new();
        for (ratio, pool) in self.state.compressed_pools.iter_mut() {
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
            active_tables.insert(*ratio, pages.clone());
        }
        Ok(active_tables)
    }

    pub(crate) fn set_prefix_cached_tokens(&mut self, seq_id: u64, tokens: i32) {
        self.state
            .prefix_cached_tokens_by_seq
            .insert(seq_id, tokens);
    }

    fn compressed_pages_needed_for_tokens(pool: &CompressedPool, tokens: i32) -> usize {
        let compressed_tokens = ((tokens.max(1) + pool.ratio - 1) / pool.ratio).max(1);
        let pages = (compressed_tokens + pool.page_size - 1) / pool.page_size;
        pages.min(pool.max_blocks_per_seq).max(1) as usize
    }
}

impl Deref for PrefixCacheCoordinator {
    type Target = CacheState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for PrefixCacheCoordinator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}
