use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use pyo3::prelude::*;

use crate::cache::table::block::{compute_block_hash, CompressedPool, EvictedBlock};
use crate::scheduler::SchedulerConfig;

use super::{CacheState, ParkedSession, PendingHostSwap};

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

pub(crate) struct AdoptedSessionSnapshot {
    pub(crate) group_id: usize,
    pub(crate) length: i32,
    pub(crate) block_table: Option<Vec<i32>>,
    pub(crate) state_slot: i32,
    pub(crate) hisparse_slot: i32,
    pub(crate) compressed_tables: HashMap<i32, Vec<i32>>,
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

    pub(crate) fn num_parked_sessions(&self) -> i32 {
        self.state.num_parked_sessions()
    }

    pub(crate) fn parked_session_keys(&self) -> Vec<u64> {
        self.state.parked_session_keys()
    }

    pub(crate) fn clear_session_cache(&mut self, group: usize) {
        self.state.clear_session_cache(group);
    }

    pub(crate) fn release_seq(&mut self, seq_id: u64, group: usize) {
        self.state.release_seq(seq_id, group);
    }

    pub(crate) fn insert_parked(&mut self, parked: ParkedSession, capacity: i32, group: usize) {
        self.state.insert_parked(parked, capacity, group);
    }

    pub(crate) fn evict_parked_by_key(&mut self, key: u64, group: usize) {
        self.state.evict_parked_by_key(key, group);
    }

    pub(crate) fn parked_session(&self, key: u64) -> Option<&ParkedSession> {
        self.state.parked_sessions.get(&key)
    }

    pub(crate) fn take_parked_session(&mut self, key: u64) -> Option<ParkedSession> {
        let parked = self.state.parked_sessions.remove(&key)?;
        self.state.parked_lru.retain(|stored| *stored != key);
        Some(parked)
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

    pub(crate) fn insert_existing_hbm_blocks(
        &mut self,
        flat: usize,
        seq_id: u64,
        blocks: Vec<i32>,
    ) {
        self.state.hbm_pools[flat].insert_existing(seq_id, blocks);
    }

    pub(crate) fn take_hbm_seq_without_release(&mut self, flat: usize, seq_id: u64) -> Vec<i32> {
        self.state.hbm_pools[flat].take_seq_without_release(seq_id)
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

    pub(crate) fn ensure_hisparse_slot(&mut self, seq_id: u64) -> Option<i32> {
        self.state.hisparse_slots.ensure(seq_id)
    }

    pub(crate) fn insert_existing_state_slot(&mut self, seq_id: u64, slot: i32) {
        self.state.state_slots.insert_existing(seq_id, slot);
    }

    pub(crate) fn insert_existing_hisparse_slot(&mut self, seq_id: u64, slot: i32) {
        self.state.hisparse_slots.insert_existing(seq_id, slot);
    }

    pub(crate) fn take_state_slot_without_release(&mut self, seq_id: u64) -> Option<i32> {
        self.state.state_slots.take_without_release(seq_id)
    }

    pub(crate) fn take_hisparse_slot_without_release(&mut self, seq_id: u64) -> Option<i32> {
        self.state.hisparse_slots.take_without_release(seq_id)
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

    pub(crate) fn insert_existing_compressed_pages(
        &mut self,
        seq_id: u64,
        ratio: i32,
        pages: Vec<i32>,
    ) {
        if let Some(pool) = self.state.compressed_pools.get_mut(&ratio) {
            pool.seq_pages.insert(seq_id, pages);
        }
    }

    pub(crate) fn take_compressed_pages(&mut self, seq_id: u64) -> HashMap<i32, Vec<i32>> {
        let mut compressed_tables = HashMap::new();
        for (ratio, pool) in self.state.compressed_pools.iter_mut() {
            if let Some(pages) = pool.seq_pages.remove(&seq_id) {
                compressed_tables.insert(*ratio, pages);
            }
        }
        compressed_tables
    }

    pub(crate) fn set_prefix_cached_tokens(&mut self, seq_id: u64, tokens: i32) {
        self.state
            .prefix_cached_tokens_by_seq
            .insert(seq_id, tokens);
    }

    pub(crate) fn session_snapshot_enabled(gdn_state_cache_slots: i32, group: usize) -> bool {
        gdn_state_cache_slots > 0 && group == 1
    }

    pub(crate) fn take_matching_session_snapshot(
        &mut self,
        group: usize,
        seq_id: u64,
        dp_idx: usize,
        affinity: u64,
        full_len: i32,
        token_ids: &[i32],
    ) -> Option<AdoptedSessionSnapshot> {
        let Some(parked) = self.parked_session(affinity) else {
            return None;
        };
        if parked.dp_idx != dp_idx {
            return None;
        }
        if !Self::session_snapshot_prefix_matches(parked, full_len, token_ids) {
            self.evict_parked_by_key(affinity, group);
            return None;
        }

        let mut parked = self.take_parked_session(affinity)?;
        self.assign_seq(seq_id, dp_idx, parked.group_id);

        let block_table = parked.block_tables.remove(&(parked.group_id as i32));
        if let Some(blocks) = &block_table {
            let flat = Self::flat_idx(group, dp_idx, parked.group_id);
            self.insert_existing_hbm_blocks(flat, seq_id, blocks.clone());
        }
        if parked.state_slot >= 0 {
            self.insert_existing_state_slot(seq_id, parked.state_slot);
        }
        if parked.hisparse_slot >= 0 {
            self.insert_existing_hisparse_slot(seq_id, parked.hisparse_slot);
        }
        for (ratio, pages) in &parked.compressed_tables {
            self.insert_existing_compressed_pages(seq_id, *ratio, pages.clone());
        }

        Some(AdoptedSessionSnapshot {
            group_id: parked.group_id,
            length: parked.length,
            block_table,
            state_slot: parked.state_slot,
            hisparse_slot: parked.hisparse_slot,
            compressed_tables: parked.compressed_tables,
        })
    }

    pub(crate) fn park_session_snapshot(
        &mut self,
        group: usize,
        seq_id: u64,
        affinity: u64,
        token_ids: Vec<i32>,
        length: i32,
        capacity: i32,
    ) -> bool {
        let Some((dp_idx, group_id)) = self.remove_assignment(seq_id) else {
            return false;
        };
        let flat = Self::flat_idx(group, dp_idx, group_id);
        let blocks = self.take_hbm_seq_without_release(flat, seq_id);
        let state_slot = self.take_state_slot_without_release(seq_id).unwrap_or(-1);
        let hisparse_slot = self
            .take_hisparse_slot_without_release(seq_id)
            .unwrap_or(-1);
        let compressed_tables = self.take_compressed_pages(seq_id);
        let parked = ParkedSession {
            affinity_key: affinity,
            state_slot,
            hisparse_slot,
            dp_idx,
            group_id,
            length,
            token_ids: token_ids.into_iter().take(length.max(0) as usize).collect(),
            block_tables: HashMap::from([(group_id as i32, blocks)]),
            compressed_tables,
        };
        self.insert_parked(parked, capacity, group);
        true
    }

    fn session_snapshot_prefix_matches(
        parked: &ParkedSession,
        full_len: i32,
        token_ids: &[i32],
    ) -> bool {
        parked.length > 0
            && parked.length < full_len
            && token_ids.len() >= parked.length as usize
            && token_ids[..parked.length as usize] == parked.token_ids[..]
    }

    fn flat_idx(group: usize, dp_idx: usize, group_id: usize) -> usize {
        dp_idx * group + group_id
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
