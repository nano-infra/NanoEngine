use std::collections::HashMap;

use super::Scheduler;
use pyo3::prelude::*;

pub(super) struct ParkedSession {
    pub(super) affinity_key: u64,
    pub(super) state_slot: i32,
    pub(super) hisparse_slot: i32,
    pub(super) dp_idx: usize,
    pub(super) group_id: usize,
    pub(super) length: i32,
    pub(super) token_ids: Vec<i32>,
    pub(super) block_tables: HashMap<i32, Vec<i32>>,
    pub(super) compressed_tables: HashMap<i32, Vec<i32>>,
}

impl Scheduler {
    pub(super) fn try_adopt_session(
        &mut self,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        batch_tokens: &[i32],
    ) -> PyResult<Option<i32>> {
        if self.config.gdn_state_cache_slots <= 0 || self.group() != 1 {
            return Ok(None);
        }
        let affinity = self
            .seq_table
            .get(&seq_id)
            .map(|seq| seq.affinity_key)
            .unwrap_or(0);
        if affinity == 0 {
            return Ok(None);
        }
        let Some(parked) = self.parked_sessions.get(&affinity) else {
            return Ok(None);
        };
        if parked.dp_idx != dp_idx {
            return Ok(None);
        }
        let (full_len, token_ids, seq_id) = {
            let Some(s) = self.seq_table.get(&seq_id) else {
                return Ok(None);
            };
            (self.prompt_target(&s), s.token_ids.clone(), s.seq_id)
        };
        let prefix_ok = parked.length > 0
            && parked.length < full_len
            && token_ids.len() >= parked.length as usize
            && token_ids[..parked.length as usize] == parked.token_ids[..];
        if !prefix_ok {
            self.evict_parked_by_key(affinity);
            return Ok(None);
        }
        let budget = (self.config.max_num_batched_tokens.max(1)
            - batch_tokens.get(0).copied().unwrap_or(0))
        .max(0);
        if budget <= 0 {
            return Ok(None);
        }
        let mut parked = self.parked_sessions.remove(&affinity).unwrap();
        self.parked_lru.retain(|key| *key != affinity);
        self.seq_assignment
            .insert(seq_id, (dp_idx, parked.group_id));
        if let Some(blocks) = parked.block_tables.remove(&(parked.group_id as i32)) {
            let flat = self.flat_idx(dp_idx, parked.group_id);
            self.hbm_pools[flat].insert_existing(seq_id, blocks.clone());
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(None);
            };
            let group_id = parked.group_id as i32;
            s.active_group_id = group_id;
            s.active_block_table = blocks.clone();
            s.active_block_tables.insert(group_id, blocks.clone());
            s.migrate_group_id = group_id;
            s.migrate_block_table = blocks.clone();
            s.migrate_block_tables.insert(group_id, blocks);
        }
        if parked.state_slot >= 0 {
            self.state_slots.insert_existing(seq_id, parked.state_slot);
            if let Some(s) = self.seq_table.get_mut(&seq_id) {
                s.active_state_slot = parked.state_slot;
                s.migrate_state_slot = parked.state_slot;
            }
        }
        if parked.hisparse_slot >= 0 {
            self.hisparse_slots
                .insert_existing(seq_id, parked.hisparse_slot);
            if let Some(s) = self.seq_table.get_mut(&seq_id) {
                s.active_hisparse_slot = parked.hisparse_slot;
                s.migrate_hisparse_slot = parked.hisparse_slot;
            }
        }
        for (ratio, pages) in parked.compressed_tables {
            if let Some(pool) = self.compressed_pools.get_mut(&ratio) {
                pool.seq_pages.insert(seq_id, pages.clone());
            }
            if let Some(s) = self.seq_table.get_mut(&seq_id) {
                s.active_compressed_block_tables
                    .insert(ratio, pages.clone());
                s.migrate_compressed_block_tables.insert(ratio, pages);
            }
        }

        let new_tokens = (full_len - parked.length).min(budget).max(0);
        let chunk_end = parked.length + new_tokens;
        let dispatch = self.dispatch_for_master(parked.group_id, chunk_end);
        {
            let Some(s) = self.seq_table.get_mut(&seq_id) else {
                return Ok(None);
            };
            s.num_cached_tokens = parked.length;
            s.prefill_start_offset = parked.length;
            s.num_tokens = chunk_end;
            s.active_dp_idx = dp_idx as i32;
            s.active_group_id = parked.group_id as i32;
            s.active_dispatched_tokens = dispatch;
        }
        self.cache_ensure_group_blocks(py, seq_id, dp_idx, parked.group_id, full_len, false)?;
        self.prefix_cached_tokens_by_seq
            .insert(seq_id, parked.length);
        Ok(Some(new_tokens))
    }

    pub(super) fn release_seq(&mut self, seq_id: u64) {
        if let Some(pending) = self.pending_host_swaps.remove(&seq_id) {
            let flat = self.flat_idx(pending.dp_idx, pending.group_id);
            self.host_pools[flat].remove_seq(seq_id);
        }
        if let Some((dp_idx, group_id)) = self.seq_assignment.remove(&seq_id) {
            let flat = self.flat_idx(dp_idx, group_id);
            self.hbm_pools[flat].remove_seq(seq_id);
        }
        for pool in &mut self.host_pools {
            pool.remove_seq(seq_id);
        }
        self.state_slots.remove(seq_id);
        self.hisparse_slots.remove(seq_id);
        for pool in self.compressed_pools.values_mut() {
            if let Some(mut pages) = pool.seq_pages.remove(&seq_id) {
                pool.free_pages.append(&mut pages);
            }
        }
    }

    pub(super) fn park_or_release(&mut self, _py: Python<'_>, seq_id: u64) {
        let (seq_id, affinity) = {
            let Some(s) = self.seq_table.get(&seq_id) else {
                return;
            };
            (s.seq_id, s.affinity_key)
        };
        if self.config.gdn_state_cache_slots <= 0 || self.group() != 1 || affinity == 0 {
            self.release_seq(seq_id);
            return;
        }
        let Some((dp_idx, group_id)) = self.seq_assignment.remove(&seq_id) else {
            self.release_seq(seq_id);
            return;
        };
        let flat = self.flat_idx(dp_idx, group_id);
        let blocks = self.hbm_pools[flat].take_seq_without_release(seq_id);
        let state_slot = self.state_slots.take_without_release(seq_id).unwrap_or(-1);
        let hisparse_slot = self
            .hisparse_slots
            .take_without_release(seq_id)
            .unwrap_or(-1);
        let mut compressed_tables = HashMap::new();
        for (ratio, pool) in self.compressed_pools.iter_mut() {
            if let Some(pages) = pool.seq_pages.remove(&seq_id) {
                compressed_tables.insert(*ratio, pages);
            }
        }
        let (token_ids, length) = {
            let Some(s) = self.seq_table.get(&seq_id) else {
                return;
            };
            (s.token_ids.clone(), s.num_tokens)
        };
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
        self.insert_parked(parked);
    }

    fn insert_parked(&mut self, parked: ParkedSession) {
        let key = parked.affinity_key;
        if self.parked_sessions.contains_key(&key) {
            self.evict_parked_by_key(key);
        }
        self.parked_lru.push(key);
        self.parked_sessions.insert(key, parked);
        while self.parked_sessions.len() > self.config.gdn_state_cache_slots.max(0) as usize {
            if let Some(key) = self.parked_lru.first().copied() {
                self.evict_parked_by_key(key);
            } else {
                break;
            }
        }
    }

    pub(super) fn evict_parked_by_key(&mut self, key: u64) {
        self.parked_lru.retain(|k| *k != key);
        let Some(parked) = self.parked_sessions.remove(&key) else {
            return;
        };
        if let Some(blocks) = parked.block_tables.get(&(parked.group_id as i32)) {
            let flat = self.flat_idx(parked.dp_idx, parked.group_id);
            self.hbm_pools[flat].release_blocks_without_owner(blocks);
        }
        if parked.state_slot >= 0 {
            self.state_slots.release_slot(parked.state_slot);
        }
        if parked.hisparse_slot >= 0 {
            self.hisparse_slots.release_slot(parked.hisparse_slot);
        }
        for (ratio, pages) in parked.compressed_tables {
            if let Some(pool) = self.compressed_pools.get_mut(&ratio) {
                pool.free_pages.extend(pages);
            }
        }
    }
}
