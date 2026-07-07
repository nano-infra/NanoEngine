use std::collections::HashMap;

use crate::cache::{CacheState, ParkedSession};

pub(crate) struct SessionSnapshot;

pub(crate) struct AdoptedSessionSnapshot {
    pub(crate) group_id: usize,
    pub(crate) length: i32,
    pub(crate) block_table: Option<Vec<i32>>,
    pub(crate) state_slot: i32,
    pub(crate) hisparse_slot: i32,
    pub(crate) compressed_tables: HashMap<i32, Vec<i32>>,
}

impl SessionSnapshot {
    pub(crate) fn enabled(gdn_state_cache_slots: i32, group: usize) -> bool {
        gdn_state_cache_slots > 0 && group == 1
    }

    pub(crate) fn prefix_matches(parked: &ParkedSession, full_len: i32, token_ids: &[i32]) -> bool {
        parked.length > 0
            && parked.length < full_len
            && token_ids.len() >= parked.length as usize
            && token_ids[..parked.length as usize] == parked.token_ids[..]
    }

    pub(crate) fn take_matching(
        cache: &mut CacheState,
        group: usize,
        seq_id: u64,
        dp_idx: usize,
        affinity: u64,
        full_len: i32,
        token_ids: &[i32],
    ) -> Option<AdoptedSessionSnapshot> {
        let Some(parked) = cache.parked_sessions.get(&affinity) else {
            return None;
        };
        if parked.dp_idx != dp_idx {
            return None;
        }
        if !Self::prefix_matches(parked, full_len, token_ids) {
            cache.evict_parked_by_key(affinity, group);
            return None;
        }

        let mut parked = cache.parked_sessions.remove(&affinity)?;
        cache.parked_lru.retain(|key| *key != affinity);
        cache
            .seq_assignment
            .insert(seq_id, (dp_idx, parked.group_id));

        let block_table = parked.block_tables.remove(&(parked.group_id as i32));
        if let Some(blocks) = &block_table {
            let flat = Self::flat_idx(group, dp_idx, parked.group_id);
            cache.hbm_pools[flat].insert_existing(seq_id, blocks.clone());
        }
        if parked.state_slot >= 0 {
            cache.state_slots.insert_existing(seq_id, parked.state_slot);
        }
        if parked.hisparse_slot >= 0 {
            cache
                .hisparse_slots
                .insert_existing(seq_id, parked.hisparse_slot);
        }
        let mut compressed_tables = HashMap::new();
        for (ratio, pages) in parked.compressed_tables {
            if let Some(pool) = cache.compressed_pools.get_mut(&ratio) {
                pool.seq_pages.insert(seq_id, pages.clone());
            }
            compressed_tables.insert(ratio, pages);
        }

        Some(AdoptedSessionSnapshot {
            group_id: parked.group_id,
            length: parked.length,
            block_table,
            state_slot: parked.state_slot,
            hisparse_slot: parked.hisparse_slot,
            compressed_tables,
        })
    }

    pub(crate) fn park(
        cache: &mut CacheState,
        group: usize,
        seq_id: u64,
        affinity: u64,
        token_ids: Vec<i32>,
        length: i32,
        capacity: i32,
    ) -> bool {
        let Some((dp_idx, group_id)) = cache.seq_assignment.remove(&seq_id) else {
            return false;
        };
        let flat = Self::flat_idx(group, dp_idx, group_id);
        let blocks = cache.hbm_pools[flat].take_seq_without_release(seq_id);
        let state_slot = cache.state_slots.take_without_release(seq_id).unwrap_or(-1);
        let hisparse_slot = cache
            .hisparse_slots
            .take_without_release(seq_id)
            .unwrap_or(-1);
        let mut compressed_tables = HashMap::new();
        for (ratio, pool) in cache.compressed_pools.iter_mut() {
            if let Some(pages) = pool.seq_pages.remove(&seq_id) {
                compressed_tables.insert(*ratio, pages);
            }
        }
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
        cache.insert_parked(parked, capacity, group);
        true
    }

    fn flat_idx(group: usize, dp_idx: usize, group_id: usize) -> usize {
        dp_idx * group + group_id
    }
}
