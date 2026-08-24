use std::collections::{HashMap, HashSet};

use crate::cache::table::block::{BlockPool, CompressedPool};
use crate::cache::table::slot::SlotPool;
use crate::scheduler::SchedulerConfig;

#[derive(Clone, Debug)]
pub(crate) struct PendingHostSwap {
    pub(crate) dp_idx: usize,
    pub(crate) group_id: usize,
}

pub(crate) struct CacheState {
    pub(crate) hbm_pools: Vec<BlockPool>,
    pub(crate) host_pools: Vec<BlockPool>,
    pub(crate) pending_host_swaps: HashMap<u64, PendingHostSwap>,
    pub(crate) pending_swap_out_tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    pub(crate) pending_swap_in_tasks: Vec<Vec<(u64, Vec<i32>, Vec<i32>)>>,
    pub(crate) seq_assignment: HashMap<u64, (usize, usize)>,
    pub(crate) session_affinity: HashMap<u64, usize>,
    pub(crate) session_wait: HashMap<u64, i32>,
    pub(crate) state_slots: Vec<SlotPool>,
    pub(crate) hisparse_slots: Vec<SlotPool>,
    pub(crate) compressed_pools: HashMap<i32, CompressedPool>,
    pub(crate) prefix_caching_allowed: bool,
    pub(crate) prefix_caching_enabled: bool,
    pub(crate) prefix_cached_tokens_by_seq: HashMap<u64, i32>,
    pub(crate) prefix_counted_seq_ids: HashSet<u64>,
}

impl CacheState {
    pub(crate) fn new(config: &SchedulerConfig, dp: usize, group: usize) -> Self {
        let num_blocks = config.num_kvcache_blocks.max(0);
        let prefix_caching_allowed =
            config.enable_prefix_cache && config.cache_plan.flags & (1 << 2) == 0;
        let mut hbm_pools = (0..(dp * group))
            .map(|_| BlockPool::new(num_blocks, config.kvcache_block_size))
            .collect::<Vec<_>>();
        let mut host_pools = (0..(dp * group))
            .map(|_| {
                BlockPool::new(
                    config.num_host_kvcache_blocks.max(0),
                    config.kvcache_block_size,
                )
            })
            .collect::<Vec<_>>();
        if !prefix_caching_allowed {
            for pool in &mut hbm_pools {
                pool.set_prefix_caching_enabled(false);
            }
        }
        for pool in &mut host_pools {
            pool.set_prefix_caching_enabled(false);
        }

        let mut compressed_pools = HashMap::new();
        if config.cache_plan.flags & ((1 << 3) | (1 << 4)) != 0 {
            for spec in [
                (
                    config.cache_plan.hca.compression_ratio,
                    config.cache_plan.hca.num_pages,
                    config.cache_plan.hca.page_size,
                    config.cache_plan.hca.max_blocks_per_seq,
                ),
                (
                    config.cache_plan.csa.compression_ratio,
                    config.cache_plan.csa.num_pages,
                    config.cache_plan.csa.page_size,
                    config.cache_plan.csa.max_blocks_per_seq,
                ),
            ] {
                let (ratio, num_pages, page_size, max_blocks_per_seq) = spec;
                if ratio > 0 && num_pages > 0 && page_size > 0 && max_blocks_per_seq > 0 {
                    compressed_pools.insert(
                        ratio,
                        CompressedPool {
                            ratio,
                            page_size,
                            max_blocks_per_seq,
                            free_pages: (0..num_pages).rev().collect(),
                            seq_pages: HashMap::new(),
                        },
                    );
                }
            }
        }

        // Recurrent MTP PD handoff reuses the same stable per-sequence slot
        // identity as GDN/DSv4 state. The payload lives in a separate MR, so
        // allocating slots here does not couple MTP to those model caches.
        let needs_mtp_handoff = config.num_speculative_tokens > 1 && config.mode != "hybrid";
        let state_slots = if needs_mtp_handoff
            || config.cache_plan.flags & ((1 << 2) | (1 << 3) | (1 << 4)) != 0
        {
            config.cache_plan.gdn.state_slots.max(config.max_num_seqs)
        } else {
            0
        };
        let hisparse_slots = if config.cache_plan.flags & (1 << 6) != 0 {
            config
                .cache_plan
                .hisparse
                .max_num_seqs
                .max(config.max_num_seqs)
        } else {
            0
        };

        Self {
            hbm_pools,
            host_pools,
            pending_host_swaps: HashMap::new(),
            pending_swap_out_tasks: (0..(dp * group)).map(|_| Vec::new()).collect(),
            pending_swap_in_tasks: (0..(dp * group)).map(|_| Vec::new()).collect(),
            seq_assignment: HashMap::new(),
            session_affinity: HashMap::new(),
            session_wait: HashMap::new(),
            state_slots: (0..dp).map(|_| SlotPool::new(state_slots)).collect(),
            hisparse_slots: (0..dp)
                .map(|_| SlotPool::new(hisparse_slots))
                .collect(),
            compressed_pools,
            prefix_caching_allowed,
            prefix_caching_enabled: prefix_caching_allowed,
            prefix_cached_tokens_by_seq: HashMap::new(),
            prefix_counted_seq_ids: HashSet::new(),
        }
    }

    pub(crate) fn set_prefix_caching_enabled(&mut self, enabled: bool) {
        let enabled = enabled && self.prefix_caching_allowed;
        self.prefix_caching_enabled = enabled;
        for pool in &mut self.hbm_pools {
            pool.set_prefix_caching_enabled(enabled);
        }
        for pool in &mut self.host_pools {
            pool.set_prefix_caching_enabled(enabled);
        }
    }

    pub(crate) fn release_seq(&mut self, seq_id: u64, group: usize) {
        if let Some(pending) = self.pending_host_swaps.remove(&seq_id) {
            let flat = Self::flat_idx(group, pending.dp_idx, pending.group_id);
            self.host_pools[flat].remove_seq(seq_id);
        }
        if let Some((dp_idx, group_id)) = self.seq_assignment.remove(&seq_id) {
            let flat = Self::flat_idx(group, dp_idx, group_id);
            self.hbm_pools[flat].remove_seq(seq_id);
        }
        for pool in &mut self.host_pools {
            pool.remove_seq(seq_id);
        }
        for pool in &mut self.state_slots {
            pool.remove(seq_id);
        }
        for pool in &mut self.hisparse_slots {
            pool.remove(seq_id);
        }
        for pool in self.compressed_pools.values_mut() {
            if let Some(mut pages) = pool.seq_pages.remove(&seq_id) {
                pool.free_pages.append(&mut pages);
            }
        }
    }

    fn flat_idx(group: usize, dp_idx: usize, group_id: usize) -> usize {
        dp_idx * group + group_id
    }
}
