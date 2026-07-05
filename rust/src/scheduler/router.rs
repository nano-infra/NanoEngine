use super::{RoutingStrategy, Scheduler};
use pyo3::prelude::*;

impl Scheduler {
    pub(super) fn choose_assignment(&mut self) -> (usize, usize) {
        let dp = self.dp();
        let group = self.group();
        if self.routing_strategy == RoutingStrategy::LeastCache {
            let mut best = 0usize;
            let mut best_free = i32::MIN;
            for (idx, pool) in self.hbm_pools.iter().enumerate() {
                let free = pool.num_free_blocks() as i32;
                if free > best_free {
                    best = idx;
                    best_free = free;
                }
            }
            return (best / group, best % group);
        }
        let total = (dp * group).max(1);
        let idx = self.rr_cursor % total;
        self.rr_cursor = (self.rr_cursor + 1) % total;
        (idx / group, idx % group)
    }

    pub(super) fn dp_running_tokens(&self, _py: Python<'_>, dp_idx: usize) -> i32 {
        self.running
            .get(dp_idx)
            .into_iter()
            .flatten()
            .chain(self.prefilling.get(dp_idx).into_iter().flatten())
            .filter_map(|seq_id| self.seq_table.get(seq_id))
            .map(|seq| seq.num_tokens)
            .sum()
    }

    pub(super) fn dp_running_seqs(&self, dp_idx: usize) -> i32 {
        self.running.get(dp_idx).map(|v| v.len()).unwrap_or(0) as i32
            + self.prefilling.get(dp_idx).map(|v| v.len()).unwrap_or(0) as i32
    }

    pub(super) fn route_candidates(&mut self, py: Python<'_>, seq_id: u64) -> Vec<usize> {
        let dp = self.dp();
        let affinity = self
            .seq_table
            .get(&seq_id)
            .map(|seq| seq.affinity_key)
            .unwrap_or(0);

        let mut order: Vec<usize> = (0..dp).collect();
        match self.routing_strategy {
            x if x == RoutingStrategy::RoundRobin => {
                let start = self.rr_cursor % dp.max(1);
                self.rr_cursor = (self.rr_cursor + 1) % dp.max(1);
                order.sort_by_key(|rank| (*rank + dp - start) % dp);
            }
            x if x == RoutingStrategy::LeastBatch => {
                order.sort_by_key(|rank| self.dp_running_seqs(*rank));
            }
            x if x == RoutingStrategy::LeastCache => {
                order.sort_by_key(|rank| self.dp_running_tokens(py, *rank));
            }
            x if x == RoutingStrategy::SessionPrefix => {
                order.sort_by_key(|rank| self.dp_running_tokens(py, *rank));
                if affinity != 0 {
                    if let Some(preferred) = self.session_affinity.get(&affinity).copied() {
                        let waited = self.session_wait.entry(seq_id).or_insert(0);
                        if *waited < 3 {
                            *waited += 1;
                            return vec![preferred.min(dp.saturating_sub(1))];
                        }
                        order.retain(|rank| *rank != preferred);
                        order.insert(0, preferred.min(dp.saturating_sub(1)));
                    } else if let Some(parked) = self.parked_sessions.get(&affinity) {
                        order.retain(|rank| *rank != parked.dp_idx);
                        order.insert(0, parked.dp_idx.min(dp.saturating_sub(1)));
                    }
                }
            }
            _ => {}
        }
        order
    }
}
