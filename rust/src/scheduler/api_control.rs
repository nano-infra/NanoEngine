use super::*;

impl Scheduler {
    pub(super) fn num_waiting_migration_api(&self) -> i32 {
        self.waiting_migration.len() as i32
    }

    pub(super) fn set_session_cache_slots_api(&mut self, capacity: i32) {
        self.config.gdn_state_cache_slots = capacity.max(0);
    }

    pub(super) fn set_prefix_caching_enabled_api(&mut self, enabled: bool) {
        let enabled = enabled && self.prefix_caching_allowed;
        self.prefix_caching_enabled = enabled;
        for pool in &mut self.hbm_pools {
            pool.set_prefix_caching_enabled(enabled);
        }
        for pool in &mut self.host_pools {
            pool.set_prefix_caching_enabled(enabled);
        }
    }

    pub(super) fn num_parked_sessions_api(&self) -> i32 {
        self.parked_sessions.len() as i32
    }

    pub(super) fn parked_session_keys_api(&self) -> Vec<u64> {
        self.parked_lru.clone()
    }

    pub(super) fn clear_session_cache_api(&mut self) {
        let keys = self.parked_lru.clone();
        for key in keys {
            self.evict_parked_by_key(key);
        }
    }

    pub(super) fn is_finished_api(&self) -> bool {
        self.waiting.is_empty()
            && self.waiting_migration.is_empty()
            && self.to_be_migrated.is_empty()
            && self.running.iter().all(|seqs| seqs.is_empty())
            && self.prefilling.iter().all(|seqs| seqs.is_empty())
    }

    pub(super) fn num_waiting_api(&self) -> i32 {
        self.waiting.len() as i32
    }

    pub(super) fn prefix_cached_tokens_api(&self, seq_id: u64) -> i32 {
        self.prefix_cached_tokens_impl(seq_id)
    }

    pub(super) fn clear_finished_metric_state_api(&mut self, seq_id: u64) {
        self.clear_finished_metric_state_impl(seq_id)
    }

    pub(super) fn abort_api(&mut self, seq_id: u64) -> bool {
        self.abort_impl(seq_id)
    }

    pub(super) fn abort_many_api(&mut self, seq_ids: Vec<u64>) -> Vec<u64> {
        self.abort_many_impl(seq_ids)
    }

    pub(super) fn free_to_be_migrated_ids_api(&mut self, py: Python<'_>, seq_ids: Vec<u64>) {
        self.free_to_be_migrated_ids_impl(py, seq_ids)
    }
}
