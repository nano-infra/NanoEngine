use super::*;

impl Scheduler {
    pub(super) fn num_waiting_migration_api(&self) -> i32 {
        self.waiting_migration.len() as i32
    }

    pub(super) fn set_prefix_caching_enabled_api(&mut self, enabled: bool) {
        self.cache.set_prefix_caching_enabled(enabled);
    }

    pub(super) fn is_finished_api(&self) -> bool {
        self.waiting.is_empty()
            && self.waiting_migration.is_empty()
            && self.to_be_migrated.is_empty()
            && self.running.iter().all(|seqs| seqs.is_empty())
            && self.prefilling.iter().all(|seqs| seqs.is_empty())
    }

    pub(super) fn has_runnable_work_api(&self) -> bool {
        // A prefill request remains in `to_be_migrated` until the decode
        // engine acknowledges migration and sends /pd/free. It still owns KV
        // blocks, but it must not trigger another model forward while waiting
        // for that control-plane acknowledgement.
        !self.waiting.is_empty()
            || !self.waiting_migration.is_empty()
            || self.running.iter().any(|seqs| !seqs.is_empty())
            || self.prefilling.iter().any(|seqs| !seqs.is_empty())
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
