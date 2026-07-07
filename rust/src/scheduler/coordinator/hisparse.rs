use pyo3::prelude::*;

use crate::cache::CacheHit;
use crate::scheduler::coordinator::snapshot::{
    park_session_snapshot_or_release, try_adopt_session_snapshot,
};
use crate::scheduler::coordinator::strategy::CacheCoordinator;
use crate::scheduler::Scheduler;

pub(super) struct HisparseCoordinator;

impl CacheCoordinator for HisparseCoordinator {
    fn try_adopt_session(
        scheduler: &mut Scheduler,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        batch_tokens: &[i32],
    ) -> PyResult<CacheHit> {
        try_adopt_session_snapshot(scheduler, py, seq_id, dp_idx, batch_tokens)
    }

    fn park_or_release(scheduler: &mut Scheduler, py: Python<'_>, seq_id: u64) {
        park_session_snapshot_or_release(scheduler, py, seq_id);
    }

    fn adjust_prefix_cached_tokens(scheduler: &Scheduler, tokens: i32, cached: i32) -> i32 {
        let has_gqa = scheduler.config.cache_plan.flags & (1 << 0) != 0;
        let has_hisparse = scheduler.config.cache_plan.flags & (1 << 6) != 0;
        if !has_gqa || !has_hisparse {
            return cached;
        }
        let tail = scheduler
            .config
            .cache_plan
            .hisparse
            .swap_in_block_size
            .max(1);
        cached.min((tokens - tail).max(0))
    }
}
