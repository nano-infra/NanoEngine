use pyo3::prelude::*;

use crate::cache::CacheHit;
use crate::scheduler::Scheduler;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CoordinatorKind {
    Default,
    Hisparse,
}

pub(super) trait CacheCoordinator {
    fn try_adopt_session(
        scheduler: &mut Scheduler,
        py: Python<'_>,
        seq_id: u64,
        dp_idx: usize,
        batch_tokens: &[i32],
    ) -> PyResult<CacheHit>;

    fn park_or_release(scheduler: &mut Scheduler, py: Python<'_>, seq_id: u64);

    fn adjust_prefix_cached_tokens(scheduler: &Scheduler, tokens: i32, cached: i32) -> i32;
}
