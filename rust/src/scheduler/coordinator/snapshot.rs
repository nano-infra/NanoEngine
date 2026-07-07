use pyo3::prelude::*;

use crate::cache::snapshot::session::SessionSnapshot;
use crate::cache::CacheHit;
use crate::scheduler::Scheduler;

pub(super) fn try_adopt_session_snapshot(
    scheduler: &mut Scheduler,
    py: Python<'_>,
    seq_id: u64,
    dp_idx: usize,
    batch_tokens: &[i32],
) -> PyResult<CacheHit> {
    if !SessionSnapshot::enabled(scheduler.config.gdn_state_cache_slots, scheduler.group()) {
        return Ok(CacheHit::None);
    }
    let affinity = scheduler
        .seq_table
        .get(&seq_id)
        .map(|seq| seq.affinity_key)
        .unwrap_or(0);
    if affinity == 0 {
        return Ok(CacheHit::None);
    }
    let (full_len, token_ids, seq_id) = {
        let Some(s) = scheduler.seq_table.get(&seq_id) else {
            return Ok(CacheHit::None);
        };
        (scheduler.prompt_target(s), s.token_ids.clone(), s.seq_id)
    };
    let budget = (scheduler.config.max_num_batched_tokens.max(1)
        - batch_tokens.first().copied().unwrap_or(0))
    .max(0);
    if budget <= 0 {
        return Ok(CacheHit::None);
    }

    let group = scheduler.group();
    let Some(snapshot) = SessionSnapshot::take_matching(
        &mut scheduler.cache,
        group,
        seq_id,
        dp_idx,
        affinity,
        full_len,
        &token_ids,
    ) else {
        return Ok(CacheHit::None);
    };

    if let Some(blocks) = &snapshot.block_table {
        let Some(s) = scheduler.seq_table.get_mut(&seq_id) else {
            return Ok(CacheHit::None);
        };
        let group_id = snapshot.group_id as i32;
        s.active_group_id = group_id;
        s.active_block_table = blocks.clone();
        s.active_block_tables.insert(group_id, blocks.clone());
        s.migrate_group_id = group_id;
        s.migrate_block_table = blocks.clone();
        s.migrate_block_tables.insert(group_id, blocks.clone());
    }
    if snapshot.state_slot >= 0 {
        if let Some(s) = scheduler.seq_table.get_mut(&seq_id) {
            s.active_state_slot = snapshot.state_slot;
            s.migrate_state_slot = snapshot.state_slot;
        }
    }
    if snapshot.hisparse_slot >= 0 {
        if let Some(s) = scheduler.seq_table.get_mut(&seq_id) {
            s.active_hisparse_slot = snapshot.hisparse_slot;
            s.migrate_hisparse_slot = snapshot.hisparse_slot;
        }
    }
    for (ratio, pages) in snapshot.compressed_tables {
        if let Some(s) = scheduler.seq_table.get_mut(&seq_id) {
            s.active_compressed_block_tables
                .insert(ratio, pages.clone());
            s.migrate_compressed_block_tables.insert(ratio, pages);
        }
    }

    let new_tokens = (full_len - snapshot.length).min(budget).max(0);
    let chunk_end = snapshot.length + new_tokens;
    let dispatch = scheduler.dispatch_for_master(snapshot.group_id, chunk_end);
    {
        let Some(s) = scheduler.seq_table.get_mut(&seq_id) else {
            return Ok(CacheHit::None);
        };
        s.num_cached_tokens = snapshot.length;
        s.prefill_start_offset = snapshot.length;
        s.num_tokens = chunk_end;
        s.active_dp_idx = dp_idx as i32;
        s.active_group_id = snapshot.group_id as i32;
        s.active_dispatched_tokens = dispatch;
    }
    scheduler.cache_ensure_group_blocks(py, seq_id, dp_idx, snapshot.group_id, full_len, false)?;
    scheduler
        .cache
        .prefix_cached_tokens_by_seq
        .insert(seq_id, snapshot.length);
    Ok(CacheHit::Session { new_tokens })
}

pub(super) fn park_session_snapshot_or_release(
    scheduler: &mut Scheduler,
    _py: Python<'_>,
    seq_id: u64,
) {
    let (seq_id, affinity) = {
        let Some(s) = scheduler.seq_table.get(&seq_id) else {
            return;
        };
        (s.seq_id, s.affinity_key)
    };
    if !SessionSnapshot::enabled(scheduler.config.gdn_state_cache_slots, scheduler.group())
        || affinity == 0
    {
        scheduler.release_seq(seq_id);
        return;
    }
    let (token_ids, length) = {
        let Some(s) = scheduler.seq_table.get(&seq_id) else {
            return;
        };
        (s.token_ids.clone(), s.num_tokens)
    };
    let group = scheduler.group();
    let parked = SessionSnapshot::park(
        &mut scheduler.cache,
        group,
        seq_id,
        affinity,
        token_ids,
        length,
        scheduler.config.gdn_state_cache_slots,
    );
    if !parked {
        scheduler.release_seq(seq_id);
    }
}
