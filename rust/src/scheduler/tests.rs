use super::*;
use crate::config::CachePlan;
use crate::metrics::ServerMetric;
use crate::sampling::SamplingParams;

fn make_scheduler() -> Scheduler {
    make_scheduler_with_flags(1)
}

fn make_scheduler_with_flags(flags: u32) -> Scheduler {
    make_scheduler_with_flags_and_prefix_cache(flags, true)
}

fn make_scheduler_with_flags_and_prefix_cache(flags: u32, enable_prefix_cache: bool) -> Scheduler {
    make_scheduler_with_flags_prefix_cache_and_batch_tokens(flags, enable_prefix_cache, 64)
}

fn make_scheduler_with_flags_prefix_cache_and_batch_tokens(
    flags: u32,
    enable_prefix_cache: bool,
    max_num_batched_tokens: i32,
) -> Scheduler {
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 0,
        max_num_seqs: 8,
        max_num_batched_tokens,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 16,
        num_host_kvcache_blocks: 0,
        kvcache_block_size: 4,
        mode: "hybrid".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache,
        cache_plan: CachePlan::new(flags),
    })
}

fn make_mtp_scheduler_with_flags(flags: u32) -> Scheduler {
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 5,
        max_num_seqs: 8,
        max_num_batched_tokens: 64,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 16,
        num_host_kvcache_blocks: 0,
        kvcache_block_size: 4,
        mode: "prefill".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache: false,
        cache_plan: CachePlan::new(flags),
    })
}

fn make_scheduler_with_host_blocks() -> Scheduler {
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 0,
        max_num_seqs: 8,
        max_num_batched_tokens: 64,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 16,
        num_host_kvcache_blocks: 16,
        kvcache_block_size: 4,
        mode: "hybrid".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache: true,
        cache_plan: CachePlan::new(1),
    })
}

fn make_single_slot_linear_scheduler() -> Scheduler {
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 0,
        max_num_seqs: 1,
        max_num_batched_tokens: 64,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 16,
        num_host_kvcache_blocks: 0,
        kvcache_block_size: 4,
        mode: "hybrid".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache: false,
        cache_plan: CachePlan::new(1 << 2),
    })
}

fn make_scheduler_with_host_prefix_cache() -> Scheduler {
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 0,
        max_num_seqs: 8,
        max_num_batched_tokens: 64,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 3,
        num_host_kvcache_blocks: 8,
        kvcache_block_size: 4,
        mode: "hybrid".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache: true,
        cache_plan: CachePlan::new(1),
    })
}

fn make_scheduler_with_gqa_hisparse_tail(tail_tokens: i32) -> Scheduler {
    let mut plan = CachePlan::new((1 << 0) | (1 << 6));
    plan.hisparse.swap_in_block_size = tail_tokens;
    Scheduler::new(SchedulerConfig {
        engine_id: "engine".to_string(),
        num_speculative_tokens: 0,
        max_num_seqs: 8,
        max_num_batched_tokens: 64,
        max_model_len: 128,
        eos_ids: Vec::new(),
        attention_dp: 1,
        group_size: 1,
        num_kvcache_blocks: 16,
        num_host_kvcache_blocks: 0,
        kvcache_block_size: 4,
        mode: "hybrid".to_string(),
        routing_strategy: RoutingStrategy::RoundRobin,
        gdn_state_cache_slots: 0,
        enable_prefix_cache: true,
        cache_plan: plan,
    })
}

fn add_tokens(
    py: Python<'_>,
    scheduler: &mut Scheduler,
    seq_id: u64,
    tokens: Vec<i32>,
) -> PyResult<()> {
    let sampling = Py::new(py, SamplingParams::new(1.0, 16, false, false))?;
    scheduler.add_request(py, seq_id, tokens, sampling, 0, None)?;
    Ok(())
}

fn run_one_prefill(py: Python<'_>, scheduler: &mut Scheduler) -> PyResult<Vec<u64>> {
    let scheduled = scheduler.schedule_prefill(py)?;
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].len(), 1);
    let seqs = scheduled[0].clone();
    scheduler.postprocess_impl(py, vec![seqs.clone()], vec![vec![Vec::new()]], None);
    Ok(seqs)
}

fn run_prefill_until_ready(py: Python<'_>, scheduler: &mut Scheduler) -> PyResult<u64> {
    let mut seq_id = None;
    loop {
        let scheduled = scheduler.schedule_prefill(py)?;
        let Some(id) = scheduled.first().and_then(|seqs| seqs.first()).copied() else {
            break;
        };
        seq_id = Some(id);
        scheduler.postprocess_impl(py, vec![vec![id]], vec![vec![Vec::new()]], None);
        let target = {
            let seq = &scheduler.seq_table[&id];
            scheduler.prompt_target(seq)
        };
        if scheduler.seq_table[&id].num_tokens >= target {
            break;
        }
    }
    seq_id.ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("no prefill scheduled"))
}

#[test]
fn pending_migration_is_retained_but_not_runnable() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 42, vec![1, 2, 3]).unwrap();
        assert!(scheduler.has_runnable_work_api());

        scheduler.waiting.clear();
        scheduler.to_be_migrated.insert(42, 0);

        assert!(!scheduler.is_finished_api());
        assert!(!scheduler.has_runnable_work_api());

        scheduler.free_to_be_migrated_ids_impl(py, vec![42]);
        assert!(scheduler.is_finished_api());
    });
}

#[test]
fn exhausted_linear_state_slot_keeps_request_waiting() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_single_slot_linear_scheduler();
        add_tokens(py, &mut scheduler, 1, vec![1, 2, 3]).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![1]);
        add_tokens(py, &mut scheduler, 2, vec![4, 5, 6]).unwrap();

        let scheduled = scheduler.schedule_prefill(py).unwrap();
        assert!(scheduled[0].is_empty());
        assert_eq!(scheduler.waiting, vec![2]);
    });
}

#[test]
fn preempted_sequence_restores_from_host_without_prefill() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_host_blocks();
        add_tokens(py, &mut scheduler, 42, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![42]);

        scheduler.preempt_impl(py, 0, 42).unwrap();
        assert_eq!(scheduler.cache.pending_swap_out_tasks[0].len(), 1);
        let swap_out = scheduler.cache.pending_swap_out_tasks.clone();
        scheduler.complete_host_swap_outs_impl(swap_out);
        assert!(scheduler.cache_seq_has_host_blocks(42));

        scheduler.current_step += 3;
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        assert!(scheduled[0].is_empty());
        assert_eq!(scheduler.cache.pending_swap_in_tasks[0].len(), 1);
        assert_eq!(scheduler.running[0], vec![42]);

        let swap_in = scheduler.cache.pending_swap_in_tasks.clone();
        scheduler.complete_host_swap_ins_impl(swap_in);
        assert!(!scheduler.cache_seq_has_host_blocks(42));
    });
}

#[test]
fn host_swap_preempted_sequence_waits_behind_new_request() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_host_blocks();
        add_tokens(py, &mut scheduler, 10, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![10]);
        add_tokens(py, &mut scheduler, 11, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![11]);
        add_tokens(py, &mut scheduler, 12, (0..8).collect()).unwrap();

        scheduler.preempt_impl(py, 0, 10).unwrap();

        assert_eq!(scheduler.waiting.first().copied(), Some(12));
        assert_eq!(scheduler.waiting.last().copied(), Some(10));
        assert_eq!(scheduler.cache.pending_swap_out_tasks[0].len(), 1);
    });
}

#[test]
fn host_swap_restore_respects_cooldown_and_capacity() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_host_blocks();
        add_tokens(py, &mut scheduler, 42, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![42]);

        scheduler.current_step = 10;
        scheduler.preempt_impl(py, 0, 42).unwrap();
        let swap_out = scheduler.cache.pending_swap_out_tasks.clone();
        scheduler.complete_host_swap_outs_impl(swap_out);

        assert!(!scheduler.cache_try_restore_host_blocks(42, 0).unwrap());
        scheduler.current_step += 3;
        assert!(scheduler.cache_try_restore_host_blocks(42, 0).unwrap());
    });
}

#[test]
fn host_swap_victim_uses_oldest_scheduled_sequence() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_host_blocks();
        add_tokens(py, &mut scheduler, 20, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![20]);
        add_tokens(py, &mut scheduler, 21, (0..8).collect()).unwrap();
        assert_eq!(run_one_prefill(py, &mut scheduler).unwrap(), vec![21]);

        scheduler
            .seq_table
            .get_mut(&20)
            .unwrap()
            .last_scheduled_step = 100;
        scheduler
            .seq_table
            .get_mut(&21)
            .unwrap()
            .last_scheduled_step = 10;

        assert!(scheduler.preempt_one_for_allocation(py, 0, 999).unwrap());
        assert!(scheduler.cache_seq_has_host_blocks(21));
        assert!(!scheduler.cache_seq_has_host_blocks(20));
    });
}

#[test]
fn evicted_device_prefix_writes_back_and_promotes_from_host() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_host_prefix_cache();
        let prefix: Vec<i32> = (0..8).collect();
        add_tokens(py, &mut scheduler, 30, prefix.clone()).unwrap();
        let a = run_one_prefill(py, &mut scheduler).unwrap()[0];
        scheduler.release_seq(a);

        add_tokens(py, &mut scheduler, 31, (100..112).collect()).unwrap();
        let b = run_one_prefill(py, &mut scheduler).unwrap()[0];
        assert_eq!(scheduler.cache.pending_swap_out_tasks[0].len(), 2);
        let swap_out = scheduler.cache.pending_swap_out_tasks.clone();
        scheduler.complete_host_swap_outs_impl(swap_out);
        scheduler.release_seq(b);

        let mut round_two = prefix.clone();
        round_two.push(999);
        add_tokens(py, &mut scheduler, 32, round_two).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        assert_eq!(scheduled[0].len(), 1);
        let seq_id = scheduled[0][0];
        assert_eq!(scheduler.seq_table[&seq_id].num_cached_tokens, 8);
        assert_eq!(scheduler.cache.pending_swap_in_tasks[0].len(), 2);
    });
}

fn server_metric() -> ServerMetric {
    ServerMetric {
        total_tokens: 0,
        total_prompt_tokens: 0,
        total_generated_tokens: 0,
        num_running_requests: 0,
        num_waiting_requests: 0,
        num_waiting_migration_requests: 0,
        num_completed_requests: 0,
        num_waiting_head_blocks: 0,
        num_waiting_total_blocks: 0,
        prefill_throughput_samples: Vec::new(),
        decode_throughput_samples: Vec::new(),
        token_usage_by_dp: std::collections::HashMap::new(),
        group_send_request_counts: std::collections::HashMap::new(),
        group_recv_request_counts: std::collections::HashMap::new(),
        start_time: 0.0,
    }
}

#[test]
fn chunked_prefill_keeps_prefix_hit_separate_from_chunk_offset() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags_prefix_cache_and_batch_tokens(1, true, 4);
        add_tokens(py, &mut scheduler, 100, (0..12).collect()).unwrap();
        let first = run_prefill_until_ready(py, &mut scheduler).unwrap();
        scheduler.release_seq(first);

        add_tokens(py, &mut scheduler, 101, (0..12).collect()).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let second = scheduled[0][0];
        let seq = &scheduler.seq_table[&second];
        assert_eq!(seq.num_cached_tokens, 11);
        assert_eq!(seq.prefill_start_offset, 11);
        assert_eq!(seq.num_tokens, 12);

        let result = ScheduleResult {
            dp_seq_ids: vec![vec![second]],
            is_prefill: true,
            ..ScheduleResult::default()
        };
        let mut metric = server_metric();
        let snapshot = scheduler.record_step_metric_impl(py, &mut metric, &result, None);
        assert_eq!(snapshot.prefill_tokens_per_dp, vec![1]);
        assert_eq!(snapshot.prefix_cached_tokens_per_dp, vec![11]);
        assert_eq!(snapshot.prefix_prompt_tokens_per_dp, vec![12]);
    });
}

#[test]
fn gqa_hisparse_prefix_hit_recomputes_sliding_tail() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_gqa_hisparse_tail(4);
        add_tokens(py, &mut scheduler, 110, (0..12).collect()).unwrap();
        let first = run_prefill_until_ready(py, &mut scheduler).unwrap();
        scheduler.release_seq(first);

        add_tokens(py, &mut scheduler, 111, (0..13).collect()).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let second = scheduled[0][0];
        let seq = &scheduler.seq_table[&second];
        assert_eq!(seq.num_cached_tokens, 9);
        assert_eq!(seq.prefill_start_offset, 9);
        assert_eq!(seq.num_tokens, 13);
    });
}

#[test]
fn prefix_cache_survives_decode_growth_and_release() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags_prefix_cache_and_batch_tokens(1, true, 4);
        add_tokens(py, &mut scheduler, 100, (0..12).collect()).unwrap();
        let first = run_prefill_until_ready(py, &mut scheduler).unwrap();

        scheduler.seq_table.get_mut(&first).unwrap().last_token = 11;
        scheduler
            .cache_ensure_blocks_for_seq(py, first, false)
            .unwrap();
        scheduler
            .seq_table
            .get_mut(&first)
            .unwrap()
            .token_ids
            .push(12);
        scheduler.seq_table.get_mut(&first).unwrap().num_tokens = 13;
        scheduler.release_seq(first);

        add_tokens(py, &mut scheduler, 101, (0..12).collect()).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let second = scheduled[0][0];
        assert_eq!(scheduler.seq_table[&second].num_cached_tokens, 11);
    });
}

#[test]
fn scheduler_reuses_hbm_prefix_after_release() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
        let first = run_one_prefill(py, &mut scheduler).unwrap();
        let first_table = scheduler.seq_table[&first[0]].active_block_table.clone();
        scheduler.release_seq(first[0]);

        add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
        let second = run_one_prefill(py, &mut scheduler).unwrap();
        assert_eq!(
            scheduler.seq_table[&second[0]].active_block_table,
            first_table
        );
        assert_eq!(scheduler.prefix_cached_tokens_impl(101), 7);
    });
}

#[test]
fn scheduler_prefix_cache_can_be_disabled() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
        let first = run_one_prefill(py, &mut scheduler).unwrap();
        scheduler.release_seq(first[0]);

        scheduler.set_prefix_caching_enabled(false);
        add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
        run_one_prefill(py, &mut scheduler).unwrap();
        assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
    });
}

#[test]
fn step_metric_reports_prefix_cache_hits_per_dp() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
        let first = run_one_prefill(py, &mut scheduler).unwrap();
        scheduler.release_seq(first[0]);

        add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        assert_eq!(scheduled[0].len(), 1);
        let result = ScheduleResult {
            dp_seq_ids: vec![vec![scheduled[0][0]]],
            is_prefill: true,
            ..ScheduleResult::default()
        };
        let mut metric = server_metric();
        let snapshot = scheduler.record_step_metric_impl(py, &mut metric, &result, None);
        assert_eq!(snapshot.prefill_tokens_per_dp, vec![1]);
        assert_eq!(snapshot.prefix_cached_tokens_per_dp, vec![7]);
        assert_eq!(snapshot.prefix_prompt_tokens_per_dp, vec![8]);
    });
}

#[test]
fn gdn_scheduler_never_enables_prefix_cache() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags(1 << 2);
        scheduler.set_prefix_caching_enabled(true);
        add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
        let first = run_one_prefill(py, &mut scheduler).unwrap();
        scheduler.release_seq(first[0]);

        add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
        run_one_prefill(py, &mut scheduler).unwrap();
        assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
    });
}

#[test]
fn scheduler_config_can_disable_prefix_cache() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags_and_prefix_cache(1, false);
        scheduler.set_prefix_caching_enabled(true);
        add_tokens(py, &mut scheduler, 100, (0..8).collect()).unwrap();
        let first = run_one_prefill(py, &mut scheduler).unwrap();
        scheduler.release_seq(first[0]);

        add_tokens(py, &mut scheduler, 101, (0..8).collect()).unwrap();
        run_one_prefill(py, &mut scheduler).unwrap();
        assert_eq!(scheduler.prefix_cached_tokens_impl(101), 0);
    });
}

#[test]
fn prefill_postprocess_preserves_existing_running_sequences() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 100, vec![1, 2, 3]).unwrap();
        run_one_prefill(py, &mut scheduler).unwrap();
        assert_eq!(scheduler.running[0].len(), 1);
        assert_eq!(scheduler.running[0][0], 100);

        add_tokens(py, &mut scheduler, 101, vec![4, 5, 6]).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        assert_eq!(scheduled[0].len(), 1);
        assert_eq!(scheduled[0][0], 101);
        scheduler.postprocess_impl(
            py,
            vec![vec![scheduled[0][0]]],
            vec![vec![Vec::new()]],
            None,
        );

        let running_ids = scheduler.running[0].clone();
        assert_eq!(running_ids, vec![100, 101]);
    });
}

#[test]
fn runner_out_postprocess_and_metrics_entrypoints_work() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        add_tokens(py, &mut scheduler, 100, vec![1, 2, 3]).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let seq_id = scheduled[0][0];
        let runner_out = RunnerOut {
            token_ids: vec![vec![4]],
            logprobs: Some(vec![vec![0.25]]),
            server_handler_ns: 0,
        };
        scheduler.postprocess_runner_outs(py, vec![vec![seq_id]], vec![runner_out.clone()], None);
        assert_eq!(scheduler.seq_table[&seq_id].last_token, 4);
        assert_eq!(scheduler.seq_table[&seq_id].completion_logprobs, vec![0.25]);

        let result = ScheduleResult {
            dp_seq_ids: vec![vec![seq_id]],
            is_prefill: false,
            ..ScheduleResult::default()
        };
        let mut metric = server_metric();
        let snapshot =
            scheduler.record_step_metric_runner_outs(py, &mut metric, &result, vec![runner_out]);
        assert_eq!(snapshot.decode_tokens, 1);
        assert_eq!(snapshot.decode_tokens_per_dp, vec![1]);
    });
}

#[test]
fn recurrent_mtp_allocates_and_reuses_state_slot_for_mla_only_plan() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_mtp_scheduler_with_flags(1 << 1);
        add_tokens(py, &mut scheduler, 710, vec![1, 2, 3]).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let seq_id = scheduled[0][0];
        let slot = scheduler.seq_table[&seq_id].active_state_slot;

        assert!(slot >= 0);
        assert_eq!(scheduler.seq_table[&seq_id].migrate_state_slot, slot);
        assert_eq!(scheduler.cache.state_slot(seq_id), Some(slot));

        assert!(scheduler.abort_impl(seq_id));
        add_tokens(py, &mut scheduler, 711, vec![4, 5, 6]).unwrap();
        let next = scheduler.schedule_prefill(py).unwrap()[0][0];
        assert_eq!(scheduler.seq_table[&next].active_state_slot, slot);
        assert_eq!(scheduler.seq_table[&next].migrate_state_slot, slot);
    });
}

#[test]
fn speculative_decode_reserves_all_recurrent_draft_positions() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags_and_prefix_cache(1, false);
        scheduler.config.num_speculative_tokens = 5;
        add_tokens(py, &mut scheduler, 700, vec![1, 2, 3]).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let seq_id = scheduled[0][0];

        // The predictor runs during prefill, before postprocess. With
        // block_size=4 and num_tokens=3, N=5 must already reserve through
        // token 8 and expose two pages in this first RunnerIn.
        assert_eq!(scheduler.seq_table[&seq_id].active_block_table.len(), 2);
    });
}

#[test]
fn decode_migration_admission_reserves_remote_mtp_lookahead() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_mtp_scheduler_with_flags(0);
        scheduler.config.mode = "decode".to_string();
        add_tokens(py, &mut scheduler, 704, vec![1, 2, 3]).unwrap();
        let scheduled = scheduler.schedule_prefill(py).unwrap();
        let seq_id = scheduled[0][0];

        // Remote prefill migrates prompt + N positions. Decode admission must
        // allocate the same two pages before receiving them; allocating only
        // the three prompt tokens would leave a one-page-short destination.
        assert_eq!(scheduler.seq_table[&seq_id].active_block_table.len(), 2);
    });
}

#[test]
fn speculative_decode_reserves_verify_and_next_draft_windows() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler_with_flags_and_prefix_cache(1, false);
        scheduler.config.num_speculative_tokens = 5;
        add_tokens(py, &mut scheduler, 703, vec![1, 2, 3]).unwrap();
        let seq_id = run_one_prefill(py, &mut scheduler).unwrap()[0];

        // Before acceptance is known, decode must cover the current K=6
        // verify and the following recurrent draft line. With block_size=4,
        // 3 + 2*5 tokens require four pages.
        assert_eq!(scheduler.seq_table[&seq_id].active_block_table.len(), 4);
    });
}

#[test]
fn speculative_bundle_stops_at_first_eos_and_metrics_count_applied_tokens() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        scheduler.config.eos_ids = vec![9];
        let sampling = Py::new(py, SamplingParams::new(0.8, 16, false, true)).unwrap();
        scheduler
            .add_request(py, 701, vec![1, 2, 3], sampling, 0, None)
            .unwrap();
        let seq_id = run_one_prefill(py, &mut scheduler).unwrap()[0];

        scheduler.postprocess_impl(
            py,
            vec![vec![seq_id]],
            vec![vec![vec![4, 9, 7, 8]]],
            Some(vec![vec![vec![-0.1, -0.2, -0.3, -0.4]]]),
        );

        let seq = &scheduler.seq_table[&seq_id];
        assert_eq!(seq.token_ids, vec![1, 2, 3, 4, 9]);
        assert_eq!(seq.completion_logprobs, vec![-0.1, -0.2]);
        assert_eq!(seq.status, 2);
        assert_eq!(scheduler.last_step_token_ids[&seq_id], vec![4, 9]);

        let result = ScheduleResult {
            dp_group_seq_ids: vec![vec![seq_id]],
            is_prefill: false,
            ..ScheduleResult::default()
        };
        let mut metric = server_metric();
        let snapshot = scheduler.record_step_metric_impl(
            py,
            &mut metric,
            &result,
            Some(vec![vec![vec![4, 9, 7, 8]]]),
        );
        assert_eq!(snapshot.decode_tokens, 2);
        assert_eq!(snapshot.decode_tokens_per_dp, vec![2]);
    });
}

#[test]
fn speculative_bundle_stops_at_request_token_limit() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let mut scheduler = make_scheduler();
        let sampling = Py::new(py, SamplingParams::new(0.0, 2, true, false)).unwrap();
        scheduler
            .add_request(py, 702, vec![1, 2, 3], sampling, 0, None)
            .unwrap();
        let seq_id = run_one_prefill(py, &mut scheduler).unwrap()[0];

        scheduler.postprocess_impl(
            py,
            vec![vec![seq_id]],
            vec![vec![vec![4, 5, 6]]],
            None,
        );

        assert_eq!(scheduler.seq_table[&seq_id].token_ids, vec![1, 2, 3, 4, 5]);
        assert_eq!(scheduler.last_step_token_ids[&seq_id], vec![4, 5]);
        assert_eq!(scheduler.seq_table[&seq_id].status, 2);
    });
}
