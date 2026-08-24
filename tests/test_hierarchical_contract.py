from pathlib import Path

import pytest

import nanodeploy.engine.local_scheduler as local_scheduler_module
from nanodeploy._cpp import BlockContextSlot
from nanodeploy.config import Config
from nanodeploy.engine.hierarchical_contract import (
    AddCommand,
    AdmissionReservation,
    HIERARCHICAL_LOOP_COUNT,
    LocalDecodeBatch,
    LoadSnapshot,
    RankLoad,
    RequestState,
    WorkerDecodeResult,
    round_up,
    validate_add_request,
)
from nanodeploy.engine.local_scheduler import LocalScheduler
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.topology import build_hierarchical_topology
from nanodeploy.engine.sequence import Sequence
from nanodeploy.router.admission_planner import (
    AdmissionPlanner,
    AdmissionPlannerConfig,
)
from nanodeploy.sampling_params import SamplingParams


DEEPSEEK_MODEL = Path(
    "/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3"
)


pytestmark = pytest.mark.skipif(
    not (DEEPSEEK_MODEL / "config.json").is_file(),
    reason=f"DeepSeek-V3 config not found at {DEEPSEEK_MODEL}",
)


def make_hierarchical_config(**overrides) -> Config:
    values = {
        "model": str(DEEPSEEK_MODEL),
        "scheduler_arch": "hierarchical",
        "mode": "decode",
        "dummy_prefill": True,
        "attention_dp": 2,
        "attention_sp": 4,
        "attention_tp": 1,
        "ffn_dp": 1,
        "ffn_ep": 8,
        "ffn_tp": 1,
        "kvcache_block_size": 64,
        "num_kvcache_blocks": 32,
        "max_model_len": 16384,
        "max_num_batched_tokens": 16384,
    }
    values.update(overrides)
    return Config(**values)


def test_dp2_sp4_topology_has_two_disjoint_engines():
    topology = build_hierarchical_topology(
        attention_dp=2,
        attention_sp=4,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
    )

    assert len(topology.engines) == 2
    assert topology.engine(0).global_ranks == (0, 1, 2, 3)
    assert topology.engine(1).global_ranks == (4, 5, 6, 7)
    assert topology.engine(1).engine_local_rank(6) == 2
    assert topology.engine(1).global_rank(sp_idx=3) == 7


@pytest.mark.parametrize(
    ("attention_dp", "attention_sp", "world_size"),
    [
        (8, 1, 8),
        (2, 4, 8),
        (1, 8, 8),
        (16, 1, 16),
        (2, 8, 16),
        (32, 1, 32),
        (4, 8, 32),
    ],
)
def test_complete_hierarchical_topology_whitelist(
    attention_dp, attention_sp, world_size
):
    topology = build_hierarchical_topology(
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=world_size,
        ffn_tp=1,
    )

    assert topology.world_size == world_size
    assert len(topology.engines) == attention_dp
    flattened = tuple(
        rank for engine in topology.engines for rank in engine.global_ranks
    )
    assert flattened == tuple(range(world_size))
    assert all(engine.world_size == attention_sp for engine in topology.engines)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("loop_count", 8, "loop_count=16"),
        ("mode", "hybrid", "mode='decode'"),
        ("dummy_prefill", False, "dummy_prefill=True"),
        ("ffn_ep", 4, "topology is not supported"),
        (
            "max_ingress_batch_requests",
            0,
            "max_ingress_batch_requests must be positive",
        ),
        (
            "max_ingress_drain_ms",
            -0.1,
            "max_ingress_drain_ms must be non-negative",
        ),
    ],
)
def test_hierarchical_config_rejects_out_of_contract_values(
    field, value, message
):
    with pytest.raises(ValueError, match=message):
        make_hierarchical_config(**{field: value})


def test_hierarchical_config_uses_deepseek_v3_mla_contract():
    config = make_hierarchical_config()

    assert config.hf_config.architectures == ["DeepseekV3ForCausalLM"]
    assert config.hf_config.num_key_value_heads == 1
    assert config.kvcache_block_size == 64
    assert not config.hierarchical_execution_trace
    assert not config.hierarchical_quantum_diagnostics
    assert config.hierarchical_worker_transport == "ray"
    assert config.max_ingress_batch_requests == 256
    assert config.max_ingress_drain_ms == 0.0
    assert len(config.collective_fingerprint()) == 64

    zmq_config = make_hierarchical_config(
        hierarchical_worker_transport="zmq"
    )
    assert zmq_config.collective_fingerprint() != (
        config.collective_fingerprint()
    )


def test_add_validation_rounds_completion_and_excludes_bootstrap():
    validation = validate_add_request(
        request_id=7,
        prompt_token_ids=[1, 2, 3],
        max_tokens=17,
        ignore_eos=True,
        max_model_len=36,
        vocab_size=129280,
    )

    assert round_up(17) == 32
    assert validation.original_prompt_len == 3
    assert validation.internal_prompt_len == 4
    assert validation.padded_completion_len == 32
    assert validation.total_capacity_len == 36


def test_add_validation_rejects_bad_eos_token_and_padded_length():
    with pytest.raises(ValueError, match="ignore_eos=True"):
        validate_add_request(
            request_id=1,
            prompt_token_ids=[1],
            max_tokens=1,
            ignore_eos=False,
            max_model_len=64,
            vocab_size=129280,
        )
    with pytest.raises(ValueError, match="outside"):
        validate_add_request(
            request_id=1,
            prompt_token_ids=[129280],
            max_tokens=1,
            ignore_eos=True,
            max_model_len=64,
            vocab_size=129280,
        )
    with pytest.raises(ValueError, match="padded model length"):
        validate_add_request(
            request_id=1,
            prompt_token_ids=[1, 2],
            max_tokens=17,
            ignore_eos=True,
            max_model_len=34,
            vocab_size=129280,
        )


def test_control_dummies_are_deterministic_and_have_reserved_blocks():
    config = make_hierarchical_config()
    scheduler = Scheduler(config)

    assert len(scheduler.worker_state) == 2
    for state in scheduler.worker_state:
        assert len(state.dummy_seqs) == 4
        assert state.num_control_dummy_blocks() == 4
        for sp_idx, dummy in enumerate(state.dummy_seqs):
            assert state.is_control_dummy(dummy)
            assert dummy.token_ids == [0, 0]
            assert dummy.ignore_eos
            assert dummy.block_ctx().master_sp_idx == sp_idx
            assert (
                len(dummy.block_table(BlockContextSlot.ACTIVE, sp_idx)) == 1
            )
            assert state.block_manager[sp_idx].num_free_blocks == 31


def test_admission_and_decode_are_separate_cpp_calls():
    config = make_hierarchical_config()
    scheduler = Scheduler(config)
    seq = Sequence(
        [11, 12, 13],
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=17,
            ignore_eos=True,
        ),
    )
    scheduler.add(seq)

    admitted = scheduler.admit()
    assert sum(map(len, admitted)) == 1
    assert scheduler.get_total_waiting_migration_size() == 0
    dp_idx = seq.block_ctx().dp_idx
    state = scheduler.worker_state[dp_idx]
    assert state.can_fit_lifetime(
        seq, 1 + round_up(seq.max_tokens)
    )

    decode_batches = scheduler.plan_decode()
    assert len(decode_batches) == 2
    assert seq in decode_batches[dp_idx]
    assert all(
        any(
            scheduled.block_ctx().master_sp_idx == sp_idx
            for scheduled in decode_batches[dp_idx]
        )
        for sp_idx in range(config.attention_sp)
    )


def test_frontend_admission_mirror_matches_local_cpp_placements():
    config = make_hierarchical_config(
        enable_non_uniform_split=True,
        segment_size=64,
    )
    local = LocalScheduler(
        config, config.hierarchical_topology.engine(0)
    )
    planner = AdmissionPlanner(
        AdmissionPlannerConfig.from_config(config)
    )
    shadow = planner.shadow_from_snapshot(
        local.load_snapshot(wave_id=1, quantum_id=0)
    )
    assert shadow is not None

    commands = tuple(
        AddCommand(
            request_id=request_id,
            prompt_token_ids=tuple(
                (index % 100) + 1 for index in range(prompt_len)
            ),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
        for request_id, prompt_len in enumerate(
            (1, 65, 129, 257), start=1
        )
    )
    reservations = tuple(
        planner.plan(shadow, command) for command in commands
    )
    assert all(reservation is not None for reservation in reservations)

    results = local.commit_planned_batch(
        commands,
        tuple(
            reservation
            for reservation in reservations
            if reservation is not None
        ),
    )
    assert all(result.accepted for result in results)
    running_by_id = {
        sequence.seq_id: sequence
        for sequence in local.cpp_scheduler.running(0)
    }
    actual_reservations = []
    for command in commands:
        reservation = reservations[command.request_id - 1]
        assert reservation is not None
        sequence = running_by_id[command.request_id]
        block_ctx = sequence.block_ctx(BlockContextSlot.ACTIVE)
        actual_prompt_placement = list(block_ctx.num_dispatched_tokens)
        actual_prompt_placement[block_ctx.master_sp_idx] -= 1
        actual_reservations.append(
            (
                block_ctx.master_sp_idx,
                tuple(actual_prompt_placement),
            )
        )
    assert tuple(actual_reservations) == tuple(
        (
            reservation.master_sp_idx,
            reservation.dispatched_tokens,
        )
        for reservation in reservations
        if reservation is not None
    )


def test_legacy_frontend_batch_master_reservation_is_counted_once():
    planner = AdmissionPlanner(
        AdmissionPlannerConfig(
            attention_sp=1,
            kvcache_block_size=64,
            max_num_seqs=2,
            max_num_batched_tokens=3,
            max_num_recv_seqs=1,
            reserved_blocks_per_req=1.0,
            segment_size=64,
            queue_capacity=2,
            sp_master_selector="RoundRobin",
        )
    )
    shadow = planner.shadow_from_snapshot(
        LoadSnapshot(
            engine_id=0,
            ready=True,
            waiting=0,
            running=0,
            free_blocks_min=4,
            wave_id=1,
            quantum_id=0,
            rank_loads=(
                RankLoad(
                    global_rank=0,
                    sp_idx=0,
                    tp_idx=0,
                    master_batch_size=0,
                    active_master_requests=0,
                    free_blocks=4,
                    total_blocks=4,
                    master_assignments=0,
                    mastered_decode_tokens=0,
                ),
            ),
        )
    )
    assert shadow is not None

    commands = tuple(
        AddCommand(
            request_id=request_id,
            prompt_token_ids=(1,),
            max_tokens=1,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
        for request_id in (1, 2)
    )

    first = planner.plan(shadow, commands[0])
    second = planner.plan(shadow, commands[1])

    assert first is not None
    assert second is not None
    assert shadow.master_counts == [2]
    assert shadow.batch_master_counts == [2]
    assert shadow.free_blocks == [2]


def test_planned_admission_mismatch_keeps_local_fifo_clean():
    config = make_hierarchical_config()
    local = LocalScheduler(
        config, config.hierarchical_topology.engine(0)
    )
    commands = tuple(
        AddCommand(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
        for request_id in (10, 11)
    )
    reservations = (
        AdmissionReservation(10, 0, 0, (0, 0, 0, 0)),
        AdmissionReservation(11, 0, 1, (0, 2, 0, 0)),
    )

    results = local.commit_planned_batch(commands, reservations)

    assert tuple(result.reason for result in results) == (
        "admission_state_mismatch",
        "admission_state_mismatch",
    )
    assert not local.cpp_scheduler.waiting_migration
    assert not local.cpp_scheduler.running(0)
    assert local.is_finished()


def test_planned_admission_scans_live_records_once_per_batch():
    class CountingRecords(dict):
        def __init__(self, records):
            super().__init__(records)
            self.values_calls = 0
            self.items_calls = 0
            self.yielded_records = 0

        def values(self):
            self.values_calls += 1
            for record in super().values():
                self.yielded_records += 1
                yield record

        def items(self):
            self.items_calls += 1
            for item in super().items():
                self.yielded_records += 1
                yield item

    config = make_hierarchical_config()
    local = LocalScheduler(
        config, config.hierarchical_topology.engine(0)
    )
    planner = AdmissionPlanner(
        AdmissionPlannerConfig.from_config(config)
    )
    shadow = planner.shadow_from_snapshot(
        local.load_snapshot(wave_id=1, quantum_id=0)
    )
    assert shadow is not None
    seed_commands = tuple(
        AddCommand(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
        for request_id in range(100, 104)
    )
    seed_reservations = tuple(
        planner.plan(shadow, command) for command in seed_commands
    )
    assert all(
        reservation is not None for reservation in seed_reservations
    )
    seed_results = local.commit_planned_batch(
        seed_commands,
        tuple(
            reservation
            for reservation in seed_reservations
            if reservation is not None
        ),
    )
    assert all(result.accepted for result in seed_results)

    commands = tuple(
        AddCommand(
            request_id=request_id,
            prompt_token_ids=(1, 2),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
        for request_id in range(104, 112)
    )
    reservations = tuple(
        planner.plan(shadow, command) for command in commands
    )
    assert all(reservation is not None for reservation in reservations)
    records = CountingRecords(local._records)
    local._records = records

    results = local.commit_planned_batch(
        commands,
        tuple(
            reservation
            for reservation in reservations
            if reservation is not None
        ),
    )

    assert all(result.accepted for result in results)
    assert records.values_calls == 1
    assert records.items_calls == 0
    assert records.yielded_records == len(seed_commands)

    records.values_calls = 0
    records.yielded_records = 0
    load = local.load_snapshot(wave_id=1, quantum_id=0)
    assert load.running == len(seed_commands) + len(commands)
    assert records.values_calls == 1
    assert records.yielded_records == load.running
    assert not local.is_finished()
    assert records.values_calls == 1
    assert records.yielded_records == load.running


def test_local_decode_batch_validates_rank_order_and_forward_count():
    batch = LocalDecodeBatch(
        wave_id=3,
        quantum_id=5,
        engine_id=1,
        engine_has_real=True,
        per_rank_sequences={4: [], 5: []},
        request_master_global_rank={10: 4},
        frozen_request_order={4: (10,), 5: ()},
        control_dummy_ids=frozenset({99}),
    )
    valid_results = [
        WorkerDecodeResult(
            wave_id=3,
            quantum_id=5,
            global_rank=4,
            forward_count=HIERARCHICAL_LOOP_COUNT,
            mastered_request_ids=(10,),
            sampled_token_ids=(tuple(range(HIERARCHICAL_LOOP_COUNT)),),
        ),
        WorkerDecodeResult(
            wave_id=3,
            quantum_id=5,
            global_rank=5,
            forward_count=HIERARCHICAL_LOOP_COUNT,
            mastered_request_ids=(),
            sampled_token_ids=(),
        ),
    ]

    assert set(batch.validate_worker_results(valid_results)) == {4, 5}

    invalid = valid_results[0]
    invalid = WorkerDecodeResult(
        wave_id=invalid.wave_id,
        quantum_id=invalid.quantum_id,
        global_rank=invalid.global_rank,
        forward_count=15,
        mastered_request_ids=invalid.mastered_request_ids,
        sampled_token_ids=invalid.sampled_token_ids,
    )
    with pytest.raises(ValueError, match="exactly 16"):
        batch.validate_worker_results([invalid, valid_results[1]])


def make_worker_results(
    batch: LocalDecodeBatch, *, token_base: int = 100
) -> list[WorkerDecodeResult]:
    return [
        WorkerDecodeResult(
            wave_id=batch.wave_id,
            quantum_id=batch.quantum_id,
            global_rank=global_rank,
            forward_count=HIERARCHICAL_LOOP_COUNT,
            mastered_request_ids=batch.expected_request_ids(global_rank),
            sampled_token_ids=tuple(
                tuple(
                    token_base + offset
                    for offset in range(HIERARCHICAL_LOOP_COUNT)
                )
                for _ in batch.expected_request_ids(global_rank)
            ),
        )
        for global_rank in batch.per_rank_sequences
    ]


def test_local_scheduler_bootstrap_and_final_overrun_accounting():
    config = make_hierarchical_config()
    local = LocalScheduler(config, config.hierarchical_topology.engine(1))
    result = local.add(
        AddCommand(
            request_id=42,
            prompt_token_ids=(10, 11, 12),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    )
    assert result.accepted
    assert local.admit() == (42,)

    sequence = local.cpp_scheduler.running(0)[0]
    assert sequence.num_prompt_tokens == 3
    assert sequence.num_bootstrap_tokens == 1
    assert sequence.num_completed_tokens == 0
    assert sequence.completion_token_ids == []
    assert sequence.block_ctx().dp_idx == 1
    assert local.state_manager.num_running_seqs == 1
    assert local.state_manager.num_running_tokens == 4

    first = local.plan_decode(wave_id=1, quantum_id=0)
    assert first.engine_has_real
    assert first.request_master_global_rank[42] in {4, 5, 6, 7}
    first_schedule_events = local.mark_first_forward_started(first)
    assert len(first_schedule_events) == 1
    assert first_schedule_events[0].request_id == 42
    assert first_schedule_events[0].engine_id == 1
    assert first_schedule_events[0].local_scheduler_queue_ms >= 0
    assert first_schedule_events[0].global_capacity_queue_ms == 0
    assert (
        first_schedule_events[0].first_schedule_latency_ms
        == first_schedule_events[0].local_scheduler_queue_ms
    )
    assert local.mark_first_forward_started(first) == ()
    first_load = local.load_snapshot(wave_id=1, quantum_id=0)
    assert len(first_load.rank_loads) == config.attention_sp
    assert tuple(load.global_rank for load in first_load.rank_loads) == (
        4,
        5,
        6,
        7,
    )
    assert sum(load.master_batch_size for load in first_load.rank_loads) == 1
    assert sum(
        load.active_master_requests for load in first_load.rank_loads
    ) == 1
    assert sum(
        load.active_dispatched_tokens for load in first_load.rank_loads
    ) == sequence.num_tokens
    assert all(
        load.active_receiver_requests >= 0
        and load.control_dummy_blocks > 0
        for load in first_load.rank_loads
    )
    assert sum(load.master_assignments for load in first_load.rank_loads) == 1
    assert all(load.total_blocks == 32 for load in first_load.rank_loads)
    assert local.postprocess(first, make_worker_results(first)) == ()
    assert sequence.num_completed_tokens == 16
    assert local.last_itl_token_slots == 15
    first_load = local.load_snapshot(wave_id=1, quantum_id=1)
    assert sum(
        load.mastered_decode_tokens for load in first_load.rank_loads
    ) == 16
    first_token_events = local.drain_first_token_events()
    assert len(first_token_events) == 1
    assert first_token_events[0].request_id == 42
    assert first_token_events[0].generated_count == 16

    assert local.admit() == ()
    final = local.plan_decode(wave_id=1, quantum_id=1)
    events = local.postprocess(
        final,
        make_worker_results(final, token_base=200),
        execute_latency_ms=0.0,
    )
    assert len(events) == 1
    assert events[0].request_id == 42
    assert events[0].generated_count == 17
    assert events[0].status == "FINISHED"
    assert events[0].first_forward_to_terminal_ms is not None
    assert events[0].first_forward_to_terminal_ms >= 0
    assert events[0].final_quantum_execute_ms == 0.0
    assert events[0].final_quantum_real_tokens == 1
    assert events[0].final_quantum_unused_decode_ms == 0.0
    assert events[0].global_capacity_queue_ms == 0
    assert local.drain_first_token_events() == ()
    assert sequence.num_completed_tokens == 17
    assert local.last_itl_token_slots == 1
    assert len(sequence.completion_token_ids) == 17
    assert local.state_manager.num_running_seqs == 0
    assert local.state_manager.num_running_tokens == 0
    assert local.is_finished()
    assert 42 not in local._records
    assert local._terminal_states[42] is RequestState.FINISHED
    duplicate = local.add(
        AddCommand(
            request_id=42,
            prompt_token_ids=(10, 11, 12),
            max_tokens=17,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    )
    assert not duplicate.accepted
    assert duplicate.reason == "duplicate request in state FINISHED"
    assert local.abort(42).status == "already_terminal"
    load = local.load_snapshot(wave_id=1, quantum_id=2)
    assert load.useful_decode_tokens == 17
    assert load.raw_token_slots == 128
    assert load.control_dummy_slots == 96
    assert load.total_rank_forwards == 128
    assert load.all_dummy_rank_forwards == 0
    assert len(load.rank_loads) == config.attention_sp
    assert sum(load.active_master_requests for load in load.rank_loads) == 0
    assert sum(load.active_receiver_requests for load in load.rank_loads) == 0
    assert sum(load.active_dispatched_tokens for load in load.rank_loads) == 0
    assert sum(load.master_assignments for load in load.rank_loads) == 1
    assert sum(load.mastered_decode_tokens for load in load.rank_loads) == 17


def test_local_scheduler_rejects_illegal_exclusive_sp_lifetime_on_add(
    monkeypatch,
):
    config = make_hierarchical_config(
        num_kvcache_blocks=3,
        reserved_blocks_per_req=0,
    )
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    created_sequences = []
    sequence_type = local_scheduler_module.Sequence

    def create_sequence(*args, **kwargs):
        created_sequences.append(args[0])
        return sequence_type(*args, **kwargs)

    monkeypatch.setattr(local_scheduler_module, "Sequence", create_sequence)

    rejected = local.add(
        AddCommand(
            request_id=43,
            prompt_token_ids=tuple(range(120)),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    )

    assert not rejected.accepted
    assert "exclusive LocalEngine SP placement" in rejected.reason
    assert local.cpp_scheduler.get_total_waiting_migration_size() == 0
    assert local.state_manager.num_running_seqs == 0
    assert local.admit() == ()
    assert created_sequences == []


def test_local_scheduler_accepts_distributed_exclusive_sp_lifetime(monkeypatch):
    config = make_hierarchical_config(
        num_kvcache_blocks=3,
        reserved_blocks_per_req=0,
        segment_size=64,
    )
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    created_sequences = []
    sequence_type = local_scheduler_module.Sequence

    def create_sequence(*args, **kwargs):
        created_sequences.append(args[0])
        return sequence_type(*args, **kwargs)

    monkeypatch.setattr(local_scheduler_module, "Sequence", create_sequence)

    accepted = local.add(
        AddCommand(
            request_id=44,
            prompt_token_ids=tuple(range(120)),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    )

    assert accepted.accepted
    assert local.admit() == (44,)
    sequence = local.cpp_scheduler.running(0)[0]
    assert local.state_manager.can_fit_lifetime(
        sequence, 1 + round_up(sequence.max_tokens)
    )
    assert created_sequences == [list(range(120))]


def test_local_scheduler_defers_admission_without_bootstrap_capacity():
    config = make_hierarchical_config(
        attention_dp=8,
        attention_sp=1,
        ffn_ep=8,
        num_kvcache_blocks=4,
        reserved_blocks_per_req=0,
    )
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    for request_id, prompt_tokens in (
        (45, tuple(range(120))),
        (46, tuple(range(1000, 1064))),
    ):
        assert local.add(
            AddCommand(
                request_id=request_id,
                prompt_token_ids=prompt_tokens,
                max_tokens=16,
                temperature=0.1,
                ignore_eos=True,
                wave_id=1,
            )
        ).accepted

    assert local.admit() == (45,)
    assert [
        sequence.seq_id for sequence in local.cpp_scheduler.waiting_migration
    ] == [46]
    assert local.state_manager.num_running_seqs == 1
    assert local.state_manager.num_running_tokens == 121


def test_local_scheduler_try_admit_rolls_back_transient_infeasibility():
    config = make_hierarchical_config(
        attention_dp=8,
        attention_sp=1,
        ffn_ep=8,
        num_kvcache_blocks=4,
        reserved_blocks_per_req=0,
    )
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    first = AddCommand(
        request_id=47,
        prompt_token_ids=tuple(range(120)),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
        wave_id=1,
    )
    second = AddCommand(
        request_id=48,
        prompt_token_ids=tuple(range(1000, 1064)),
        max_tokens=16,
        temperature=0.1,
        ignore_eos=True,
        wave_id=1,
    )

    assert local.try_admit(first).accepted
    deferred = local.try_admit(second)

    assert not deferred.accepted
    assert deferred.reason == "admission_deferred"
    assert [
        sequence.seq_id for sequence in local.cpp_scheduler.running(0)
    ] == [47]
    assert list(local.cpp_scheduler.waiting_migration) == []
    assert local.abort(48).status == "not_found"


def test_local_scheduler_inflight_abort_wins_before_commit():
    config = make_hierarchical_config()
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    assert local.add(
        AddCommand(
            request_id=77,
            prompt_token_ids=(10, 11),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    ).accepted
    local.admit()
    batch = local.plan_decode(wave_id=1, quantum_id=0)

    assert local.abort(77).status == "abort_pending"
    events = local.postprocess(batch, make_worker_results(batch))
    assert len(events) == 1
    assert events[0].status == "ABORTED"
    assert events[0].generated_count == 0
    assert local.state_manager.num_running_seqs == 0
    assert local.state_manager.num_running_tokens == 0
    assert local.abort(77).status == "already_terminal"
    assert local.is_finished()


def test_local_scheduler_waiting_abort_emits_terminal_immediately():
    config = make_hierarchical_config()
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    assert local.add(
        AddCommand(
            request_id=78,
            prompt_token_ids=(10, 11),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    ).accepted

    assert local.abort(78).status == "aborted"
    events = local.drain_terminal_events()
    assert [(event.request_id, event.status) for event in events] == [
        (78, "ABORTED")
    ]
    assert local.drain_terminal_events() == ()
    assert local.state_manager.num_running_seqs == 0
    assert local.state_manager.num_running_tokens == 0
    assert local.is_finished()


def test_local_scheduler_preempts_running_tail_and_readmits_cleanly():
    config = make_hierarchical_config(
        attention_dp=8,
        attention_sp=1,
        ffn_ep=8,
        num_kvcache_blocks=3,
        reserved_blocks_per_req=0,
    )
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    for request_id in (101, 102):
        assert local.add(
            AddCommand(
                request_id=request_id,
                prompt_token_ids=tuple(range(60)),
                max_tokens=16,
                temperature=0.1,
                ignore_eos=True,
                wave_id=1,
            )
        ).accepted

    assert local.admit() == (101, 102)
    assert local.state_manager.num_running_tokens == 122

    batch = local.plan_decode(wave_id=1, quantum_id=0)
    assert batch.expected_request_ids(0) == (101,)
    waiting = list(local.cpp_scheduler.waiting_migration)
    assert [sequence.seq_id for sequence in waiting] == [102]
    assert waiting[0].num_tokens == 60
    assert waiting[0].num_bootstrap_tokens == 0
    assert local.state_manager.num_running_tokens == 61
    assert local.load_snapshot(wave_id=1, quantum_id=0).preemption_count == 1

    local.mark_first_forward_started(batch)
    events = local.postprocess(batch, make_worker_results(batch))
    assert [(event.request_id, event.status) for event in events] == [
        (101, "FINISHED")
    ]
    assert local.state_manager.num_running_tokens == 0

    assert local.admit() == (102,)
    readmitted = local.cpp_scheduler.running(0)[0]
    assert readmitted.num_tokens == 61
    assert readmitted.num_bootstrap_tokens == 1
    assert local.state_manager.num_running_tokens == 61


def test_real_request_id_can_match_control_dummy_internal_id():
    config = make_hierarchical_config()
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    colliding_id = local.state_manager.dummy_seqs[0].seq_id
    assert local.add(
        AddCommand(
            request_id=colliding_id,
            prompt_token_ids=(10, 11),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    ).accepted
    assert local.admit() == (colliding_id,)

    batch = local.plan_decode(wave_id=1, quantum_id=0)
    assert batch.engine_has_real
    assert colliding_id in batch.request_master_global_rank
    local.mark_first_forward_started(batch)
    events = local.postprocess(batch, make_worker_results(batch))
    assert [(event.request_id, event.status) for event in events] == [
        (colliding_id, "FINISHED")
    ]


def test_control_dummy_id_collision_does_not_remove_dummy_on_abort():
    config = make_hierarchical_config()
    local = LocalScheduler(config, config.hierarchical_topology.engine(0))
    collision_dummy = local.state_manager.dummy_seqs[1]
    colliding_id = collision_dummy.seq_id
    assert local.add(
        AddCommand(
            request_id=colliding_id,
            prompt_token_ids=(10, 11),
            max_tokens=16,
            temperature=0.1,
            ignore_eos=True,
            wave_id=1,
        )
    ).accepted
    local.admit()
    batch = local.plan_decode(wave_id=1, quantum_id=0)
    assert any(
        id(sequence) == id(collision_dummy)
        for sequence in batch._all_sequences
    )

    assert local.abort(colliding_id).status == "abort_pending"
    events = local.postprocess(batch, make_worker_results(batch))

    assert [(event.request_id, event.status) for event in events] == [
        (colliding_id, "ABORTED")
    ]
    assert any(
        id(sequence) == id(collision_dummy)
        for sequence in batch._all_sequences
    )
