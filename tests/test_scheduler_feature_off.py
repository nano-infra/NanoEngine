import pytest

from nanodeploy._cpp import (
    LSFatalCode,
    LSSchedulerFatalError,
    ScheduleAction,
    Scheduler,
    Sequence,
)


def _scheduler(*, enable_ls: bool) -> Scheduler:
    return Scheduler(
        "",
        1,
        16,
        4096,
        16,
        -1,
        1,
        1,
        64,
        4,
        "decode",
        0.0,
        4,
        False,
        False,
        "legacy",
        100_000,
        0,
        False,
        "",
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1,
        1,
        1,
        False,
        "RoundRobin",
        False,
        0,
        enable_ls,
        0,
        64,
        True,
        "centralized",
        "off",
        0.50,
        0.80,
        32,
        64,
        8,
        0,
        True,
        10,
        1000,
        4096,
        128,
    )


def test_feature_off_scheduler_add_preserves_python_none_return():
    scheduler = _scheduler(enable_ls=False)
    sequence = Sequence([1, 2, 3], 1.0, 8, True)
    original_id = sequence.seq_id

    assert scheduler.add(sequence) is None
    assert list(scheduler.waiting_migration) == [sequence]
    assert sequence.assigned_dp == -1

    sequence.seq_id = original_id + 1000
    assert sequence.seq_id == original_id + 1000


def test_ls_fatal_exception_exposes_typed_code():
    scheduler = _scheduler(enable_ls=True)
    scheduler.latch_ls_fatal(LSFatalCode.NO_PROGRESS_INVARIANT)

    with pytest.raises(LSSchedulerFatalError) as raised:
        scheduler.schedule()

    assert raised.value.fatal_code == LSFatalCode.NO_PROGRESS_INVARIANT


def test_ls_pool_epoch_publishes_once_per_step():
    scheduler = _scheduler(enable_ls=True)
    sequence = Sequence([1, 2, 3], 1.0, 8, True)
    assert scheduler.add(sequence).accepted is True

    admission = scheduler.schedule()
    assert admission.action == ScheduleAction.ADMISSION
    assert admission.ls_pool_resource_epoch_before == [0]
    assert admission.ls_pool_resource_epoch_after == [1]

    decode = scheduler.schedule()
    assert decode.action == ScheduleAction.DECODE
    assert decode.ls_pool_resource_epoch_before == [1]
    assert decode.ls_pool_resource_epoch_after == [2]
