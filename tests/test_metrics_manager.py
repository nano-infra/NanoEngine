import pytest

from nanodeploy.metrics import MetricsManager


def test_metric_ticket_is_unpublished_and_unaccounted_until_commit():
    manager = MetricsManager()

    ticket = manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=11)

    assert ticket.metric.arrival_time is not None
    assert ticket.metric.decode_arrival_time is not None
    assert manager.get_sequence_metric(7) is None
    assert manager.server_metric.total_prompt_tokens == 0

    metric = manager.commit_sequence_metric(ticket)

    assert metric is ticket.metric
    assert manager.get_sequence_metric(7) is metric
    assert manager.server_metric.total_prompt_tokens == 11


def test_metric_ticket_abort_does_not_publish_or_account_prompt():
    manager = MetricsManager()
    ticket = manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=11)

    assert manager.abort_sequence_metric(ticket) is True
    assert manager.abort_sequence_metric(ticket) is False
    assert manager.get_sequence_metric(7) is None
    assert manager.server_metric.total_prompt_tokens == 0

    replacement = manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=13)
    manager.commit_sequence_metric(replacement)
    assert manager.get_sequence_metric(7) is replacement.metric
    assert manager.server_metric.total_prompt_tokens == 13


def test_metric_ticket_uses_insert_if_absent_and_cannot_commit_twice():
    manager = MetricsManager()
    ticket = manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=11)

    with pytest.raises(ValueError, match="already exists"):
        manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=99)

    manager.commit_sequence_metric(ticket)

    with pytest.raises(ValueError, match="not active"):
        manager.commit_sequence_metric(ticket)
    with pytest.raises(ValueError, match="already exists"):
        manager.prepare_sequence_metric(seq_id=7, num_prompt_tokens=99)


def test_metric_ticket_cannot_cross_managers():
    first = MetricsManager()
    second = MetricsManager()
    ticket = first.prepare_sequence_metric(seq_id=7, num_prompt_tokens=11)

    with pytest.raises(ValueError, match="another manager"):
        second.commit_sequence_metric(ticket)
    with pytest.raises(ValueError, match="another manager"):
        second.abort_sequence_metric(ticket)

    assert first.abort_sequence_metric(ticket) is True


def test_ls_rejection_telemetry_is_server_level_only():
    manager = MetricsManager()
    error = type("FakeError", (), {"name": "FUTURE_TOKEN_NO_FIT"})()

    manager.record_ls_rejected_request(error)
    manager.record_ls_rejected_request(error)

    summary = manager.get_server_summary()
    assert summary["ls_rejected_requests"] == 2
    assert summary["ls_rejected_requests_by_reason"] == {
        "FUTURE_TOKEN_NO_FIT": 2
    }
    assert manager.server_metric.total_prompt_tokens == 0
