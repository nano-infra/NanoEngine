import logging

from nanodeploy.logging import _configured_log_level


def test_configured_log_level_reads_environment(monkeypatch):
    monkeypatch.setenv("NANODEPLOY_LOG_LEVEL", "warning")

    assert _configured_log_level() == logging.WARNING


def test_configured_log_level_falls_back_for_invalid_value(monkeypatch):
    monkeypatch.setenv("NANODEPLOY_LOG_LEVEL", "not-a-level")

    assert _configured_log_level() == logging.DEBUG
