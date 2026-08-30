from dlengine.server.engine_server import _event_token_ids


def test_event_token_ids_preserves_multi_token_step():
    event = {"token_ids": [11, 12, 13], "last_token": 13, "num_tokens": 9}

    assert _event_token_ids(event) == [11, 12, 13]


def test_event_token_ids_falls_back_for_legacy_scheduler():
    event = {"last_token": 17, "num_tokens": 9}

    assert _event_token_ids(event) == [17]


def test_event_token_ids_falls_back_for_empty_compatibility_field():
    event = {"token_ids": [], "last_token": 0, "num_tokens": 9}

    assert _event_token_ids(event) == [0]


def test_event_token_ids_does_not_invent_token_for_empty_legacy_event():
    event = {"last_token": 0, "num_tokens": 0}

    assert _event_token_ids(event) == []
