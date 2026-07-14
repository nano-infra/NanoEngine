from types import SimpleNamespace

import dlengine._rust.proto as proto
from dlengine.engine.llm_engine import LLMEngine


class _FakeRunnerOut:
    def __init__(self, token_ids, logprobs=None, server_handler_ns=0):
        self.token_ids = token_ids
        self.logprobs = logprobs
        self.server_handler_ns = server_handler_ns


class _FakeRunnerIn:
    fragments = {}

    def __init__(self, payload):
        self.payload = payload

    @classmethod
    def from_bytes(cls, payload):
        return cls(payload)

    @classmethod
    def dummy(cls, *_args):
        return cls(b"dummy")

    def to_bytes(self):
        return self.payload

    def prefill_microbatches(self, _max_tokens):
        return self.fragments[self.payload]


class _FakeExecutor:
    def __init__(self, token_by_payload):
        self.token_by_payload = token_by_payload
        self.events = []
        self.submitted_payloads = []

    def run_batch_bytes_async(self, payloads, is_prefill):
        assert is_prefill is True
        self.events.append(("submit", payloads[0]))
        self.submitted_payloads.append(payloads)
        return payloads

    def max_inflight_requests(self):
        return 16

    def run_wait_runner_outs(self, payloads):
        self.events.append(("wait", payloads[0]))
        return [
            _FakeRunnerOut([[self.token_by_payload.get(payload, 0)]], None, 1)
            for payload in payloads
        ]


def test_static_pp_prefill_pipelines_long_and_independent_requests(monkeypatch):
    monkeypatch.setattr(proto, "RunnerIn", _FakeRunnerIn)
    monkeypatch.setattr(proto, "RunnerOut", _FakeRunnerOut)
    _FakeRunnerIn.fragments = {
        b"batch": [
            (b"request0_chunk0", 0, False),
            (b"request0_chunk1", 0, True),
            (b"request1_chunk0", 1, True),
        ]
    }

    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_id = "engine"
    engine.config = SimpleNamespace(
        pp_prefill_microbatch_tokens=4,
        pp_prefill_pipeline_depth=3,
        num_kvcache_blocks=16,
        executor_backend="ray",
    )
    engine.executor = _FakeExecutor(
        {
            b"request0_chunk0": 1,
            b"request0_chunk1": 2,
            b"request1_chunk0": 3,
        }
    )
    schedule_result = SimpleNamespace(dp_group_seq_ids=[[100, 200]])

    outputs = engine._run_static_pp_prefill_pipeline(
        [b"batch"], schedule_result, inner=1, tp_size=1, pp_size=2
    )

    assert outputs[0].token_ids == [[2], [3]]
    assert engine.executor.events == [
        ("submit", b"request0_chunk0"),
        ("submit", b"request0_chunk1"),
        ("submit", b"request1_chunk0"),
        ("wait", b"request0_chunk0"),
        ("wait", b"request0_chunk1"),
        ("wait", b"request1_chunk0"),
    ]


def test_static_pp_prefill_pads_shorter_dp_cells_with_dummy(monkeypatch):
    monkeypatch.setattr(proto, "RunnerIn", _FakeRunnerIn)
    monkeypatch.setattr(proto, "RunnerOut", _FakeRunnerOut)
    _FakeRunnerIn.fragments = {
        b"cell0": [(b"cell0_chunk0", 0, False), (b"cell0_chunk1", 0, True)],
        b"cell1": [(b"cell1_chunk0", 0, True)],
    }

    engine = LLMEngine.__new__(LLMEngine)
    engine.engine_id = "engine"
    engine.config = SimpleNamespace(
        pp_prefill_microbatch_tokens=4,
        pp_prefill_pipeline_depth=2,
        num_kvcache_blocks=16,
    )
    engine.executor = _FakeExecutor(
        {b"cell0_chunk0": 1, b"cell0_chunk1": 2, b"cell1_chunk0": 3}
    )

    outputs = engine._run_static_pp_prefill_pipeline(
        [b"cell0", b"cell1"],
        SimpleNamespace(dp_group_seq_ids=[[100], [200]]),
        inner=2,
        tp_size=1,
        pp_size=2,
    )

    assert [out.token_ids for out in outputs] == [[[2]], [[3]]]
    assert engine.executor.submitted_payloads == [
        [b"cell0_chunk0", b"cell1_chunk0"] * 2,
        [b"cell0_chunk1", b"dummy"] * 2,
    ]
