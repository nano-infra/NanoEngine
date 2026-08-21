from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from nanodeploy import SamplingParams
from nanodeploy._cpp import SequenceStatus
from nanodeploy.metrics import MetricsManager


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import bench_pd_serving as pd_serving  # noqa: E402


def test_pd_serving_cli_has_no_loop_count_override():
    parser = pd_serving.build_arg_parser()
    args = parser.parse_args([])

    assert not hasattr(args, "decode_loop_count")
    assert "loop-count" not in parser.format_help()
    assert pd_serving.DECODE_LOOP_COUNT == 1


def test_poisson_arrivals_are_seeded_and_bounded_by_duration():
    first = pd_serving.generate_poisson_arrival_times(
        10.0,
        5.0,
        np.random.default_rng(7),
    )
    second = pd_serving.generate_poisson_arrival_times(
        10.0,
        5.0,
        np.random.default_rng(7),
    )

    assert first == second
    assert first == sorted(first)
    assert len(first) > 1
    assert all(0 < arrival <= 10.0 for arrival in first)


class _FakePrefill:
    def __init__(self):
        self.metrics_manager = MetricsManager()
        self.waiting = []
        self.migrated = set()
        self.freed = set()

    def add_request(self, sequence):
        sequence.metric = self.metrics_manager.create_sequence_metric(
            sequence.seq_id,
            sequence.num_prompt_tokens,
        )
        sequence.metric.record_arrival()
        self.waiting.append(sequence)

    def is_finished(self):
        return not self.waiting

    def step(self):
        batch, self.waiting = self.waiting, []
        for sequence in batch:
            sequence.metric.record_first_scheduled()
            sequence.metric.record_first_token()
            sequence.metric.record_step_tokens(1, 1.0)
            sequence.status = SequenceStatus.TO_BE_MIGRATED
            self.migrated.add(sequence.seq_id)
        return [], 0, len(batch), 0.0, 0.0

    def free_to_be_migrated(self, sequences):
        for sequence in sequences:
            assert sequence.seq_id in self.migrated
            self.freed.add(sequence.seq_id)


class _FakeDecode:
    def __init__(self):
        self.metrics_manager = MetricsManager()
        self.waiting = []
        self.running = []

    def add_request(self, sequence):
        sequence.metric = self.metrics_manager.create_sequence_metric(
            sequence.seq_id,
            sequence.num_prompt_tokens,
        )
        sequence.metric.record_arrival()
        sequence.metric.record_decode_arrival()
        self.waiting.append(sequence)

    def is_finished(self):
        return not self.waiting and not self.running

    def step(self):
        if self.waiting:
            self.running.extend(self.waiting)
            self.waiting.clear()
            for sequence in self.running:
                sequence.status = SequenceStatus.RUNNING
                sequence.metric.record_decode_scheduled()
            return [], 0, len(self.running), 0.0, 0.0

        batch, self.running = self.running, []
        outputs = []
        for sequence in batch:
            sequence.metric.record_step_tokens(1, 1.0)
            sequence.status = SequenceStatus.FINISHED
            self.metrics_manager.complete_sequence(sequence.seq_id)
            outputs.append((sequence.seq_id, []))
        return outputs, -len(batch), len(batch), 0.0, 0.0


def test_pd_driver_preserves_metric_and_releases_prefill_kv():
    prefill = _FakePrefill()
    decode = _FakeDecode()
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=2,
        ignore_eos=True,
    )

    _, sequence_map, workload_by_sequence = pd_serving.run_pd_benchmark(
        prefill,
        decode,
        [pd_serving.WorkloadRequest([1] * 16, sampling_params)],
        [0.0],
        show_progress=False,
    )

    sequence = next(iter(sequence_map.values()))
    metric = sequence.metric
    assert metric is prefill.metrics_manager.sequence_metrics[sequence.seq_id]
    assert metric is decode.metrics_manager.sequence_metrics[sequence.seq_id]
    assert metric.arrival_time < metric.decode_arrival_time
    assert metric.first_token_time < metric.completion_time
    assert metric.num_generated_tokens == 2
    assert prefill.freed == {sequence.seq_id}
    assert workload_by_sequence[sequence.seq_id].prompt_token_ids == [1] * 16


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize and add_generation_prompt
        return [1, 2, 3] + list(range(len(messages[0]["content"].split())))

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(range(len(text.split())))


def test_sharegpt_uses_first_valid_human_to_gpt_pair_and_real_text():
    record = {
        "id": "conversation-1",
        "conversations": [
            {"from": "gpt", "value": "orphan answer"},
            {"from": "human", "value": "real user prompt"},
            {"from": "gpt", "value": "one two three four"},
        ],
    }

    pair = pd_serving._first_sharegpt_pair(record)
    requests = pd_serving.build_sharegpt_requests(
        [pair],
        _FakeTokenizer(),
        count=2,
        max_model_len=32,
        max_request_tokens=0,
        max_output_tokens=2,
        temperature=0.6,
    )

    assert pair == (
        "real user prompt",
        "one two three four",
        "conversation-1",
    )
    assert len(requests) == 2
    assert requests[0].prompt_text == "real user prompt"
    assert requests[0].reference_text == "one two three four"
    assert requests[0].sampling_params.max_tokens == 2
    assert not requests[0].sampling_params.ignore_eos


def test_pd_serving_rejects_nonpositive_duration():
    args = pd_serving.build_arg_parser().parse_args([])
    args.duration = 0

    with pytest.raises(ValueError, match="--duration must be positive"):
        pd_serving.validate_args(args)
