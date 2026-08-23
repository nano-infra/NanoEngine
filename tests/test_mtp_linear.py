from types import SimpleNamespace

import torch
import torch.nn as nn
from dlengine.runtime.context.batch import reset_batch_context, set_batch_context
from dlengine.runtime.models.deepseek_v2.deepseek_v2_mtp import (
    DeepSeekMTP,
    DeepSeekMTPSharedHead,
)
from dlengine.runtime.runner.mtp_runner import (
    _active_ragged_last_rows,
    _localize_packed_topk,
    _nonempty_ragged_bounds,
    linear_greedy_verify,
    linear_rejection_sample,
    MTPRunner,
)


def _greedy_case_logits(accepted: int, num_drafts: int = 5):
    vocab_size = 16
    drafts = torch.arange(1, num_drafts + 1, dtype=torch.int64)[None, :]
    logits = torch.full((1, num_drafts + 1, vocab_size), -20.0)
    for step in range(num_drafts):
        target = int(drafts[0, step]) if step < accepted else 12
        logits[0, step, target] = 20.0
    logits[0, num_drafts, 13] = 20.0
    return logits, drafts


def test_linear_mtp_supports_every_acceptance_length_through_five():
    for expected_accepted in range(6):
        logits, drafts = _greedy_case_logits(expected_accepted)
        next_tokens, accepted, next_logprobs, draft_logprobs = linear_rejection_sample(
            logits, drafts, torch.tensor([0.0])
        )

        assert accepted.tolist() == [expected_accepted]
        assert next_tokens.tolist() == [13 if expected_accepted == 5 else 12]
        assert next_logprobs.shape == (1,)
        assert draft_logprobs.shape == (1, 5)


def test_linear_greedy_verify_supports_batched_acceptance_lengths_through_five():
    cases = [_greedy_case_logits(accepted) for accepted in range(6)]
    logits = torch.cat([case[0] for case in cases], dim=0)
    drafts = torch.cat([case[1] for case in cases], dim=0)

    next_tokens, accepted = linear_greedy_verify(logits, drafts)

    assert accepted.tolist() == list(range(6))
    assert next_tokens.tolist() == [12, 12, 12, 12, 12, 13]


def test_stochastic_one_hot_rejection_preserves_target_distribution():
    torch.manual_seed(7)
    batch_size = 50_000
    probabilities = torch.tensor([0.2, 0.3, 0.5])
    logits = probabilities.log().repeat(batch_size, 2, 1)
    drafts = torch.zeros((batch_size, 1), dtype=torch.int64)

    next_tokens, accepted, _, _ = linear_rejection_sample(
        logits, drafts, torch.ones(batch_size)
    )
    first_emitted = torch.where(accepted == 1, drafts[:, 0], next_tokens)
    observed = torch.bincount(first_emitted, minlength=3).float() / batch_size

    torch.testing.assert_close(observed, probabilities, atol=0.012, rtol=0)


def test_stochastic_recovery_logprob_uses_original_target_distribution():
    # Force rejection of draft token 0 by giving it zero target probability.
    logits = torch.tensor([[[-float("inf"), 0.0, 1.0], [0.0, 0.0, 0.0]]])
    drafts = torch.tensor([[0]])
    next_tokens, accepted, next_logprobs, _ = linear_rejection_sample(
        logits, drafts, torch.tensor([0.7]), generator=torch.Generator().manual_seed(3)
    )

    assert accepted.tolist() == [0]
    expected = torch.log_softmax(logits[0, 0] / 0.7, dim=-1)[next_tokens[0]]
    torch.testing.assert_close(next_logprobs[0], expected)


def test_mtp_shared_head_add_norm_is_the_recurrent_hidden():
    class AddNorm(nn.Module):
        def forward(self, hidden, residual=None):
            assert residual is not None
            combined = hidden + residual
            return combined, combined

    shared_head = object.__new__(DeepSeekMTPSharedHead)
    nn.Module.__init__(shared_head)
    shared_head.norm = AddNorm()
    shared_head.head = nn.Identity()

    hidden = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])

    assert torch.equal(shared_head(hidden, residual), hidden + residual)


def test_mtp_compute_logits_does_not_normalize_hidden_twice():
    class FailIfCalled(nn.Module):
        def forward(self, hidden, residual=None):
            raise AssertionError("shared-head norm was applied twice")

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_head = object.__new__(DeepSeekMTPSharedHead)
            nn.Module.__init__(self.shared_head)
            self.shared_head.norm = FailIfCalled()
            self.shared_head.head = nn.Identity()

    model = object.__new__(DeepSeekMTP)
    nn.Module.__init__(model)
    model.mtp_start_layer_idx = 78
    model.num_mtp_layers = 1
    model.layers = nn.ModuleDict({"78": Layer()})
    hidden = torch.tensor([[1.0, 2.0]])

    assert torch.equal(model.compute_logits(hidden), hidden)


def test_lazy_verify_state_is_reordered_by_sequence_id():
    runner = object.__new__(MTPRunner)
    runner.config = SimpleNamespace(num_speculative_tokens=2, max_model_len=128)
    runner._prev_seq_ids = (10, 20, 30)
    runner._prev_drafts = torch.tensor([[1, 2], [3, 4], [5, 6]])
    runner._selected_prev_drafts = None

    assert runner.can_lazy_verify([30, 10], torch.tensor([7, 8]), 2)
    assert runner._selected_prev_drafts.tolist() == [[5, 6], [1, 2]]
    assert not runner.can_lazy_verify([30, 99], torch.tensor([7, 8]), 2)


def test_lazy_verify_falls_back_before_crossing_model_limit():
    runner = object.__new__(MTPRunner)
    runner.config = SimpleNamespace(num_speculative_tokens=5, max_model_len=16)
    runner._prev_seq_ids = (10,)
    runner._prev_drafts = torch.tensor([[1, 2, 3, 4, 5]])
    runner._selected_prev_drafts = None

    set_batch_context(is_prefill=False, mtp_draft_safe=True)
    assert runner.can_lazy_verify([10], torch.tensor([10]), 1)
    set_batch_context(is_prefill=False, mtp_draft_safe=False)
    assert not runner.can_lazy_verify([10], torch.tensor([11]), 1)
    reset_batch_context()


def test_mtp_prefill_bounds_reject_zero_length_prefix_cache_segments():
    assert _nonempty_ragged_bounds(torch.tensor([0, 2, 5]), 2) == [0, 2, 5]
    assert _nonempty_ragged_bounds(torch.tensor([0, 0]), 1) is None
    assert _nonempty_ragged_bounds(torch.tensor([0, 2, 2]), 2) is None
    assert _nonempty_ragged_bounds(torch.tensor([0]), 1) is None


def test_mtp_prefill_last_rows_ignore_padded_cu_seqlens_suffix():
    padded_cu = torch.tensor([0, 53, 106, 0, 0], dtype=torch.int32)
    assert _active_ragged_last_rows(padded_cu, 2).tolist() == [52, 105]


def test_mtp_prefill_topk_is_localized_before_page_table_gather():
    packed = torch.tensor([[0, 52, -1], [53, 105, -1]], dtype=torch.int32)
    starts = torch.tensor([0, 53], dtype=torch.int32)

    assert _localize_packed_topk(packed, starts).tolist() == [
        [0, 52, -1],
        [0, 52, -1],
    ]


def test_empty_prefill_padding_still_runs_every_predictor_collective():
    class FakeMTP:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, positions, hidden, *, spec_step_idx):
            self.calls += 1
            assert input_ids.shape == (1,)
            assert positions.shape == (1,)
            assert hidden.shape == (1, 3)
            return hidden

        def compute_logits(self, hidden, *, spec_step_idx):
            return torch.zeros(hidden.size(0), 4)

    runner = object.__new__(MTPRunner)
    runner.config = SimpleNamespace(
        num_speculative_tokens=5,
        hf_config=SimpleNamespace(hidden_size=3),
    )
    runner.mtp_model = FakeMTP()
    context_sizes = []
    runner._set_mtp_context = context_sizes.append
    runner._greedy_draft = lambda logits, ids: ids.new_zeros(logits.size(0))

    runner._run_uncached_collective_padding(
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, 3),
        num_seqs=0,
    )

    assert runner.mtp_model.calls == 5
    assert context_sizes == [1, 1, 1, 1, 1]


def test_collective_padding_reuses_cached_recurrent_chain():
    class FakeMTP:
        def __init__(self):
            self.calls = 0

        def __call__(self, input_ids, positions, hidden, *, spec_step_idx):
            self.calls += 1
            assert spec_step_idx == 0
            return hidden + 1

        def compute_logits(self, hidden, *, spec_step_idx):
            return torch.zeros(hidden.size(0), 4)

    class FakeChainGraph:
        def __init__(self):
            self.calls = []

        def run_padding(self, input_ids, positions, hidden_states, bs):
            self.calls.append(
                (input_ids.clone(), positions.clone(), hidden_states.clone(), bs)
            )
            return True

    runner = object.__new__(MTPRunner)
    runner.config = SimpleNamespace(
        num_speculative_tokens=5,
        hf_config=SimpleNamespace(hidden_size=3),
    )
    runner.mtp_model = FakeMTP()
    runner.cached_mtp_graph_runner = FakeChainGraph()
    context_sizes = []
    runner._set_mtp_context = context_sizes.append
    runner._greedy_draft = lambda logits, ids: ids.new_zeros(logits.size(0))

    runner._run_uncached_collective_padding(
        torch.tensor([7]),
        torch.tensor([11]),
        torch.ones(1, 3),
        num_seqs=1,
    )

    assert runner.mtp_model.calls == 1
    assert context_sizes == [1]
    [(ids, positions, hidden, bs)] = runner.cached_mtp_graph_runner.calls
    assert ids.tolist() == [0]
    assert positions.tolist() == [13]
    assert hidden.tolist() == [[2.0, 2.0, 2.0]]
    assert bs == 1
