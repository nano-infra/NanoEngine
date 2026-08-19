import torch
import torch.nn.functional as F
from torch import nn

from nanodeploy.layers.embed_head import ParallelLMHead
from nanodeploy.worker.context import reset_context, set_context
from nanodeploy.worker.prefill_logits import compute_prefill_logits


class _RecordingCausalLM(nn.Module):
    def __init__(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
    ) -> None:
        super().__init__()
        self.hidden_states = hidden_states
        self.lm_head = lm_head
        self.logits_input: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids, positions
        return self.hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.logits_input = hidden_states
        return self.lm_head(hidden_states)


def _make_tp1_lm_head(weight: torch.Tensor) -> ParallelLMHead:
    lm_head = ParallelLMHead.__new__(ParallelLMHead)
    nn.Module.__init__(lm_head)
    lm_head.tp_rank = 0
    lm_head.tp_size = 1
    lm_head.weight = nn.Parameter(weight)
    return lm_head


def test_prefill_selects_last_hidden_states_only_in_lm_head() -> None:
    hidden_states = torch.arange(7 * 3, dtype=torch.float32).reshape(7, 3)
    weight = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)
    cu_seqlens_q = torch.tensor([0, 3, 4, 7], dtype=torch.int32)
    model = _RecordingCausalLM(
        hidden_states,
        _make_tp1_lm_head(weight),
    )

    set_context(True, cu_seqlens_q=cu_seqlens_q)
    try:
        logits = compute_prefill_logits(
            model,
            torch.zeros(7, dtype=torch.int64),
            torch.arange(7, dtype=torch.int64),
        )
    finally:
        reset_context()

    assert model.logits_input is hidden_states
    expected_hidden_states = hidden_states[torch.tensor([2, 3, 6])]
    torch.testing.assert_close(logits, F.linear(expected_hidden_states, weight))
