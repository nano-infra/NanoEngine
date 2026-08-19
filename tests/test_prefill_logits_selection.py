import pytest
import torch

from nanodeploy.worker.prefill_logits import select_prefill_last_hidden_states


def test_selects_last_query_row_for_each_prefill_sequence() -> None:
    hidden_states = torch.arange(7 * 3).reshape(7, 3)

    selected = select_prefill_last_hidden_states(
        hidden_states,
        (0, 3, 4, 7),
    )

    torch.testing.assert_close(
        selected,
        hidden_states[torch.tensor([2, 3, 6])],
    )


@pytest.mark.parametrize(
    ("offsets", "rows", "message"),
    [
        ((1, 2), 2, "start at zero"),
        ((0, 1), 2, "do not cover"),
        ((0, 1, 1), 1, "has no query tokens"),
    ],
)
def test_rejects_invalid_prefill_query_offsets(
    offsets: tuple[int, ...],
    rows: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        select_prefill_last_hidden_states(
            torch.empty(rows, 3),
            offsets,
        )
