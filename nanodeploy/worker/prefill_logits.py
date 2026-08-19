import torch


def select_prefill_last_hidden_states(
    hidden_states: torch.Tensor,
    cu_seqlens_q: tuple[int, ...],
) -> torch.Tensor:
    """Select the final query row for each packed prefill sequence."""
    if len(cu_seqlens_q) < 2 or cu_seqlens_q[0] != 0:
        raise ValueError(
            "Prefill query offsets must start at zero and contain a sequence"
        )
    if cu_seqlens_q[-1] != hidden_states.size(0):
        raise ValueError(
            "Prefill query offsets do not cover the packed hidden states: "
            f"offset_end={cu_seqlens_q[-1]}, rows={hidden_states.size(0)}"
        )

    last_token_indices: list[int] = []
    for seq_idx, (start, end) in enumerate(
        zip(cu_seqlens_q[:-1], cu_seqlens_q[1:], strict=True)
    ):
        if end <= start:
            raise ValueError(
                f"Prefill sequence {seq_idx} has no query tokens: "
                f"start={start}, end={end}"
            )
        last_token_indices.append(end - 1)

    indices = torch.tensor(
        last_token_indices,
        dtype=torch.int64,
        device=hidden_states.device,
    )
    return hidden_states.index_select(0, indices)
