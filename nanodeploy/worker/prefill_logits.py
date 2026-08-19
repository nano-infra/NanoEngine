import torch


def compute_prefill_logits(
    model,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
):
    """Run prefill while leaving last-token selection to the model's LM head."""
    hidden_states = model(input_ids, positions)
    return model.compute_logits(hidden_states)
