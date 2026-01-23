import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Check for greedy search (temperature close to 0)
        # Assuming temperatures is [batch_size] or broadcastable

        # We need to handle the case where some are greedy and some are not,
        # or just all greedy if all temps are low.
        # For simplicity and performance in compile, let's try a condition.
        # Note: torch.compile might graph break on distinct paths if not carefully written.

        # Use a mask for greedy decoding
        greedy_mask = temperatures < 1e-5

        # Branchless implementation?
        # But we can't divide by zero/small temp.

        # Safe division temperature
        safe_temps = torch.where(
            greedy_mask, torch.ones_like(temperatures), temperatures
        )

        # Compute sampling
        scaled_logits = logits.float().div_(safe_temps.unsqueeze(dim=1))
        probs = torch.softmax(scaled_logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        # Compute greedy
        greedy_tokens = logits.argmax(dim=-1)

        # Select
        # temperatures shape is usually [batch_size]
        # output shape is [batch_size]
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)
