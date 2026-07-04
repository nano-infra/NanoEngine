import torch
from torch import nn

from dlengine.compile_utils import maybe_compile


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        # Compile lazily in __init__ rather than via @torch.compile class
        # decorators. The decorator form attaches torch._dynamo / _inductor
        # ConfigModuleInstance references to the class, which break
        # cloudpickle when this module is imported by a Ray actor.
        self.forward = maybe_compile(self.forward)
        self.forward_with_logprobs = maybe_compile(self.forward_with_logprobs)

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

        # Compute sampling in log-space. This is equivalent to the classic
        # probs / exponential_noise Gumbel-max trick, but avoids underflowing
        # very small probabilities to zero.
        scaled_logits = logits.float() / safe_temps.unsqueeze(dim=1)
        log_probs = torch.log_softmax(scaled_logits, dim=-1)
        gumbel = torch.empty_like(log_probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = (log_probs - gumbel.log()).argmax(dim=-1)

        # Compute greedy
        greedy_tokens = logits.argmax(dim=-1)

        # Select
        # temperatures shape is usually [batch_size]
        # output shape is [batch_size]
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

    def forward_with_logprobs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """Same sampling as ``forward`` but also returns log-prob of the
        chosen token under the temperature-scaled distribution.

        Returns ``(token_ids [B], logprobs [B])`` both shape-aligned with
        the input ``logits[B, V]``. The greedy branch logprob is computed
        from the *temperature-scaled* distribution as well so it reflects
        what the policy "thought" it was doing — clamping to the actual
        argmax token ensures numerical stability when one logit dominates.

        Sibling method (kept separate from ``forward``) so callers that
        don't need logprobs aren't impacted and torch.compile traces both
        paths independently.
        """
        greedy_mask = temperatures < 1e-5
        safe_temps = torch.where(
            greedy_mask, torch.ones_like(temperatures), temperatures
        )

        # Compute scaled distribution in log-space. log_softmax is
        # numerically stable; we use it for both stochastic sampling and
        # chosen-token logprobs.
        scaled_logits = logits.float() / safe_temps.unsqueeze(dim=1)
        log_probs = torch.log_softmax(scaled_logits, dim=-1)

        # Stochastic Gumbel-max sample (does not corrupt log_probs).
        gumbel = torch.empty_like(log_probs).exponential_(1).clamp_min_(1e-10)
        sample_tokens = (log_probs - gumbel.log()).argmax(dim=-1)

        greedy_tokens = logits.argmax(dim=-1)
        chosen = torch.where(greedy_mask, greedy_tokens, sample_tokens)

        # gather log_prob of the chosen token along the vocab dim.
        chosen_lp = log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        return chosen, chosen_lp
