"""Flash-Linear-Attention (FLA) GDN recurrence strategy.

Uses FLA's ``chunk_gated_delta_rule`` (prefill) and
``fused_recurrent_gated_delta_rule`` (decode). Composed by ``FlaGatedDeltaNet``;
requires flash-linear-attention at construction time.
"""

import torch
import torch.nn.functional as F

from . import kernels
from .recurrence import RecurrenceStateMixin


class FlaRecurrenceMixin(RecurrenceStateMixin):
    """FLA chunk-prefill + fused-recurrent-decode recurrence."""

    def _gdn_prefill(self, q, k, v, g, alpha, beta, scale, context) -> torch.Tensor:
        cu_seqlens = context.cu_seqlens_q.long()
        num_seqs = cu_seqlens.shape[0] - 1
        initial_state, states, slots = self._load_prefill_initial_state(
            context, num_seqs
        )
        o, final_state = kernels.fla_chunk_gated_delta_rule(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            scale=scale,
            initial_state=(
                initial_state.to(q.dtype) if initial_state is not None else None
            ),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        o = o.squeeze(0)
        self._store_prefill_final_state(final_state, states, slots, num_seqs)
        return o

    def _gdn_decode(self, q, k, v, a, b, scale, context) -> torch.Tensor:
        bs = q.shape[0]
        states = getattr(context, "gdn_recurrent_states", None)
        slots = getattr(context, "gdn_state_slots", None)
        if states is None:
            raise RuntimeError("FLA GDN decode requires an allocated state pool.")
        if slots is not None:
            initial_state = states[self.layer_idx, slots[:bs]]
        else:
            initial_state = states[self.layer_idx, :bs]
        beta = b.sigmoid()
        A_exp = -self.A_log.float().exp()
        g = A_exp * F.softplus(a.float() + self.dt_bias)
        o, updated_state = kernels.fla_fused_recurrent_gated_delta_rule(
            q.unsqueeze(1),
            k.unsqueeze(1),
            v.unsqueeze(1),
            g=g.unsqueeze(1),
            beta=beta.unsqueeze(1),
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
        )
        o = o.squeeze(1)
        self._store_decode_updated_state(updated_state, states, slots, bs)
        return o


__all__ = ["FlaRecurrenceMixin"]
