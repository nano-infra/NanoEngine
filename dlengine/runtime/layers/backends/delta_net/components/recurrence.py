"""GDN delta-rule recurrence — shared state plumbing + naive reference.

The recurrence strategy is chosen by *which mixin a backend composes*, not by
runtime ``_has_*`` flags:

- ``NaiveRecurrenceMixin``      : pure PyTorch reference (this module).
- ``FlashInferRecurrenceMixin`` : ``recurrence_flashinfer.py``.
- ``FlaRecurrenceMixin``        : ``recurrence_fla.py``.

``RecurrenceStateMixin`` holds the shared recurrent-state-pool gather/scatter
(the subtle bit) so the three strategies do not duplicate it. State layout is
K-last: [N, H, V, K].
"""

import torch

from ._common import l2norm


class RecurrenceStateMixin:
    """Shared recurrent-state-pool load/store helpers.

    Requires the host to provide ``layer_idx``, ``num_v_heads``, ``head_v_dim``,
    ``head_k_dim`` and the ``StateMixin._continuation_keep_mask`` helper.
    """

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return l2norm(x, dim, eps)

    def _load_prefill_initial_state(self, context, num_seqs):
        """Gather (and fresh-slot-mask) the per-sequence prefill initial state."""
        states = getattr(context, "gdn_recurrent_states", None)
        slots = getattr(context, "gdn_state_slots", None)
        initial_state = None
        if states is not None:
            if context.block_tables is not None:
                if slots is not None:
                    initial_state = states[self.layer_idx, slots[:num_seqs]]
                else:
                    initial_state = states[self.layer_idx, :num_seqs]
                # Force fresh (first-chunk) sequences to start from zero state;
                # block_tables is batch-global so a fresh seq batched with a
                # continuation would otherwise inherit a stale reused slot.
                keep = self._continuation_keep_mask(
                    context, num_seqs, initial_state.dtype
                )
                if keep is not None:
                    initial_state = initial_state * keep.view(-1, 1, 1, 1)
            else:
                initial_state = states.new_zeros(
                    num_seqs, self.num_v_heads, self.head_v_dim, self.head_k_dim
                )
        return initial_state, states, slots

    def _store_prefill_final_state(self, final_state, states, slots, num_seqs):
        if states is not None and final_state is not None:
            final_state = final_state.to(states.dtype)
            if slots is not None:
                states[self.layer_idx, slots[:num_seqs]] = final_state
            else:
                states[self.layer_idx, :num_seqs] = final_state

    def _load_decode_initial_state(self, context, bs):
        states = getattr(context, "gdn_recurrent_states", None)
        slots = getattr(context, "gdn_state_slots", None)
        if states is None:
            return None, None, None
        if slots is not None:
            return states[self.layer_idx, slots[:bs]], states, slots
        return states[self.layer_idx, :bs], states, slots

    def _store_decode_updated_state(self, updated, states, slots, bs):
        updated = updated.to(states.dtype)
        if slots is not None:
            states[self.layer_idx, slots[:bs]] = updated
        else:
            states[self.layer_idx, :bs] = updated


class NaiveRecurrenceMixin(RecurrenceStateMixin):
    """Pure-PyTorch reference recurrence (no FlashInfer / FLA)."""

    def _gdn_prefill(self, q, k, v, g, alpha, beta, scale, context) -> torch.Tensor:
        cu_seqlens = context.cu_seqlens_q.long()
        num_seqs = cu_seqlens.shape[0] - 1
        initial_state, states, slots = self._load_prefill_initial_state(
            context, num_seqs
        )
        o, final_state = self._naive_gdn_prefill(
            q, k, v, g, beta, scale, cu_seqlens, initial_state
        )
        self._store_prefill_final_state(final_state, states, slots, num_seqs)
        return o

    def _gdn_decode(self, q, k, v, a, b, scale, context) -> torch.Tensor:
        import torch.nn.functional as F

        bs = q.shape[0]
        initial_state, states, slots = self._load_decode_initial_state(context, bs)
        beta = b.sigmoid()
        A_exp = -self.A_log.float().exp()
        g = A_exp * F.softplus(a.float() + self.dt_bias)
        if initial_state is None:
            initial_state = q.new_zeros(
                bs, self.num_v_heads, self.head_v_dim, self.head_k_dim
            )
        o, updated_state = self._naive_gdn_decode(q, k, v, g, beta, scale, initial_state)
        if states is not None:
            self._store_decode_updated_state(updated_state, states, slots, bs)
        return o

    def _naive_gdn_prefill(
        self, q, k, v, g, beta, scale, cu_seqlens, initial_state=None
    ):
        """Naive sequential scan for prefill. State is K-last [H, V, K]."""
        num_seqs = cu_seqlens.shape[0] - 1
        outputs = []
        final_states = []

        q = self._l2norm(q.float(), dim=-1)
        k = self._l2norm(k.float(), dim=-1)
        q = q * scale

        for i in range(num_seqs):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()

            if initial_state is not None:
                S = initial_state[i].float().clone()  # [H, V, K]
            else:
                S = torch.zeros(
                    self.num_v_heads,
                    self.head_v_dim,
                    self.head_k_dim,
                    dtype=torch.float32,
                    device=q.device,
                )

            if end <= start:
                final_states.append(S)
                continue
            qi, ki, vi = q[start:end], k[start:end], v[start:end]
            gi, bi = g[start:end], beta[start:end]

            out_seq = []
            for t in range(end - start):
                qt = qi[t]
                kt = ki[t]
                vt = vi[t].float()
                bt = bi[t].float()
                gt = gi[t].float()

                decay = gt.exp().unsqueeze(-1).unsqueeze(-1)
                S = decay * S  # [H, V, K]

                vk = torch.einsum("hv,hk->hvk", vt, kt)
                Sk = torch.einsum("hvk,hk->hv", S, kt)
                correction = torch.einsum("hv,hk->hvk", Sk, kt)
                S = S + bt.unsqueeze(-1).unsqueeze(-1) * (vk - correction)

                out_t = torch.einsum("hk,hvk->hv", qt, S)
                out_seq.append(out_t)

            out_seq = torch.stack(out_seq, dim=0).to(v.dtype)
            outputs.append(out_seq)
            final_states.append(S)

        if outputs:
            output = torch.cat(outputs, dim=0)
        else:
            output = v.new_zeros(0, self.num_v_heads, self.head_v_dim)

        final_state = torch.stack(final_states, dim=0)
        return output, final_state

    def _naive_gdn_decode(self, q, k, v, g, beta, scale, state):
        """Naive recurrent step for decode. State is K-last [B, H, V, K]."""
        q = self._l2norm(q.float(), dim=-1) * scale
        k = self._l2norm(k.float(), dim=-1)
        state_f32 = state.float()  # [B, H, V, K]

        decay = g.float().exp().unsqueeze(-1).unsqueeze(-1)
        state_f32.mul_(decay)

        vk = torch.einsum("bhv,bhk->bhvk", v.float(), k)
        Sk = torch.einsum("bhvk,bhk->bhv", state_f32, k)
        correction = torch.einsum("bhv,bhk->bhvk", Sk, k)
        state_f32.add_(beta.float().unsqueeze(-1).unsqueeze(-1) * (vk - correction))

        out = torch.einsum("bhk,bhvk->bhv", q, state_f32)

        return out.to(v.dtype), state_f32


# Backwards-compatible alias: some tests/imports reference ``RecurrenceMixin``.
RecurrenceMixin = NaiveRecurrenceMixin


__all__ = ["RecurrenceStateMixin", "NaiveRecurrenceMixin", "RecurrenceMixin"]
