"""FlashInfer GDN recurrence strategy (SM90 native).

Uses ``chunk_gated_delta_rule`` for prefill and the pretranspose / nontranspose
fused decode kernels. Composed by ``FlashInferGatedDeltaNet``; requires the
FlashInfer GDN kernels at construction time.
"""

import torch

from . import kernels
from .recurrence import RecurrenceStateMixin


class FlashInferRecurrenceMixin(RecurrenceStateMixin):
    """FlashInfer chunk-prefill + fused-decode recurrence."""

    def _gdn_prefill(self, q, k, v, g, alpha, beta, scale, context) -> torch.Tensor:
        cu_seqlens = context.cu_seqlens_q.long()
        num_seqs = cu_seqlens.shape[0] - 1
        initial_state, states, slots = self._load_prefill_initial_state(
            context, num_seqs
        )

        q_normed = self._l2norm(q.float(), dim=-1).to(q.dtype)
        k_normed = self._l2norm(k.float(), dim=-1).to(k.dtype)
        if initial_state is not None:
            initial_state = initial_state.float()
        o, final_state = kernels.chunk_gated_delta_rule(
            q_normed,
            k_normed,
            v,
            g=alpha,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        self._store_prefill_final_state(final_state, states, slots, num_seqs)
        return o

    def _gdn_decode(self, q, k, v, a, b, scale, context) -> torch.Tensor:
        bs = q.shape[0]
        states = getattr(context, "gdn_recurrent_states", None)
        slots = getattr(context, "gdn_state_slots", None)

        if states is None:
            raise RuntimeError("FlashInfer GDN decode requires an allocated state pool.")

        if self._has_flashinfer_pretranspose:
            state_pool = states[self.layer_idx]
            if slots is not None:
                indices = slots[:bs].to(torch.int32)
            else:
                indices = torch.arange(bs, device=q.device, dtype=torch.int32)
            o, _ = kernels.gated_delta_rule_decode_pretranspose(
                q=q.unsqueeze(1),
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                state=None,
                A_log=self.A_log.detach().float(),
                a=a.unsqueeze(1),
                dt_bias=self.dt_bias.detach(),
                b=b.unsqueeze(1),
                scale=scale,
                use_qk_l2norm=True,
                initial_state=state_pool,
                initial_state_indices=indices,
            )
            return o.squeeze(1)

        # nontranspose decode: gather a contiguous batch view, transpose it for
        # the K-major/V-last kernel, then scatter it back to the V-major pool.
        if slots is not None:
            slots_b = slots[:bs]
            initial_state = states[self.layer_idx, slots_b]
        else:
            slots_b = None
            initial_state = states[self.layer_idx, :bs]
        nontranspose_state = initial_state.transpose(-1, -2).contiguous()
        o, updated_state = kernels.gated_delta_rule_decode(
            q=q.unsqueeze(1),
            k=k.unsqueeze(1),
            v=v.unsqueeze(1),
            state=nontranspose_state,
            A_log=self.A_log.detach().float(),
            a=a.unsqueeze(1),
            dt_bias=self.dt_bias.detach(),
            b=b.unsqueeze(1),
            scale=scale,
            use_qk_l2norm=True,
        )
        o = o.squeeze(1)
        updated_state = updated_state.transpose(-1, -2).to(states.dtype)
        if slots_b is not None:
            states[self.layer_idx, slots_b] = updated_state
        else:
            states[self.layer_idx, :bs] = updated_state
        return o


__all__ = ["FlashInferRecurrenceMixin"]
