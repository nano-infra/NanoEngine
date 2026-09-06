"""GDN delta-rule recurrence component (prefill chunk + decode step).

Selects between FlashInfer, FLA, and a naive PyTorch reference implementation
using the instance capability flags (``_has_flashinfer_prefill``,
``_has_flashinfer_pretranspose``, ``_has_flashinfer_nontranspose``,
``_has_fla``). State layout is K-last: [N, H, V, K].
"""

import torch
import torch.nn.functional as F

from . import kernels
from ._common import l2norm


class RecurrenceMixin:
    """Prefill and decode delta-rule recurrence.

    Requires the host class to provide: ``layer_idx``, ``num_v_heads``,
    ``head_v_dim``, ``head_k_dim``, ``A_log``, ``dt_bias``, and the capability
    flags listed in the module docstring.
    """

    @staticmethod
    def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
        return l2norm(x, dim, eps)

    def _gdn_prefill(self, q, k, v, g, alpha, beta, scale, context) -> torch.Tensor:
        """Prefill: chunk mode. State layout is K-last [N, H, V, K].

        Args:
            g: log-space decay, used by naive fallback (negative values)
            alpha: linear decay factor = exp(g), used by flashinfer kernel (in [0, 1])
        """
        cu_seqlens = context.cu_seqlens_q.long()
        num_seqs = cu_seqlens.shape[0] - 1

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        initial_state = None

        if gdn_recurrent_states is not None:
            if context.block_tables is not None:
                if gdn_state_slots is not None:
                    initial_state = gdn_recurrent_states[
                        self.layer_idx, gdn_state_slots[:num_seqs]
                    ]
                else:
                    initial_state = gdn_recurrent_states[self.layer_idx, :num_seqs]
                # Force fresh (first-chunk) sequences to start from zero state;
                # block_tables is batch-global so a fresh seq batched with a
                # continuation would otherwise inherit a stale reused slot.
                keep = self._continuation_keep_mask(
                    context, num_seqs, initial_state.dtype
                )
                if keep is not None:
                    initial_state = initial_state * keep.view(-1, 1, 1, 1)
            else:
                initial_state = gdn_recurrent_states.new_zeros(
                    num_seqs, self.num_v_heads, self.head_v_dim, self.head_k_dim
                )

        if self._has_flashinfer_prefill:
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
        elif self._has_fla:
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
        else:
            o, final_state = self._naive_gdn_prefill(
                q, k, v, g, beta, scale, cu_seqlens, initial_state
            )

        if gdn_recurrent_states is not None and final_state is not None:
            final_state = final_state.to(gdn_recurrent_states.dtype)
            if gdn_state_slots is not None:
                gdn_recurrent_states[self.layer_idx, gdn_state_slots[:num_seqs]] = (
                    final_state
                )
            else:
                gdn_recurrent_states[self.layer_idx, :num_seqs] = final_state

        return o

    def _gdn_decode(self, q, k, v, a, b, scale, context) -> torch.Tensor:
        """Decode with FlashInfer, FLA, or a naive recurrent fallback.

        The preferred FlashInfer pretranspose kernel takes raw A_log, a,
        dt_bias, b and supports direct pool indexing. Builds without that
        optional backend use the nontranspose kernel with gather/scatter.
        State layout is K-last [pool_size, H, V, K].
        """
        bs = q.shape[0]

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)

        if self._has_flashinfer_pretranspose and gdn_recurrent_states is not None:
            state_pool = gdn_recurrent_states[self.layer_idx]
            if gdn_state_slots is not None:
                indices = gdn_state_slots[:bs].to(torch.int32)
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
            o = o.squeeze(1)
        elif self._has_flashinfer_nontranspose and gdn_recurrent_states is not None:
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:bs]
                initial_state = gdn_recurrent_states[self.layer_idx, slots]
            else:
                slots = None
                initial_state = gdn_recurrent_states[self.layer_idx, :bs]

            # The shared state pool is V-major/K-last. FlashInfer's alternate
            # decode backend expects K-major/V-last, so gather a contiguous
            # batch view, transpose it for the call, then scatter it back.
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
            updated_state = updated_state.transpose(-1, -2)
            updated_state = updated_state.to(gdn_recurrent_states.dtype)
            if slots is not None:
                gdn_recurrent_states[self.layer_idx, slots] = updated_state
            else:
                gdn_recurrent_states[self.layer_idx, :bs] = updated_state
        elif self._has_fla and gdn_recurrent_states is not None:
            if gdn_state_slots is not None:
                initial_state = gdn_recurrent_states[
                    self.layer_idx, gdn_state_slots[:bs]
                ]
            else:
                initial_state = gdn_recurrent_states[self.layer_idx, :bs]
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
            updated_state = updated_state.to(gdn_recurrent_states.dtype)
            if gdn_state_slots is not None:
                gdn_recurrent_states[self.layer_idx, gdn_state_slots[:bs]] = (
                    updated_state
                )
            else:
                gdn_recurrent_states[self.layer_idx, :bs] = updated_state
        else:
            beta = b.sigmoid()
            A_exp = -self.A_log.float().exp()
            g = A_exp * F.softplus(a.float() + self.dt_bias)

            if gdn_recurrent_states is not None:
                if gdn_state_slots is not None:
                    initial_state = gdn_recurrent_states[
                        self.layer_idx, gdn_state_slots[:bs]
                    ]
                else:
                    initial_state = gdn_recurrent_states[self.layer_idx, :bs]
            else:
                initial_state = q.new_zeros(
                    bs, self.num_v_heads, self.head_v_dim, self.head_k_dim
                )

            o, updated_state = self._naive_gdn_decode(
                q, k, v, g, beta, scale, initial_state
            )
            if gdn_recurrent_states is not None:
                if gdn_state_slots is not None:
                    gdn_recurrent_states[self.layer_idx, gdn_state_slots[:bs]] = (
                        updated_state
                    )
                else:
                    gdn_recurrent_states[self.layer_idx, :bs] = updated_state

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


__all__ = ["RecurrenceMixin"]
