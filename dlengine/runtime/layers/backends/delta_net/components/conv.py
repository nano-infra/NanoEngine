"""GDN causal depthwise convolution component (prefill + decode paths)."""

import torch
import torch.nn.functional as F

from . import kernels


class CausalConvMixin:
    """Causal depthwise conv1d over the concatenated QKV projection.

    Requires the host class to provide: ``conv1d`` (nn.Conv1d), ``conv_dim``,
    ``conv_kernel_size``, ``activation``, ``layer_idx``, and the padded-prefill
    workspace helper ``_get_conv1d_prefill_padded_workspace``. The continuation
    keep-mask helper ``_continuant_keep_mask`` is provided by StateMixin.
    """

    def _apply_conv1d(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Apply causal conv1d to concatenated QKV.

        Args:
            qkv: [total_tokens, conv_dim]
        """
        if context.is_prefill:
            if kernels.HAS_CAUSAL_CONV1D:
                return self._conv1d_prefill_fast(qkv, context)
            else:
                return self._conv1d_prefill_naive(qkv, context)
        else:
            return self._conv1d_decode(qkv, context)

    def _conv1d_prefill_fast(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Prefill conv1d using causal_conv1d_fn with seq_idx (single batched kernel)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        conv_weight = self.conv1d.weight.squeeze(1)

        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)

        has_prev_state = (
            context.block_tables is not None and gdn_conv_states is not None
        )

        if has_prev_state:
            # Chunked prefill (chunks 2+): pad variable-length sequences into
            # a fixed batch so we can call causal_conv1d_fn once with
            # initial_states, avoiding per-seq Python loops and D2H syncs.
            total_tokens = qkv.shape[0]
            max_seqlen = context.max_seqlen_q
            dim = qkv.shape[1]
            cu_seqlens_long = cu_seqlens.to(torch.int64)
            batch_values = torch.arange(num_seqs, device=qkv.device, dtype=torch.int64)
            offset_values = cu_seqlens_long[:-1].contiguous()

            if (
                kernels.repeat_interleave_from_prefix_triton is not None
                and kernels.can_use_repeat_interleave_from_prefix_triton is not None
                and kernels.can_use_repeat_interleave_from_prefix_triton(
                    batch_values, cu_seqlens_long
                )
            ):
                batch_idx = kernels.repeat_interleave_from_prefix_triton(
                    batch_values,
                    cu_seqlens_long,
                    total_tokens,
                    max_repeat_hint=max_seqlen,
                )
                offsets = kernels.repeat_interleave_from_prefix_triton(
                    offset_values,
                    cu_seqlens_long,
                    total_tokens,
                    max_repeat_hint=max_seqlen,
                )
            else:
                seq_lens = (cu_seqlens_long[1:] - cu_seqlens_long[:-1]).contiguous()
                batch_idx = torch.repeat_interleave(
                    torch.arange(num_seqs, device=qkv.device, dtype=torch.long),
                    seq_lens,
                    output_size=total_tokens,
                )
                offsets = torch.repeat_interleave(
                    cu_seqlens_long[:-1],
                    seq_lens,
                    output_size=total_tokens,
                )
            pos_in_seq = (
                torch.arange(
                    total_tokens,
                    device=qkv.device,
                    dtype=torch.long,
                )
                - offsets
            )

            padded = self._get_conv1d_prefill_padded_workspace(
                num_seqs,
                dim,
                max_seqlen,
                qkv.device,
                qkv.dtype,
            )
            if (
                kernels.ragged_to_padded_triton is not None
                and kernels.can_use_ragged_to_padded_triton is not None
                and kernels.can_use_ragged_to_padded_triton(qkv, cu_seqlens_long, padded)
            ):
                kernels.ragged_to_padded_triton(
                    qkv, cu_seqlens_long, max_seqlen, out=padded
                )
            else:
                padded[batch_idx, :, pos_in_seq] = qkv

            if gdn_state_slots is not None:
                init_states = gdn_conv_states[
                    self.layer_idx, gdn_state_slots[:num_seqs], :, 1:
                ]
            else:
                init_states = gdn_conv_states[self.layer_idx, :num_seqs, :, 1:]
            # Force fresh (first-chunk) sequences to start from a zero conv
            # state (see _continuation_keep_mask); guards against stale state in
            # a reused slot when a fresh seq is batched with a continuation.
            keep = self._continuation_keep_mask(context, num_seqs, init_states.dtype)
            if keep is not None:
                init_states = init_states * keep.view(-1, 1, 1)
            if not init_states.is_contiguous():
                init_states = init_states.contiguous()

            padded_out = kernels.causal_conv1d_fn(
                x=padded,
                weight=conv_weight,
                initial_states=init_states,
                activation=self.activation,
            )

            qkv_out = padded_out[batch_idx, :, pos_in_seq]
        else:
            # First chunk or single-chunk: batched with seq_idx
            cu_seqlens_long = cu_seqlens.to(torch.int64)
            seq_values = torch.arange(num_seqs, dtype=torch.int32, device=qkv.device)
            if (
                kernels.repeat_interleave_from_prefix_triton is not None
                and kernels.can_use_repeat_interleave_from_prefix_triton is not None
                and kernels.can_use_repeat_interleave_from_prefix_triton(
                    seq_values, cu_seqlens_long
                )
            ):
                seq_idx = kernels.repeat_interleave_from_prefix_triton(
                    seq_values,
                    cu_seqlens_long,
                    qkv.shape[0],
                    max_repeat_hint=context.max_seqlen_q,
                ).unsqueeze(0)
            else:
                seq_lens = (cu_seqlens_long[1:] - cu_seqlens_long[:-1]).contiguous()
                seq_idx = torch.repeat_interleave(
                    torch.arange(num_seqs, dtype=torch.int32, device=qkv.device),
                    seq_lens,
                ).unsqueeze(0)

            qkv_out = (
                kernels.causal_conv1d_fn(
                    x=qkv.T.unsqueeze(0),
                    weight=conv_weight,
                    bias=None,
                    seq_idx=seq_idx,
                    activation=self.activation,
                )
                .squeeze(0)
                .T
            )

        # Store conv states for future chunks/decode — batched extraction
        if gdn_conv_states is not None:
            states = kernels.causal_conv1d_varlen_states(
                qkv, cu_seqlens, self.conv_kernel_size - 1
            )
            if gdn_state_slots is not None:
                target_states = gdn_conv_states[
                    self.layer_idx, gdn_state_slots[:num_seqs]
                ]
            else:
                target_states = gdn_conv_states[self.layer_idx, :num_seqs]
            target_states.zero_()
            target_states[:, :, 1:].copy_(states)

        return qkv_out

    def _conv1d_prefill_naive(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Prefill conv1d using PyTorch (per-sequence, fallback)."""
        cu_seqlens = context.cu_seqlens_q
        num_seqs = cu_seqlens.shape[0] - 1
        qkv_out = torch.empty_like(qkv)

        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        has_prev_state = (
            context.block_tables is not None and gdn_conv_states is not None
        )

        conv_weight = self.conv1d.weight  # [conv_dim, 1, kernel_size]

        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            if end <= start:
                continue
            seq_qkv = qkv[start:end].unsqueeze(0).transpose(1, 2)  # [1, D, L]
            conv_dtype = conv_weight.dtype
            conv_input = seq_qkv.to(conv_dtype)

            if has_prev_state:
                slot = gdn_state_slots[i].item() if gdn_state_slots is not None else i
                prev = gdn_conv_states[self.layer_idx, slot, :, 1:].unsqueeze(
                    0
                )  # [1, D, k-1]
                padded = torch.cat([prev.to(conv_dtype), conv_input], dim=2)
                seq_out = F.silu(
                    F.conv1d(padded, conv_weight, groups=self.conv_dim)
                )  # [1, D, L]
            else:
                seq_out = F.silu(self.conv1d(conv_input)[:, :, : end - start])

            qkv_out[start:end] = seq_out.to(qkv.dtype).squeeze(0).transpose(0, 1)

        # Store conv state
        if gdn_conv_states is not None:
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                if end <= start:
                    continue
                seq_len = end - start
                pad_len = min(seq_len, self.conv_kernel_size - 1)
                slot = gdn_state_slots[i].item() if gdn_state_slots is not None else i
                gdn_conv_states[self.layer_idx, slot, :, :] = 0
                gdn_conv_states[self.layer_idx, slot, :, -pad_len:] = qkv[
                    end - pad_len : end
                ].T

        return qkv_out

    def _conv1d_decode(self, qkv: torch.Tensor, context) -> torch.Tensor:
        """Decode conv1d: single token per sequence, update conv state."""
        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        bs = qkv.shape[0]

        if gdn_conv_states is not None and kernels.HAS_CAUSAL_CONV1D:
            conv_weight = self.conv1d.weight.squeeze(1)
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:bs]
                conv_state = gdn_conv_states[self.layer_idx, slots]
            else:
                conv_state = gdn_conv_states[self.layer_idx, :bs]
            qkv_out = kernels.causal_conv1d_update(
                qkv,
                conv_state,
                conv_weight,
                bias=None,
                activation=self.activation,
            )
            if gdn_state_slots is not None:
                gdn_conv_states[self.layer_idx, slots] = conv_state
            return qkv_out
        elif gdn_conv_states is not None:
            if gdn_state_slots is not None:
                slots = gdn_state_slots[:bs]
                conv_state = gdn_conv_states[self.layer_idx, slots].clone()
            else:
                conv_state = gdn_conv_states[self.layer_idx, :bs]
            conv_state[:, :, :-1] = conv_state[:, :, 1:].clone()
            conv_state[:, :, -1] = qkv
            conv_weight = self.conv1d.weight.squeeze(1)
            qkv_out = F.silu(
                (conv_state.to(conv_weight.dtype) * conv_weight.unsqueeze(0)).sum(-1)
            ).to(qkv.dtype)
            if gdn_state_slots is not None:
                gdn_conv_states[self.layer_idx, slots] = conv_state
            return qkv_out
        else:
            return F.silu(qkv)


__all__ = ["CausalConvMixin"]
