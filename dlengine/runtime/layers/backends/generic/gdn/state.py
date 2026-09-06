"""GDN recurrent-state hygiene component.

Owns the persistent-pool state handling shared by every GDN backend: zeroing
fresh (first-chunk) slots before prefill reads them, and computing the
continuation keep-mask that distinguishes fresh sequences from chunked-prefill
continuations.
"""

import torch


class StateMixin:
    """Recurrent-state slot hygiene. Requires the host to provide ``layer_idx``."""

    def _zero_fresh_slots(self, context) -> None:
        """Zero the conv + recurrent state slots of fresh (first-chunk) seqs
        before prefill reads them.

        The GDN state pool is persistent across requests and is *not* cleared
        when a slot is recycled, so a slot handed to a new sequence still holds
        the previous occupant's final state. In a mixed prefill batch (a fresh
        sequence batched with a chunked-prefill continuation) ``block_tables``
        is set batch-globally, so several read paths — notably the naive conv1d
        fallback (``_conv1d_prefill_naive``), which has no per-seq mask — would
        otherwise pick up that stale state. Zeroing the fresh sequences' slots
        in place here neutralises the leak for every downstream path
        (fast/naive, conv/recurrent) in a single sync-free scatter-multiply;
        continuation sequences (keep == 1) retain their state untouched.

        Only needed when ``block_tables`` is set: when it is ``None`` no
        sequence in the batch has cached tokens, so every prefill path already
        starts from a freshly-allocated zero state.
        """
        if context.block_tables is None:
            return
        gdn_state_slots = getattr(context, "gdn_state_slots", None)
        if gdn_state_slots is None:
            return
        cu_q = getattr(context, "cu_seqlens_q", None)
        if cu_q is None:
            return
        num_seqs = cu_q.shape[0] - 1
        if num_seqs <= 0:
            return
        keep = self._continuation_keep_mask(context, num_seqs, torch.float32)
        if keep is None:
            return
        slots = gdn_state_slots[:num_seqs].long()

        gdn_recurrent_states = getattr(context, "gdn_recurrent_states", None)
        if gdn_recurrent_states is not None:
            keep_r = keep.view(-1, 1, 1, 1).to(gdn_recurrent_states.dtype)
            gdn_recurrent_states[self.layer_idx, slots] = (
                gdn_recurrent_states[self.layer_idx, slots] * keep_r
            )

        gdn_conv_states = getattr(context, "gdn_conv_states", None)
        if gdn_conv_states is not None:
            keep_c = keep.view(-1, 1, 1).to(gdn_conv_states.dtype)
            gdn_conv_states[self.layer_idx, slots] = (
                gdn_conv_states[self.layer_idx, slots] * keep_c
            )

    @staticmethod
    def _continuation_keep_mask(context, num_seqs: int, dtype: torch.dtype):
        """Per-seq multiplier: 1.0 for sequences that continue from a cached
        recurrent state (chunked-prefill chunk 2+), 0.0 for fresh first-chunk
        sequences.

        ``block_tables is not None`` is a *batch-global* flag, so a fresh
        sequence (no cached tokens) batched together with a chunked-prefill
        continuation would otherwise read stale conv/recurrent state left in
        its reused slot by a previous sequence. Multiplying the gathered
        initial state by this mask forces fresh sequences to start from zero
        without leaking across requests. Computed from cu_seqlens (cached =
        seqlen_k - seqlen_q) so it stays sync-free (no ``.item()``/``bool()``).
        """
        cu_q = getattr(context, "cu_seqlens_q", None)
        cu_k = getattr(context, "cu_seqlens_k", None)
        if cu_q is None or cu_k is None:
            return None
        cu_q = cu_q.long()
        cu_k = cu_k.long()
        seqlen_q = cu_q[1 : num_seqs + 1] - cu_q[:num_seqs]
        seqlen_k = cu_k[1 : num_seqs + 1] - cu_k[:num_seqs]
        # cached tokens = seqlen_k - seqlen_q; > 0 only for continuations.
        return (seqlen_k > seqlen_q).to(dtype)


__all__ = ["StateMixin"]
