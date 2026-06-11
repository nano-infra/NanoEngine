import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dlengine.context.context import get_context
from dlengine.context.distributed import get_dist_context

# MCCL (MetaX CCL) has issues with large tensor all_reduce/all_gather:
# - all_gather deadlocks after the first call (P2P path issue)
# - all_reduce deadlocks when tensor > ~32k elements
# Workaround: use chunked all_reduce with small chunks.
# Enable via: DLENGINE_CHUNKED_ALLREDUCE=1 (default off)
_ALLREDUCE_CHUNK_SIZE = 32768


def _chunked_all_reduce(tensor: torch.Tensor, group) -> None:
    """All-reduce, optionally chunked for MCCL large-tensor deadlock workaround.

    Enable chunking via env var DLENGINE_CHUNKED_ALLREDUCE=1 (for MetaX GPUs).
    Default: plain all_reduce (no chunking, no sync overhead).
    """
    # Check env at call time (not import) since Ray workers set env in __init__
    if os.getenv("DLENGINE_CHUNKED_ALLREDUCE", "0") != "1":
        # Normal path: plain all_reduce, no chunking
        dist.all_reduce(tensor, group=group)
        return

    # Chunked path for MCCL workaround
    if torch.cuda.is_current_stream_capturing():
        # Graph capture: no sync allowed, tensors are small anyway
        dist.all_reduce(tensor, group=group)
        return

    numel = tensor.numel()
    if numel <= _ALLREDUCE_CHUNK_SIZE:
        dist.all_reduce(tensor, group=group)
    else:
        flat = tensor.view(-1)
        for i in range(0, numel, _ALLREDUCE_CHUNK_SIZE):
            end = min(i + _ALLREDUCE_CHUNK_SIZE, numel)
            chunk = flat[i:end].contiguous()
            dist.all_reduce(chunk, group=group)
            flat[i:end] = chunk


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = get_dist_context().attn_tp_rank
        self.tp_size = get_dist_context().attn_tp_world_size
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
        self.weight.weight_loader = self.weight_loader

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None
    ):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            _chunked_all_reduce(y, get_dist_context().attn_tp_group)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            if context.sampling_token_indices is not None:
                # Chunked prefill: only extract hidden states for final-chunk sequences.
                x = x[context.sampling_token_indices].contiguous()
            else:
                last_indices = context.cu_seqlens_q[1:] - 1
                x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            # Use all_reduce instead of all_gather. On this hardware's MCCL
            # backend, all_gather deadlocks after the first call (P2P issue),
            # while all_reduce works reliably (with chunking for large tensors).
            # Strategy: each rank places its shard into a full-vocab buffer at
            # the correct offset, then all_reduce (sum) gives the complete
            # logits on every rank. Only rank 0 needs the result for sampling.
            batch_size = logits.size(0)
            vocab_per_rank = logits.size(-1)
            full_vocab = vocab_per_rank * self.tp_size
            full_logits = torch.zeros(
                batch_size, full_vocab, device=logits.device, dtype=logits.dtype
            )
            start = self.tp_rank * vocab_per_rank
            full_logits[:, start : start + vocab_per_rank] = logits
            _chunked_all_reduce(full_logits, get_dist_context().attn_tp_group)
            logits = full_logits if self.tp_rank == 0 else None
        return logits
