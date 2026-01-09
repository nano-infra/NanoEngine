import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanodeploy.worker.context import get_context
from nanodeploy.worker.distributed import get_dist_context
from torch import nn


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
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str = None, ckpt_shard_id=None, ckpt_num_shards=None, **kwargs
    ):
        param_data = param.data
        shard_size = param_data.size(0)
        
        if loaded_weight.size(0) == shard_size:
            if ckpt_shard_id is not None and self.tp_size > 1:
                # If we have shard info, only load if it matches our rank
                # This assumes 1:1 mapping between ckpt shards and TP ranks
                if ckpt_shard_id == self.tp_rank:
                    param_data.copy_(loaded_weight)
            else:
                param_data.copy_(loaded_weight)
        elif loaded_weight.size(0) > shard_size:
             start_idx = self.tp_rank * shard_size
             if start_idx + shard_size <= loaded_weight.size(0):
                loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
                param_data.copy_(loaded_weight)
        # Handle cases where loaded weight is a subset but not matching exact shard? 
        # For now, this covers the "full weight" and "exact shard" cases.

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y, group=get_dist_context().attn_tp_group)
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
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0
                else None
            )
            dist.gather(
                logits,
                all_logits,
                dist.get_process_group_ranks(get_dist_context().attn_tp_group)[0],
                group=get_dist_context().attn_tp_group,
            )
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
