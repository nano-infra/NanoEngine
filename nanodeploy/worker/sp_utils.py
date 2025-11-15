import torch
from typing import List


def build_all_gather_q_mask(
    cur_rank_req_kv_map: List[List[int]],
    max_bs: int,
    gpus_per_machine: int,
    cur_rank: int,
    device: str = "cuda"
) -> torch.Tensor:
    num_requests = len(cur_rank_req_kv_map)
    
    assert num_requests <= max_bs, \
        f"Number of requests ({num_requests}) cannot exceed max_bs ({max_bs})"

    mask = torch.zeros((max_bs, gpus_per_machine), dtype=torch.int32, device=device)

    for i, ranks_for_this_request in enumerate(cur_rank_req_kv_map):
        if ranks_for_this_request:
            if len(ranks_for_this_request) == 1:
                if ranks_for_this_request[0] != cur_rank:
                    mask[i, ranks_for_this_request[0]] = 1
            else:
                for rank in ranks_for_this_request:
                    mask[i, rank] = 1
    return mask

def count_requests_to_cur_rank(cur_rank: int, group_req_kv_map: list) -> List[List[int]]:
    """计算每个Rank中包含当前Rank的请求的索引列表"""
    num_ranks = len(group_req_kv_map)
    result = [[] for _ in range(num_ranks)]
    
    for m in range(num_ranks):
        kv_ranks_list = group_req_kv_map[m]
        for req_idx, kv_ranks in enumerate(kv_ranks_list):
            if cur_rank in kv_ranks:
                result[m].append(req_idx)

    return result

def build_all2all_res_mask(
    count_requests_to_cur_rank_res: List[List[int]],
    max_bs: int,
    gpus_per_machine: int,
    device: str = "cuda"
) -> torch.Tensor:
    """基于请求索引列表创建全收集掩码张量"""
    mask = torch.zeros((max_bs, gpus_per_machine), dtype=torch.int, device=device)
    
    for src_rank in range(gpus_per_machine):
        request_indices = count_requests_to_cur_rank_res[src_rank]
        
        for req_idx in request_indices:
            if req_idx < max_bs:
                mask[req_idx, src_rank] = 1
    
    return mask

def build_lse_mask(
    cur_rank_req_kv_map: List[List[int]],
    sp_q_max_bs: int,
    gpus_per_machine: int,
    device: str = "cuda"
) -> torch.Tensor:
    """为 LSE 结果合并构建掩码"""
    num_requests = len(cur_rank_req_kv_map)
    
    assert num_requests <= sp_q_max_bs, \
        f"Number of requests ({num_requests}) cannot exceed sp_q_max_bs ({sp_q_max_bs})"

    mask = torch.zeros((sp_q_max_bs, gpus_per_machine), dtype=torch.float32, device=device)

    for i, ranks_for_this_request in enumerate(cur_rank_req_kv_map):
        if ranks_for_this_request:
            rank_indices = torch.tensor(ranks_for_this_request, dtype=torch.long, device=device)
            mask[i, rank_indices] = 1.0
            
    return mask

def build_kv_lens_slice(global_rank: int, group_req_kv_map: List[List[List[int]]], max_bs: int, sp_rank: int):
    """
    计算在 CUDA Graph replay 时，如何将实际的 cache_seqlens 和 block_table 填充到预分配的缓冲区中
    
    在 SP 模式下，不同 rank 的请求数量可能不同，但 CUDA Graph 需要固定大小的缓冲区。
    这个函数计算：
    1. slice_to_get: 从实际数据中获取哪些索引
    2. slice_to_fill: 将数据填充到缓冲区的哪些位置
    
    Args:
        global_rank: 当前全局 rank ID (用于调试)
        group_req_kv_map: KV cache 分布映射 [sp_rank][request_idx] = [kv_rank1, kv_rank2, ...]
        max_bs: CUDA Graph 缓冲区的最大 batch size
        sp_rank: 当前 SP rank
        
    Returns:
        (slice_to_get, slice_to_fill): 两个索引列表
        - slice_to_get: 从压缩的实际数据中读取的索引
        - slice_to_fill: 写入到预分配缓冲区的索引
        
    Example:
        假设 sp_size=2, max_bs=4
        group_req_kv_map = [
            [[0], [1]],      # rank0 有 2 个请求，KV 在 rank0 和 rank1
            [[0], [1]]       # rank1 有 2 个请求，KV 在 rank0 和 rank1
        ]
        对于 sp_rank=0:
        - 它需要处理所有包含 rank0 KV 的请求
        - rank0 的请求 0 包含 rank0 KV -> result[0].append(0)
        - rank0 的请求 1 包含 rank1 KV -> 跳过
        - rank1 的请求 0 包含 rank0 KV -> result[1].append(0)
        - rank1 的请求 1 包含 rank1 KV -> 跳过
        - result = [[0], [0]]
        - slice_to_get = [0, 1]  (连续索引)
        - slice_to_fill = [0, 4] (rank0*max_bs+0, rank1*max_bs+0)
    """
    num_ranks = len(group_req_kv_map)
    result = [[] for _ in range(num_ranks)]
    
    # 找出每个 rank 中包含当前 sp_rank KV cache 的请求
    for rank in range(num_ranks):
        kv_ranks_list = group_req_kv_map[rank]
        for req_idx, kv_ranks in enumerate(kv_ranks_list):
            if sp_rank in kv_ranks:
                result[rank].append(req_idx)
    
    # slice_to_get: 从实际数据中按顺序读取
    total_elements = sum(len(req_list) for req_list in result)
    slice_to_get = list(range(total_elements))
    
    # slice_to_fill: 写入到预分配缓冲区的对应位置
    # 缓冲区布局: [rank0_req0, rank0_req1, ..., rank1_req0, rank1_req1, ...]
    slice_to_fill = []
    for rank in range(num_ranks):
        for req_idx in result[rank]:
            buffer_pos = rank * max_bs + req_idx
            slice_to_fill.append(buffer_pos)
    
    return slice_to_get, slice_to_fill