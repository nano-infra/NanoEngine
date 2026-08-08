from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from nanodeploy.kernels.deep_gemm_backend import (
    m_grouped_fp8_gemm_nt_masked,
)
from nanodeploy.kernels.moe_fp8 import (
    fused_moe_fp8_contiguous,
    per_token_group_quant_fp8,
    silu_and_mul_masked_post_quant_fwd,
)
from nanodeploy.layers.token_dispatcher import (
    DeepEPTokenDispatcherLowLatency,
    DeepEPTokenDispatcherNormal,
)


_gemm_debug_call_counts: dict[tuple[int, int, str], int] = {}


def _debug_values(name: str, default: str) -> set[str]:
    return {
        value.strip().lower()
        for value in os.getenv(name, default).split(",")
        if value.strip()
    }


def _tensor_metadata(tensor: torch.Tensor, sample_elements: int, sample=True):
    metadata = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "is_contiguous": tensor.is_contiguous(),
        "numel": tensor.numel(),
    }
    if sample and sample_elements > 0 and tensor.numel() > 0:
        values = tensor.detach().reshape(-1)[:sample_elements]
        if torch.is_floating_point(values):
            values = values.float()
        metadata["sample"] = values.cpu().tolist()
    return metadata


def _allocate_column_major_scales(
    groups: int,
    max_m: int,
    scale_groups: int,
    device: torch.device,
) -> torch.Tensor:
    backing = torch.empty(
        (groups, scale_groups, max_m),
        device=device,
        dtype=torch.float32,
    )
    scales = backing.permute(0, 2, 1)
    expected_stride = (max_m * scale_groups, 1, max_m)
    if scales.stride() != expected_stride:
        raise RuntimeError("failed to create column-major DeepGEMM scales")
    return scales


class FusedMoENormal:
    def __init__(
        self,
        *,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int = 0,
        block_size: int = 128,
        top_k: int = 8,
        out_dtype: torch.dtype = torch.bfloat16,
        chunk_size: Optional[int] = 32 * 1024,
        expert_alignment: int = 128,
    ) -> None:
        del top_k, chunk_size
        self.layer_index = layer_index
        self.block_size = block_size
        self.token_dispatcher = DeepEPTokenDispatcherNormal(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
            expert_alignment=expert_alignment,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: Optional[List[int]] = None,
    ) -> torch.Tensor:
        quantized = per_token_group_quant_fp8(hidden_states, self.block_size)
        recv_x, recv_ids, recv_weights, counts = self.token_dispatcher.dispatch(
            quantized, topk_ids, topk_weights, expert_list
        )
        local_output = fused_moe_fp8_contiguous(
            recv_x,
            recv_ids,
            recv_weights,
            (up_weights, up_scale),
            (down_weights, down_scale),
            counts,
            block_size=self.block_size,
        )
        return self.token_dispatcher.combine(local_output)


class FusedMoELowLatency:
    def __init__(
        self,
        *,
        ep_size: int,
        ep_group: dist.ProcessGroup,
        num_experts: int,
        hidden_dim: int,
        layer_index: int,
        block_size: int = 128,
        top_k: int = 8,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        del top_k
        self.layer_index = layer_index
        self.block_size = block_size
        self.out_dtype = out_dtype
        self.token_dispatcher = DeepEPTokenDispatcherLowLatency(
            group=ep_group,
            num_experts=num_experts,
            num_local_experts=num_experts // ep_size,
            hidden_size=hidden_dim,
            params_dtype=out_dtype,
        )

    def _debug_gemm(
        self,
        name: str,
        activation: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> None:
        if os.getenv("NANODEPLOY_MOE_GEMM_DEBUG", "0").lower() not in {
            "1", "true", "yes", "on"
        }:
            return
        rank = dist.get_rank() if dist.is_initialized() else -1
        ranks = _debug_values("NANODEPLOY_MOE_GEMM_DEBUG_RANKS", "all")
        layers = _debug_values("NANODEPLOY_MOE_GEMM_DEBUG_LAYERS", "1")
        gemms = _debug_values("NANODEPLOY_MOE_GEMM_DEBUG_GEMMS", "gate_up")
        if "all" not in ranks and "*" not in ranks and str(rank) not in ranks:
            return
        if (
            "all" not in layers
            and "*" not in layers
            and str(self.layer_index) not in layers
        ):
            return
        if "all" not in gemms and "*" not in gemms and name.lower() not in gemms:
            return
        key = (rank, self.layer_index, name)
        index = _gemm_debug_call_counts.get(key, 0)
        _gemm_debug_call_counts[key] = index + 1
        max_calls = int(os.getenv("NANODEPLOY_MOE_GEMM_DEBUG_MAX_CALLS", "1"))
        if max_calls > 0 and index >= max_calls:
            return
        sample = max(
            0, int(os.getenv("NANODEPLOY_MOE_GEMM_DEBUG_SAMPLE_ELEMENTS", "8"))
        )
        counts = masked_m.detach().to(device="cpu", dtype=torch.int64).tolist()
        payload = {
            "event": "post_dispatch_pre_gemm",
            "gemm": name,
            "global_rank": rank,
            "layer_index": self.layer_index,
            "call_index": index,
            "expected_m": int(expected_m),
            "masked_m": counts,
            "masked_m_sum": sum(counts),
            "input": _tensor_metadata(activation[0], sample),
            "input_scale": _tensor_metadata(activation[1], sample),
            "weight": _tensor_metadata(weight[0], sample),
            "weight_scale": _tensor_metadata(weight[1], sample),
            "output": _tensor_metadata(output, sample, sample=False),
        }
        print(
            "[NANODEPLOY_MOE_GEMM_INPUT] "
            + json.dumps(payload, sort_keys=True, separators=(",", ":")),
            flush=True,
        )

    def _gemm(
        self,
        name: str,
        activation: Tuple[torch.Tensor, torch.Tensor],
        weight: Tuple[torch.Tensor, torch.Tensor],
        output: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> None:
        self._debug_gemm(
            name, activation, weight, output, masked_m, expected_m
        )
        m_grouped_fp8_gemm_nt_masked(
            activation, weight, output, masked_m, expected_m
        )

    def _experts(
        self,
        hidden_states: Tuple[torch.Tensor, torch.Tensor],
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> torch.Tensor:
        groups, max_m, _hidden = hidden_states[0].shape
        gate_up = torch.empty(
            (groups, max_m, up_weights.shape[1]),
            device=hidden_states[0].device,
            dtype=self.out_dtype,
        )
        self._gemm(
            "gate_up",
            hidden_states,
            (up_weights, up_scale),
            gate_up,
            masked_m,
            expected_m,
        )
        intermediate = gate_up.shape[2] // 2
        down_input = torch.empty(
            (groups, max_m, intermediate),
            device=gate_up.device,
            dtype=torch.float8_e4m3fn,
        )
        scale_groups = intermediate // self.block_size
        down_input_scale = _allocate_column_major_scales(
            groups, max_m, scale_groups, gate_up.device
        )
        silu_and_mul_masked_post_quant_fwd(
            gate_up,
            down_input,
            down_input_scale,
            self.block_size,
            masked_m,
        )
        output = torch.empty(
            (groups, max_m, down_weights.shape[1]),
            device=gate_up.device,
            dtype=self.out_dtype,
        )
        self._gemm(
            "down",
            (down_input, down_input_scale),
            (down_weights, down_scale),
            output,
            masked_m,
            expected_m,
        )
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        up_weights: torch.Tensor,
        up_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_scale: torch.Tensor,
        expert_list: Optional[List[int]] = None,
    ) -> torch.Tensor:
        del expert_list
        packed, ids, weights, masked_m, expected_m = (
            self.token_dispatcher.dispatch(
                hidden_states, topk_ids, topk_weights
            )
        )
        local_output = self._experts(
            packed,
            up_weights,
            up_scale,
            down_weights,
            down_scale,
            masked_m,
            expected_m,
        )
        return self.token_dispatcher.combine(local_output, ids, weights)


def build_deepep_moe(
    *,
    low_latency_mode: bool,
    ep_size: int,
    ep_group: dist.ProcessGroup,
    num_experts: int,
    hidden_dim: int,
    block_size: int = 128,
    top_k: int = 8,
    out_dtype: torch.dtype = torch.bfloat16,
    layer_idx: int = 0,
    chunk_size: Optional[int] = 32 * 1024,
):
    common = dict(
        ep_size=ep_size,
        ep_group=ep_group,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        block_size=block_size,
        top_k=top_k,
        out_dtype=out_dtype,
        layer_index=layer_idx,
    )
    if low_latency_mode:
        return FusedMoELowLatency(**common)
    return FusedMoENormal(**common, chunk_size=chunk_size)


__all__ = ["FusedMoELowLatency", "FusedMoENormal", "build_deepep_moe"]
