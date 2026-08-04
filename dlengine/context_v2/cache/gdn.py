from dataclasses import dataclass

import torch

from dlengine.context_v2 import BaseContext
from dlengine.logging import get_logger
from dlengine.utils.cuda import get_cuda_compute_capability


def _recurrent_state_dtype() -> torch.dtype:
    """Use FlashInfer's bf16 GDN state-pool kernel on SM100+."""
    capability = get_cuda_compute_capability()
    if capability is not None and capability[0] >= 10:
        return torch.bfloat16
    return torch.float32


logger = get_logger("dlengine")


@dataclass
class GDNContext(BaseContext):
    gdn_conv_states: torch.Tensor | None = None
    gdn_recurrent_states: torch.Tensor | None = None
    gdn_num_slots: int = 0
    gdn_max_active_slots: int = 0

    @classmethod
    def get_context_type(cls) -> str:
        return "gdn"

    @classmethod
    def get_context_name(cls) -> str:
        return "GDNContext"

    def clear_context(self) -> None:
        self.gdn_conv_states = None
        self.gdn_recurrent_states = None
        self.gdn_num_slots = 0
        self.gdn_max_active_slots = 0

    def reset_context(self) -> None:
        # GDN state buffers are persistent cache-backed runtime state, like a
        # KV cache. They must survive per-step runtime-context resets so the
        # decode step can continue from the recurrent state written by prefill.
        pass


_GDN_CONTEXT = GDNContext()


def get_gdn_context() -> GDNContext:
    return _GDN_CONTEXT


def reset_gdn_context() -> None:
    _GDN_CONTEXT.reset_context()


def initialize_gdn_cache_state(context) -> None:
    context.gdn_num_slots = 0


def estimate_gdn_state_bytes(
    hf_config,
    layer_types,
    max_bs: int,
    need_backup: bool = False,
    cache_slots: int = 0,
    attention_tp: int = 1,
) -> int:
    """Per-rank bytes the GDN conv + recurrent state buffers will occupy."""
    if not layer_types:
        return 0
    num_layers = len(layer_types)
    num_k_heads = getattr(hf_config, "linear_num_key_heads", 0) // attention_tp
    num_v_heads = getattr(hf_config, "linear_num_value_heads", 0) // attention_tp
    head_k_dim = getattr(hf_config, "linear_key_head_dim", 0)
    head_v_dim = getattr(hf_config, "linear_value_head_dim", 0)
    conv_kernel_size = getattr(hf_config, "linear_conv_kernel_dim", 4)
    if num_v_heads == 0:
        return 0
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim
    active_capacity = max_bs + max(0, cache_slots)
    num_slots = active_capacity * 2 + 1 if need_backup else active_capacity + 1
    conv_bytes = num_layers * num_slots * conv_dim * conv_kernel_size * 2
    state_size = torch.empty((), dtype=_recurrent_state_dtype()).element_size()
    recurrent_bytes = (
        num_layers
        * num_slots
        * num_v_heads
        * head_v_dim
        * head_k_dim
        * state_size
    )
    return conv_bytes + recurrent_bytes


def allocate_gdn_states(
    context,
    hf_config,
    layer_types,
    max_bs: int,
    need_backup: bool = False,
    cache_slots: int = 0,
    attention_tp: int | None = None,
) -> None:
    """Allocate fixed-size GDN state buffers for linear_attention layers."""
    max_bs = max_bs + max(0, cache_slots)
    num_layers = len(layer_types)
    attention_tp = context.attention_tp if attention_tp is None else attention_tp
    num_k_heads = getattr(hf_config, "linear_num_key_heads", 0) // attention_tp
    num_v_heads = getattr(hf_config, "linear_num_value_heads", 0) // attention_tp
    head_k_dim = getattr(hf_config, "linear_key_head_dim", 0)
    head_v_dim = getattr(hf_config, "linear_value_head_dim", 0)
    conv_kernel_size = getattr(hf_config, "linear_conv_kernel_dim", 4)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = key_dim * 2 + value_dim

    if num_v_heads == 0:
        return

    num_slots = max_bs * 2 + 1 if need_backup else max_bs + 1
    context.gdn_num_slots = num_slots
    context.gdn_max_active_slots = max_bs

    context.gdn_conv_states = torch.zeros(
        num_layers,
        num_slots,
        conv_dim,
        conv_kernel_size,
        dtype=torch.bfloat16,
        device=context.device,
    )

    context.gdn_recurrent_states = torch.zeros(
        num_layers,
        num_slots,
        num_v_heads,
        head_v_dim,
        head_k_dim,
        dtype=_recurrent_state_dtype(),
        device=context.device,
    )

    if need_backup:
        slot_info = (
            f"active_slots=0..{max_bs-1}, backup_slots={max_bs}..{2*max_bs-1}, "
            f"dummy_slot={2*max_bs}"
        )
    else:
        slot_info = f"active_slots=0..{max_bs-1}, dummy_slot={max_bs}"
    logger.debug(
        f"Allocated GDN states: conv={context.gdn_conv_states.shape} "
        f"({context.gdn_conv_states.element_size() * context.gdn_conv_states.nelement() / 1e9:.2f} GB), "
        f"recurrent={context.gdn_recurrent_states.shape} "
        f"({context.gdn_recurrent_states.element_size() * context.gdn_recurrent_states.nelement() / 1e9:.2f} GB), "
        f"{slot_info}"
    )


__all__ = [
    "GDNContext",
    "allocate_gdn_states",
    "estimate_gdn_state_bytes",
    "get_gdn_context",
    "initialize_gdn_cache_state",
    "reset_gdn_context",
]
