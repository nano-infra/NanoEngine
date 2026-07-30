from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
from dlengine._rust.proto import decode_flat_control
from dlengine.kernel.jit.nano.decode_metadata import (
    launch,
    launch_graph,
    register_mapped,
    unregister_mapped,
)

_QWEN35_ARCHITECTURES = {
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
}


def decode_metadata_enabled(config) -> bool:
    architectures = getattr(config.hf_config, "architectures", None) or []
    return bool(
        getattr(config, "use_decode_metadata_kernel", True)
        and architectures
        and architectures[0] in _QWEN35_ARCHITECTURES
        and not config.enforce_eager
        and config.num_speculative_tokens == 0
        and not getattr(config, "enable_hisparse", False)
    )


def _align(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) // alignment * alignment


def decode_payload_capacity(max_num_seqs: int, max_num_blocks: int) -> int:
    sizes = (
        max_num_seqs * 8,  # input_ids
        max_num_seqs * 8,  # positions
        max_num_seqs * 4,  # temperatures
        max_num_seqs * 8,  # state_slots
        max_num_seqs * 8,  # hisparse_slots
        (max_num_seqs + 1) * 4,  # block row offsets
        max_num_seqs * max_num_blocks * 4,  # block ids
        max_num_seqs * 8,  # seq_ids
    )
    capacity = 64
    for size in sizes:
        capacity = _align(capacity)
        capacity += size
    return _align(capacity)


@dataclass(frozen=True)
class DecodeMetadataViews:
    input_ids: torch.Tensor
    positions: torch.Tensor
    temperatures: torch.Tensor
    state_slots: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_ids,
            self.positions,
            self.temperatures,
            self.state_slots,
            self.slot_mapping,
            self.context_lens,
            self.block_tables,
        )

    def addresses(self) -> tuple[int, ...]:
        return tuple(tensor.data_ptr() for tensor in self.tensors())


class DecodeMetadataRuntime:
    """Persistent mapped-host staging and fixed-address CUDA output buffers."""

    def __init__(self, config):
        self.max_num_seqs = min(config.max_num_seqs, 512)
        self.block_size = config.kvcache_block_size
        self.max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
        self.sp_size = config.attention_sp
        self.capacity = decode_payload_capacity(
            self.max_num_seqs,
            self.max_num_blocks,
        )
        self.host_slab = torch.empty(self.capacity, dtype=torch.uint8, device="cpu")
        self._mapped_device_pointer = register_mapped(self.host_slab)
        self._registered = True
        self.views = DecodeMetadataViews(
            input_ids=torch.zeros(self.max_num_seqs, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(self.max_num_seqs, dtype=torch.int64, device="cuda"),
            temperatures=torch.ones(
                self.max_num_seqs, dtype=torch.float32, device="cuda"
            ),
            state_slots=torch.full(
                (self.max_num_seqs,),
                self.max_num_seqs,
                dtype=torch.int64,
                device="cuda",
            ),
            slot_mapping=torch.full(
                (self.max_num_seqs,), -1, dtype=torch.int32, device="cuda"
            ),
            context_lens=torch.ones(
                self.sp_size,
                self.max_num_seqs,
                dtype=torch.int32,
                device="cuda",
            ),
            block_tables=torch.zeros(
                self.sp_size,
                self.max_num_seqs,
                self.max_num_blocks,
                dtype=torch.int32,
                device="cuda",
            ),
        )

    def stage(self, data: bytes):
        control = decode_flat_control(data)
        if control.payload_bytes > self.capacity:
            raise ValueError(
                f"flat decode payload {control.payload_bytes} exceeds mapped slab "
                f"capacity {self.capacity}"
            )
        if (
            control.max_num_seqs != self.max_num_seqs
            or control.max_num_blocks != self.max_num_blocks
            or control.block_size != self.block_size
        ):
            raise ValueError(
                "flat decode configuration mismatch: "
                f"wire=({control.max_num_seqs}, {control.max_num_blocks}, "
                f"{control.block_size}), worker=({self.max_num_seqs}, "
                f"{self.max_num_blocks}, {self.block_size})"
            )
        ctypes.memmove(self.host_slab.data_ptr(), data, control.payload_bytes)
        return control

    def launch(self, dummy_state_slot: int) -> None:
        launch(
            self._mapped_device_pointer,
            dummy_state_slot,
            self.views.tensors(),
        )

    def launch_graph(
        self,
        dummy_state_slot: int,
        plan_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        launch_graph(
            self._mapped_device_pointer,
            dummy_state_slot,
            self.views.tensors(),
            plan_buffers,
        )

    def close(self) -> None:
        if self._registered:
            unregister_mapped(self.host_slab)
            self._registered = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DecodeMetadataRuntime",
    "DecodeMetadataViews",
    "decode_metadata_enabled",
    "decode_payload_capacity",
]
