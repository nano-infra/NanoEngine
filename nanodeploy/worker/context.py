from dataclasses import dataclass

import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_dummy: bool = False

    def print(self):
        """打印所有张量字段的形状，非张量字段打印值或类型"""
        print("Context 字段信息：")
        # 遍历所有字段
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, torch.Tensor):
                # 张量字段：打印名称和形状
                print(
                    f"  {field}: shape={value.shape}, dtype={value.dtype}, device={value.device}"
                )
            else:
                # 非张量字段：打印名称和值（或None）
                print(f"  {field}: {value}")


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    is_dummy=False,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        is_dummy=is_dummy,
    )


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
