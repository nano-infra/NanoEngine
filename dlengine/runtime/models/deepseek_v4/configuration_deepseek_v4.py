import torch
from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        dtype = kwargs.get("dtype", kwargs.get("torch_dtype", None))
        if isinstance(dtype, str):
            dtype = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }.get(dtype, None)
        if "max_position_embeddings" not in kwargs and "max_seq_len" in kwargs:
            kwargs["max_position_embeddings"] = kwargs["max_seq_len"]
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 16384)
        super().__init__(**kwargs)
        if rope_scaling is not None:
            self.rope_scaling = rope_scaling
        for key, value in kwargs.items():
            setattr(self, key, value)
        if dtype is not None:
            self.dtype = dtype
            self.torch_dtype = dtype
