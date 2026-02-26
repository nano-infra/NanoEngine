"""
1. 引入 DeepEP 的 buffer (参考 DLBLAS, 和 nano-deploy-refactor 的 CPP 的部分)
2. 引入 DeepGEMM 来实现 FP8 和 BF 16 的 Group Gemm, 以及 FP8 和 BF 16 的 linear
3. forward 执行
    3.1 DeepEP 的 dispatch combine
    3.2 Prefill 和 Decode 相关的 permute, 量化解量化操作。
4. 额外的要求
    4.1 同时支持 TP, EP 和单机执行
    4.2 同时支持 FP8 和 BF16 的执行
"""

import torch


class DeepSeekSparseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer = ...

        # weights 也转移到这里, 名字和之前的一致，这样方便多种数据类型的支持
        ...

        # 分布式的 configuration 要定义清晰
        ...

    def forward(self):
        pass
