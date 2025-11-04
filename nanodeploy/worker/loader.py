import os
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm


def default_weight_loader(param, tensor):
    """默认权重加载器（保持原逻辑）"""
    param.data.copy_(tensor)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 获取所有权重文件并初始化进度条（按文件数量）
    weight_files = glob(os.path.join(path, "*.safetensors"))
    pbar = tqdm(weight_files, desc="Loading weights", unit="file")

    try:
        for file in pbar:  # 直接迭代文件列表，进度条按文件计数
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    matched = False
                    for k in packed_modules_mapping:
                        if k in weight_name:
                            v, shard_id = packed_modules_mapping[k]
                            param_name = weight_name.replace(k, v)
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(
                                param, f.get_tensor(weight_name), shard_id, weight_name
                            )
                            matched = True
                            break

                    # 如果没有匹配的映射，使用默认加载逻辑
                    if not matched:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, f.get_tensor(weight_name))
    finally:
        pbar.close()
