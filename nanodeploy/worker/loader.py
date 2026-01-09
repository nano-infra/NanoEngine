import os
import re
from glob import glob

import torch
from safetensors import safe_open
from torch import nn
from tqdm import tqdm


def default_weight_loader(param, tensor, **kwargs):
    """默认权重加载器（保持原逻辑）"""
    param.data.copy_(tensor)


def parse_shard_info(filename):
    """从文件名解析分片信息 model-00001-of-00004.safetensors"""
    match = re.search(r"-(\d{5})-of-(\d{5})", filename)
    if match:
        return int(match.group(1)) - 1, int(match.group(2))
    return None, None


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    # 获取所有权重文件并初始化进度条（按文件数量）
    weight_files = glob(os.path.join(path, "*.safetensors"))
    pbar = tqdm(weight_files, desc="Loading weights", unit="files")

    try:
        for file in pbar:  # 直接迭代文件列表，进度条按文件计数
            ckpt_shard_id, ckpt_num_shards = parse_shard_info(file)
            
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
                                param, 
                                f.get_tensor(weight_name), 
                                loaded_shard_id=shard_id, 
                                weight_name=weight_name,
                                ckpt_shard_id=ckpt_shard_id,
                                ckpt_num_shards=ckpt_num_shards
                            )
                            matched = True
                            break

                    # 如果没有匹配的映射，使用默认加载逻辑
                    if not matched:
                        param = model.get_parameter(weight_name)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param, 
                            f.get_tensor(weight_name), 
                            ckpt_shard_id=ckpt_shard_id, 
                            ckpt_num_shards=ckpt_num_shards
                        )
    finally:
        pbar.close()
