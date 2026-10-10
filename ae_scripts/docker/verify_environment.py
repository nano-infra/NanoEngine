"""Check package versions and AE native APIs without initializing a GPU."""

import importlib
from importlib import metadata
import json
from pathlib import Path
import re


EXPECTED = {
    "vllm": "0.18.0",
    "torch": "2.10.0+cu129",
    "torchvision": "0.25.0+cu129",
    "torchaudio": "2.10.0+cu129",
    "triton": "3.6.0",
    "ray": "2.54.0",
    "ijson": "3.5.0",
    "dlslime": "0.0.1.post10",
    "nanodeploy": "0.2.0",
    "deep_ep": "1.2.1+73b6ea4",
    "deep_gemm": "2.3.0+477618c",
    "flash_mla": "1.0.0+1408756",
    "flash-attn-3": "3.0.0+20260316.cu129torch2100cxx11abitrue.71bf77",
    "flashinfer-python": "0.6.6",
    "nvidia-nccl-cu12": "2.27.5",
}


def verify_vllm_installation():
    distribution = metadata.distribution("vllm")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("vLLM must be installed as a normal wheel")
    if "archive_info" not in direct_url:
        raise RuntimeError("Expected vLLM to be installed from the built wheel")
    commit = Path("/opt/ae/vllm-source-commit.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Missing vLLM source revision")
    manifest = json.loads(Path("/opt/ae/vllm-precompiled.json").read_text())
    for name in manifest["files"]:
        if not Path(distribution.locate_file(name)).is_file():
            raise RuntimeError(f"Base vLLM precompiled file is missing: {name}")
    envs = Path(distribution.locate_file("vllm/envs.py")).read_text()
    if "VLLM_MOE_GEMM_DEBUG" not in envs:
        raise RuntimeError("The modified vLLM Python code was not installed")
    return {"source_commit": commit, "editable": False,
            "base_precompiled_files": len(manifest["files"])}


def main():
    observed = {name: metadata.version(name) for name in EXPECTED}
    mismatches = {name: [EXPECTED[name], version] for name, version in observed.items()
                  if version != EXPECTED[name]}
    if mismatches:
        raise RuntimeError(f"Package version mismatch: {mismatches}")
    vllm_installation = verify_vllm_installation()

    # Load Torch before extensions that link to its shared libraries.
    import torch
    import dlslime
    import deep_ep
    import deep_gemm
    import flash_attn_interface
    import flash_mla

    if torch.version.cuda != "12.9" or not torch.compiled_with_cxx11_abi():
        raise RuntimeError("Expected CUDA 12.9 Torch with the C++11 ABI")
    for module in ["nanodeploy._nanodeploy_cpp", "flash_mla.cuda",
                   "flash_attn_3._C", "flashinfer"]:
        importlib.import_module(module)
    for owner, symbol in [(dlslime, "AllToAllBuffer"), (dlslime, "KernelImpl"),
                          (deep_ep, "Buffer"), (deep_gemm, "fp8_gemm_nt"),
                          (flash_attn_interface, "flash_attn_varlen_func"),
                          (flash_mla, "flash_mla_with_kvcache")]:
        if not hasattr(owner, symbol):
            raise RuntimeError(f"Required symbol is missing: {owner.__name__}.{symbol}")
    if not hasattr(dlslime.KernelImpl, "Basic"):
        raise RuntimeError("DLSlime lacks the Basic intra-node kernel")
    if deep_ep.topk_idx_t is not torch.int64:
        raise RuntimeError("Expected the DeepEP build with int64 top-k indices")
    print(json.dumps({"versions": observed, "vllm_installation": vllm_installation,
                      "native_imports": "passed",
                      "gpu_execution": "not tested"}, indent=2))


if __name__ == "__main__":
    main()
