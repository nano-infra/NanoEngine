"""Export installed vLLM binaries for its supported precompiled build path.

The archive is consumed by setup.py, not installed by pip. Modified Python
code comes from the pinned source; CUDA libraries come from the base image.
"""

from importlib import metadata
import json
from pathlib import Path
import re
import sys
import zipfile


EXTENSIONS = {
    "vllm/_C.abi3.so",
    "vllm/_moe_C.abi3.so",
    "vllm/_flashmla_C.abi3.so",
    "vllm/_flashmla_extension_C.abi3.so",
    "vllm/_sparse_flashmla_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so",
    "vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so",
    "vllm/cumem_allocator.abi3.so",
    "vllm/_rocm_C.abi3.so",
}
GENERATED_INTERFACES = re.compile(
    r"vllm/(?:vllm_flash_attn|third_party/triton_kernels|third_party/flashmla)"
    r"/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py$"
)


def export(package_root, archive_path, manifest_path, base_version):
    package_root = Path(package_root)
    files = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        relative = "vllm/" + path.relative_to(package_root).as_posix()
        if relative in EXTENSIONS or GENERATED_INTERFACES.fullmatch(relative):
            files.append((relative, path))
    if "vllm/_C.abi3.so" not in {name for name, _ in files}:
        raise RuntimeError("The base image has no vLLM CUDA extension")

    manifest = {"base_vllm_version": base_version, "files": []}
    # Store the bytes directly; the temporary archive is deleted after install.
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for relative, path in files:
            archive.write(path, relative)
            manifest["files"].append(relative)
    Path(manifest_path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    distribution = metadata.distribution("vllm")
    if distribution.version != "0.18.0":
        raise RuntimeError(f"Expected base vLLM 0.18.0, got {distribution.version}")
    manifest = export(distribution.locate_file("vllm"), sys.argv[1], sys.argv[2],
                      distribution.version)
    print(f"Exported {len(manifest['files'])} base vLLM precompiled files")


if __name__ == "__main__":
    main()
