"""Multimodal (Vision-Language) inference example using NanoDeployVL.

This script demonstrates end-to-end VL inference using:
- VLEngine (vision encoder + LLM engine)
- Qwen3.5-35B-A3B or Qwen3.5-397B-A17B-FP8 model

Usage
-----
# Single GPU (35B model, BF16)
python encoder_llm.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --attn_world_size 1 \\
    --image_path /path/to/image.jpg \\
    --prompt "Describe what you see in this image."

# Multi-GPU (397B FP8 model, 8 GPUs)
python encoder_llm.py \\
    --model /models/models--Qwen--Qwen3.5-397B-A17B-FP8 \\
    --attn_world_size 8 \\
    --image_path /path/to/image.jpg \\
    --prompt "What is in this image?"

# Text-only (no image, for sanity check)
python encoder_llm.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --attn_world_size 1 \\
    --prompt "What is 1+1?"

# With a URL image
python encoder_llm.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --attn_world_size 1 \\
    --image_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg" \\
    --prompt "Describe this image."
"""

import argparse
import sys
import time

import torch
from PIL import Image


def load_image(
    image_path: str | None = None, image_url: str | None = None
) -> Image.Image | None:
    """Load an image from a local path or URL."""
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        import io
        import urllib.request

        with urllib.request.urlopen(image_url, timeout=30) as resp:  # noqa: S310
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="NanoDeployVL: Vision-Language inference example"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/models/models-Qwen-Qwen3.5-35B-A3B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--image_path", type=str, default=None, help="Path to a local image file"
    )
    parser.add_argument(
        "--image_url", type=str, default=None, help="URL of an image to download"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe what you see in this image.",
        help="Text prompt",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Sampling temperature"
    )
    # NanoDeploy config args
    parser.add_argument(
        "--attn_world_size", type=int, default=1, help="Number of attention workers"
    )
    parser.add_argument(
        "--attention_dp", type=int, default=1, help="Attention data parallelism"
    )
    parser.add_argument(
        "--attention_sp", type=int, default=1, help="Attention sequence parallelism"
    )
    parser.add_argument(
        "--max_num_seqs", type=int, default=8, help="Max number of sequences"
    )
    parser.add_argument(
        "--kvcache_block_size", type=int, default=64, help="KV cache block size"
    )
    parser.add_argument(
        "--enforce_eager", action="store_true", help="Disable CUDA graph"
    )
    parser.add_argument(
        "--vision_device",
        type=str,
        default="cuda:0",
        help="Device for vision encoder",
    )
    parser.add_argument(
        "--vision_dtype",
        type=str,
        default="bfloat16",
        help="Dtype for vision encoder (bfloat16 or float16)",
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Load image (if any)
    # ----------------------------------------------------------------
    image = load_image(args.image_path, args.image_url)
    has_image = image is not None
    if has_image:
        print(f"Loaded image: {image.size}")
    else:
        print("No image provided, running text-only mode.")

    # ----------------------------------------------------------------
    # 2. Build VLConfig
    # ----------------------------------------------------------------
    # Add NanoDeployVL to path if not installed
    import os

    nanodeployvl_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "NanoDeployVL",
    )
    if os.path.isdir(nanodeployvl_root):
        sys.path.insert(0, nanodeployvl_root)

    from nanodeploy.sampling_params import SamplingParams
    from nanodeployvl.config import VLConfig
    from nanodeployvl.engine.vl_engine import VLEngine

    config = VLConfig(
        model=args.model,
        attn_world_size=args.attn_world_size,
        attention_dp=args.attention_dp,
        attention_sp=args.attention_sp,
        max_num_seqs=args.max_num_seqs,
        kvcache_block_size=args.kvcache_block_size,
        enforce_eager=False,  # Default to eager for VL
        vision_device=args.vision_device,
        vision_dtype=args.vision_dtype,
        ray_address="10.102.97.179:7078",
        master_address="10.102.97.179:6006",
    )

    print(f"Model: {args.model}")
    print(f"Vision config: {config.vision_config is not None}")
    print(f"Image token ID: {config.image_token_id}")

    # ----------------------------------------------------------------
    # 3. Build VLEngine
    # ----------------------------------------------------------------
    print("Initializing VLEngine...")
    t0 = time.time()
    engine = VLEngine(config)
    print(f"VLEngine initialized in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------
    # 4. Build messages in OpenAI format
    # ----------------------------------------------------------------
    if has_image:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        images = [image]
    else:
        messages = [
            {
                "role": "user",
                "content": args.prompt,
            }
        ]
        images = None

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ignore_eos=False,
    )

    # ----------------------------------------------------------------
    # 5. Generate
    # ----------------------------------------------------------------
    print("\n--- Generation ---")
    print(f"Prompt: {args.prompt}")
    t0 = time.time()

    result = engine.generate_vl(
        messages=messages,
        images=images,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    elapsed = time.time() - t0
    print(f"\nCompletion ({elapsed:.2f}s):")
    print(result)
    print("--- Done ---")

    # ----------------------------------------------------------------
    # 6. Cleanup
    # ----------------------------------------------------------------
    engine.exit()


if __name__ == "__main__":
    main()
