"""Test the EP-separated EncoderEngine (standalone, no NanoCtrl / RDMA).

Validates:
1. EncoderConfig loads vision_config from HF model checkpoint.
2. EncoderEngine initialises VisionEncoder + EmbeddingPool.
3. ImageProcessor preprocesses an image into pixel_values + image_grid_thw.
4. engine.encode() writes embeddings into pool slots and returns VisionSlotMeta.
5. Slot management: allocate → read back → free → re-allocate.

Usage
-----
# With a local image
python test_encoder_engine.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --image_path /path/to/image.jpg

# With a URL image
python test_encoder_engine.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --image_url "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"

# With a synthetic random image (no real image needed)
python test_encoder_engine.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --synthetic

# Specify device / dtype
python test_encoder_engine.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --synthetic \
    --device cuda:0 \
    --dtype bfloat16
"""

import argparse
import os
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Resolve NanoDeployVL path
# ---------------------------------------------------------------------------
_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..",
    "NanoDeployVL",
)
if os.path.isdir(_root):
    sys.path.insert(0, _root)


def load_image(path: str | None, url: str | None):
    from PIL import Image

    if path:
        return Image.open(path).convert("RGB")
    if url:
        import io
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    return None


def make_synthetic_image(width: int = 672, height: int = 672):
    """Create a random RGB image for testing without a real file."""
    import numpy as np
    from PIL import Image

    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser(description="Test the EP-separated EncoderEngine")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the HF model directory",
    )
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_url", type=str, default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="Use a synthetic random image"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_slots", type=int, default=8)
    parser.add_argument("--max_tokens_per_slot", type=int, default=4096)
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1. Prepare image
    # ----------------------------------------------------------------
    image = load_image(args.image_path, args.image_url)
    if image is None and args.synthetic:
        image = make_synthetic_image()
    if image is None:
        print("ERROR: provide --image_path, --image_url, or --synthetic")
        sys.exit(1)
    print(f"[1/5] Image ready: size={image.size}")

    # ----------------------------------------------------------------
    # 2. Build EncoderConfig (no NanoCtrl)
    # ----------------------------------------------------------------
    from nanodeployvl.encoder.encoder_config import EncoderConfig

    config = EncoderConfig(
        model=args.model,
        vision_device=args.device,
        vision_dtype=args.dtype,
        num_slots=args.num_slots,
        max_tokens_per_slot=args.max_tokens_per_slot,
        nanoctrl_address=None,  # standalone, no NanoCtrl
    )
    print(
        f"[2/5] EncoderConfig: hidden_size={config.hidden_size}, "
        f"vision_config={type(config.vision_config).__name__}, "
        f"num_slots={config.num_slots}, "
        f"max_tokens_per_slot={config.max_tokens_per_slot}"
    )

    # ----------------------------------------------------------------
    # 3. Create EncoderEngine
    # ----------------------------------------------------------------
    t0 = time.time()
    from nanodeployvl.encoder.encoder_engine import EncoderEngine

    engine = EncoderEngine(config)
    print(f"[3/5] EncoderEngine created in {time.time() - t0:.1f}s")
    print(
        f"       engine_id={engine.engine_id}, "
        f"pool free={engine.pool.available_slots}/{config.num_slots}"
    )

    # ----------------------------------------------------------------
    # 4. Preprocess image via ImageProcessor
    # ----------------------------------------------------------------
    from nanodeployvl.vision.processor import ImageProcessor

    processor = ImageProcessor(args.model)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages)
    outputs = processor.process(text=prompt_text, images=[image])

    pixel_values = outputs["pixel_values"]
    image_grid_thw = outputs["image_grid_thw"]
    input_ids = outputs["input_ids"]
    print(
        f"[4/5] Preprocessed: pixel_values={pixel_values.shape}, "
        f"image_grid_thw={image_grid_thw.tolist()}, "
        f"input_ids length={input_ids.shape[-1]}"
    )

    # ----------------------------------------------------------------
    # 5. Encode and verify
    # ----------------------------------------------------------------
    print("\n--- Encode Test ---")
    t0 = time.time()
    slot_metas = engine.encode(pixel_values, image_grid_thw)
    encode_time = time.time() - t0

    assert len(slot_metas) == 1, f"Expected 1 slot, got {len(slot_metas)}"
    meta = slot_metas[0]
    print(f"  encode time: {encode_time:.3f}s")
    print(f"  slot_idx:    {meta.slot_idx}")
    print(f"  num_tokens:  {meta.num_tokens}")
    print(f"  hidden_size: {meta.hidden_size}")
    print(f"  pool status: free={engine.pool.available_slots}/{config.num_slots}")

    # Verify the buffer has non-zero data
    slot_data = engine.pool.buffer[meta.slot_idx, : meta.num_tokens, :]
    assert slot_data.abs().sum().item() > 0, "Slot buffer is all zeros!"
    print(f"  buffer norm:  {slot_data.norm().item():.4f} (non-zero ✓)")

    # ----------------------------------------------------------------
    # 6. Free slot and re-allocate
    # ----------------------------------------------------------------
    print("\n--- Slot Management Test ---")
    free_before = engine.pool.available_slots
    engine.free_slots([meta.slot_idx])
    free_after = engine.pool.available_slots
    print(
        f"  free_slots([{meta.slot_idx}]): " f"available {free_before} → {free_after}"
    )
    assert free_after == free_before + 1, "Free did not increment!"

    # Re-allocate the same image
    slot_metas_2 = engine.encode(pixel_values, image_grid_thw)
    meta2 = slot_metas_2[0]
    print(
        f"  re-encoded → slot_idx={meta2.slot_idx}, "
        f"num_tokens={meta2.num_tokens}, "
        f"pool free={engine.pool.available_slots}/{config.num_slots}"
    )

    # ----------------------------------------------------------------
    # 7. Multi-image encode (encode same image 3 times)
    # ----------------------------------------------------------------
    print("\n--- Multi-Image Encode Test ---")
    # Free previous slot first
    engine.free_slots([meta2.slot_idx])

    # Stack pixel_values 3 times
    pv3 = pixel_values.repeat(3, 1)
    thw3 = image_grid_thw.repeat(3, 1)
    t0 = time.time()
    multi_metas = engine.encode(pv3, thw3)
    multi_time = time.time() - t0
    print(f"  3 images encoded in {multi_time:.3f}s")
    print(
        f"  slots: {[m.slot_idx for m in multi_metas]}, "
        f"tokens: {[m.num_tokens for m in multi_metas]}"
    )
    print(f"  pool free={engine.pool.available_slots}/{config.num_slots}")
    assert len(multi_metas) == 3

    # Free all
    engine.free_slots([m.slot_idx for m in multi_metas])
    assert engine.pool.available_slots == config.num_slots, "Not all slots freed!"
    print(f"  all freed: pool free={engine.pool.available_slots}/{config.num_slots} ✓")

    # ----------------------------------------------------------------
    # 8. Shutdown
    # ----------------------------------------------------------------
    engine.shutdown()
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
