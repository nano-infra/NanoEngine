"""Full EP-separated VL inference integration test.

Tests the complete pipeline:
  EncoderEngine (encode → EmbeddingPool → RDMA MR)
    ↓ VisionSlotMeta
  LLM Engine (Sequence.add_vision_slot → serialize → ModelRunner
        → extract_vision_slots_from_bytes → _fetch_vision_embeds_rdma
        → _inject_vision_embeds → prefill forward → decode → output)
    ↓ P2P FreeVisionSlots (Action=4)
  EncoderEngine (free_slots → pool reclaim)

No prefill/decode disaggregation — single LLM engine does both phases.
The "EP separation" is between the vision Encoder and the LLM engine.

Prerequisites:
  - Redis running on 127.0.0.1:6379
  - NanoCtrl running on 127.0.0.1:4479
  - Ray cluster with at least 1 GPU
  - dlslime installed (RDMA support)
  - Model checkpoint with vision_config (e.g. Qwen3.5-35B-A3B)

Usage
-----
# With real image (1 GPU, eager mode for MoE)
python test_ep_full.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --image_path /tmp/test_image.jpg \\
    --enforce_eager

# With synthetic image
python test_ep_full.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --enforce_eager

# Customize
python test_ep_full.py \\
    --model /models/models-Qwen-Qwen3.5-35B-A3B \\
    --ctrl_address http://127.0.0.1:4479 \\
    --encoder_device cuda:0 \\
    --max_tokens 64 \\
    --enforce_eager
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# VL support now ships inside dlengine as the ``dlengine.vl`` subpackage,
# so no sys.path manipulation is required.


def load_or_make_image(path: str | None, synthetic: bool = True):
    from PIL import Image

    if path:
        return Image.open(path).convert("RGB")
    if synthetic:
        import numpy as np

        arr = np.random.randint(0, 256, (672, 672, 3), dtype=np.uint8)
        return Image.fromarray(arr)
    return None


# ======================================================================
# Step helpers
# ======================================================================


def step_encoder(args) -> tuple:
    """Start EncoderEngine with NanoCtrl + RDMA MR, encode an image."""
    import torch
    from dlengine.vl.encoder.encoder_config import EncoderConfig
    from dlengine.vl.encoder.encoder_engine import EncoderEngine
    from dlengine.vl.vision.processor import ImageProcessor

    image = load_or_make_image(args.image_path)
    print(f"[Encoder] Image: {image.size}")

    config = EncoderConfig(
        model=args.model,
        vision_device=args.encoder_device,
        vision_dtype="bfloat16",
        num_slots=8,
        max_tokens_per_slot=4096,
        ctrl_address=args.ctrl_address,
        ctrl_scope=args.ctrl_scope,
        host=args.host,
        p2p_port=0,
    )
    print(
        f"[Encoder] Config: hidden_size={config.hidden_size}, "
        f"num_slots={config.num_slots}"
    )

    t0 = time.time()
    engine = EncoderEngine(config)
    print(f"[Encoder] Engine ready in {time.time() - t0:.1f}s  id={engine.engine_id}")

    # Preprocess
    processor = ImageProcessor(args.model)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]
    prompt_text = processor.apply_chat_template(messages)
    outputs = processor.process(text=prompt_text, images=[image])
    pixel_values = outputs["pixel_values"]
    image_grid_thw = outputs["image_grid_thw"]
    input_ids = outputs["input_ids"].squeeze(0)
    print(
        f"[Encoder] Preprocessed: pixel_values={pixel_values.shape}, "
        f"grid_thw={image_grid_thw.tolist()}, input_ids={input_ids.shape}"
    )

    # Encode
    t0 = time.time()
    slot_metas = engine.encode(pixel_values, image_grid_thw)
    print(f"[Encoder] Encoded {len(slot_metas)} image(s) in {time.time() - t0:.3f}s")
    for m in slot_metas:
        print(
            f"  slot={m.slot_idx} tokens={m.num_tokens} hidden={m.hidden_size} "
            f"max_per_slot={m.max_tokens_per_slot}"
        )
    print(f"[Encoder] Pool free={engine.pool.available_slots}/{config.num_slots}")

    return engine, slot_metas, input_ids.tolist(), processor


def step_llm(args, slot_metas, token_ids) -> list:
    """Run LLM engine: prefill (with RDMA fetch) + decode in single engine."""
    import ray
    from dlengine.config import Config
    from dlengine.engine.sequence import Sequence
    from dlengine.llm_component import LLMComponent
    from dlengine.sampling_params import SamplingParams

    ray.init(address=args.ray_address, ignore_reinit_error=True)

    config = Config(
        model=args.model,
        ctrl_address=args.ctrl_address,
        ctrl_scope=args.ctrl_scope,
        ray_address=args.ray_address,
        master_address=args.master_address,
        kvcache_block_size=64,
        max_num_seqs=8,
        attn_world_size=args.attn_world_size,
        attention_dp=1,
        attention_sp=1,
        enforce_eager=args.enforce_eager,
    )

    print("[LLM] Starting LLMComponent engine…")
    t0 = time.time()
    llm = LLMComponent.as_remote(config)
    print(f"[LLM] Engine ready in {time.time() - t0:.1f}s")

    # Build sequence with vision slots attached
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.1,
        ignore_eos=False,
    )
    seq = Sequence(token_ids, sampling_params=sampling_params)

    # Attach vision slot metadata from encoder
    for m in slot_metas:
        seq.add_vision_slot(
            encoder_engine_id=m.encoder_engine_id,
            slot_idx=m.slot_idx,
            num_tokens=m.num_tokens,
            hidden_size=m.hidden_size,
            max_tokens_per_slot=m.max_tokens_per_slot,
        )
    print(
        f"[LLM] Sequence: {len(token_ids)} tokens, " f"{len(slot_metas)} vision slot(s)"
    )

    # Run prefill + decode
    ray.get(llm.add_request.remote([seq]))
    print("[LLM] Generating (prefill + decode)…")
    t0 = time.time()
    finished = ray.get(llm.generate.remote())
    elapsed = time.time() - t0
    print(f"[LLM] Done in {elapsed:.1f}s, {len(finished)} sequence(s)")

    # Free vision slots on encoder via P2P (engine_server does this automatically,
    # but generate() path doesn't, so do it manually in the test)
    from collections import defaultdict

    vision_free_by_encoder: dict[str, list[int]] = defaultdict(list)
    for s in finished:
        vs_list = s.vision_slots
        if not vs_list:
            continue
        for vs in vs_list:
            vision_free_by_encoder[vs["encoder_engine_id"]].append(vs["slot_idx"])
        s.clear_vision_slots()

    for encoder_id, slot_indices in vision_free_by_encoder.items():
        print(f"[LLM] Sending P2P free for encoder={encoder_id}, slots={slot_indices}")
        ray.get(llm.send_free_vision_slots.remote(encoder_id, slot_indices))

    return finished


# ======================================================================
# Main
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="Full EP VL integration test")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Describe this image in detail.")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--ctrl_address", type=str, default="http://127.0.0.1:4479")
    parser.add_argument("--ctrl_scope", type=str, default=None)
    parser.add_argument("--ray_address", type=str, default="auto")
    parser.add_argument("--master_address", type=str, default="10.102.97.179:6006")
    parser.add_argument("--host", type=str, default="10.102.97.179")
    parser.add_argument("--encoder_device", type=str, default="cuda:0")
    parser.add_argument("--attn_world_size", type=int, default=1)
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graph capture (required for MoE models)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Full EP-separated VL Integration Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Encoder — encode image, write to pool, register RDMA MR
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 1: EncoderEngine")
    print("=" * 60)
    encoder_engine, slot_metas, token_ids, processor = step_encoder(args)

    # ------------------------------------------------------------------
    # Phase 2: LLM — prefill (RDMA fetch + inject) + decode in one engine
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 2: LLM Engine (RDMA fetch + prefill + decode)")
    print("=" * 60)
    finished = step_llm(args, slot_metas, token_ids)

    # Check that encoder pool was freed via P2P
    time.sleep(2.0)  # Give P2P message time to arrive
    print(
        f"\n[Encoder] Pool after generation: "
        f"free={encoder_engine.pool.available_slots}/{encoder_engine.config.num_slots}"
    )
    if encoder_engine.pool.available_slots == encoder_engine.config.num_slots:
        print("[Encoder] All slots freed via P2P ✓")
    else:
        print("[Encoder] WARNING: some slots not yet freed (P2P may be delayed)")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    for seq in finished:
        comp_ids = seq.completion_token_ids
        text = processor.decode(comp_ids)
        print(f"Seq {seq.seq_id}: {len(comp_ids)} generated tokens")
        print(f"Output: {text}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    print("\n[Cleanup] Shutting down encoder…")
    encoder_engine.shutdown()
    print("✓ Full EP integration test complete!")


if __name__ == "__main__":
    main()
