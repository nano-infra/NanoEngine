python encoder_llm.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --attn_world_size 1 \
    --image_path /tmp/test_image.jpg \
    --prompt "描述这张图片。" \
    --max_tokens 256 2>&1 | tee run_mm.log
