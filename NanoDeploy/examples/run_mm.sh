#!/bin/bash
# Test EP-separated EncoderEngine (standalone, no NanoCtrl)

python test_encoder_engine.py \
    --model /models/models-Qwen-Qwen3.5-35B-A3B \
    --synthetic \
    --device cuda:0 \
    --dtype bfloat16 \
    --num_slots 8 \
    --max_tokens_per_slot 4096 \
    2>&1 | tee test_encoder.log
