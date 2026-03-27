# NanoDeployVL

Vision-Language encoder engine for NanoInfra. Implements an **EP-separated** (Encoder-Prefill separated) architecture where the vision encoder runs as a standalone service, producing embeddings that are transferred to LLM prefill engines via RDMA.

## Architecture

```
Client (HTTP)
  │
NanoRoute ──ZMQ(Action=5)──► EncoderEngine
  │                              │
  │                         encode images
  │                              │
  │         ◄──ZMQ(Action=6)─── VisionSlots + input_ids
  │
  ├──ZMQ──► Prefill Engine (reads embeddings via RDMA)
  └──ZMQ──► Decode Engine
```

1. NanoRoute receives a multimodal request and forwards image/video data to the encoder engine
2. **EncoderEngine** runs the ViT, writes embeddings to a GPU **EmbeddingPool**, and returns slot metadata
3. The prefill engine reads embeddings from the pool via RDMA (zero-copy) and runs the LLM forward pass
4. After prefill, the encoder reclaims the slots via a P2P free notification

## Components

| Module                       | Purpose                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `vision/encoder.py`          | Qwen3-VL ViT implementation (patch embed, rotary, attention, patch merge)    |
| `vision/processor.py`        | Image/video preprocessing and tokenization (wraps HF `Qwen3VLProcessor`)     |
| `encoder/encoder_engine.py`  | Core engine: EmbeddingPool management, RDMA registration, ZMQ encode service |
| `server/vl_engine_server.py` | FastAPI server wrapping EncoderEngine with health-check endpoint             |
| `config.py`                  | `VLConfig` — extends NanoDeploy `Config` with vision-specific parameters     |

## Configuration

VLConfig inherits all NanoDeploy engine parameters and adds:

| Parameter           | Type | Default      | Description                            |
| ------------------- | ---- | ------------ | -------------------------------------- |
| `vision_device`     | str  | `"cuda:0"`   | Device for the vision encoder          |
| `vision_dtype`      | str  | `"bfloat16"` | Torch dtype for vision encoder weights |
| `vision_batch_size` | int  | `8`          | Max images encoded in one forward pass |

Vision model config and special token IDs (`image_token_id`, `video_token_id`, etc.) are auto-extracted from the HuggingFace model config.

## Installation

```bash
pip install -e NanoDeployVL/
# or via the root meta-package
pip install ".[nanodeployvl]"
```

Requires: Python 3.10+, `nanodeploy`, `torch`, `transformers>=4.52.0`, `Pillow`, `safetensors`.

## Usage

### Start the encoder server

```bash
nanodeployvl-serve \
    --model /models/qwen3-vl \
    --vision_device cuda:0 \
    --vision_dtype bfloat16 \
    --host 0.0.0.0 --port 5002 \
    --nanoctrl_address 10.102.97.1:3000
```

The server registers with NanoCtrl as `role="encoder"` so NanoRoute can discover it dynamically.

### Examples

- `examples/test_encoder_engine.py` — standalone encoder test (no NanoCtrl/RDMA required)
- `examples/test_vl.py` — full multimodal request through NanoRoute

## Integration

- **NanoRoute**: Discovers encoder engines via NanoCtrl; forwards `EncodeRequest` (Action=5) and receives `EncodeResponse` (Action=6)
- **NanoDeploy**: Prefill engine reads vision embeddings from the EmbeddingPool via RDMA using `VisionSlot` metadata
- **DLSlime**: PeerAgent provides RDMA memory region registration for zero-copy embedding transfer
- **NanoCtrl**: Engine lifecycle (register, heartbeat, discovery)
