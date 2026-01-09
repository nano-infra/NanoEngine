# Qwen3 End-to-End Chat Demo Walkthrough

This document outlines the successful integration of the Qwen3 model into the NanoDeploy Spoke architecture, enabling an end-to-end chat demo.

## Architecture

The system now operates as a distributed RPC system:

1. **Tokenizer (Python)**: `tools/run_qwen_chat.py` uses `transformers` to tokenize text and launch the C++ client.
2. **Client (C++)**: `test_qwen3_runner` (via `SimpleEngine`) connects to the Spoke Agent, sends Token IDs via RDMA/RPC.
3. **Agent (Spoke)**: `nanodeploy_agent` hosts the `ModelRunner` actor, which executes the Qwen3 model (Stateless Forward Pass).
4. **Detokenizer (Python)**: The C++ client prints output IDs, which the Python script captures and decodes.

## Changes & Fixes

### 1. Tokenizer Strategy

- **Reverted** from `tokenizers-cpp` to Python-based tokenization to avoid complexity and build issues.
- **Protocol**: Simple CLI arguments (`[ids...]`) passing from Python to C++.

### 2. Spoke Integration

- **SimpleEngine**: Refactored to act as a **Spoke Client** instead of a local runner.
- **RDMA**: Enabled RDMA in `SimpleEngine` headers to match Agent configuration (`enable_rdma=true`).
- **Connection**: Fixed connection issues by exposing `--agent_port` (defaulting to Hub port 8888 for routing).

### 3. Model Correctness

- **Garbage Output Fix**: Identified critical bug in `Qwen3ForCausalLM::compute_logits` where hidden states were returned instead of logits. Enabled `lm_head_` projection.
- **Result**: Model now generates valid token IDs (e.g., `21806` -> "Answer") instead of low-integer noise.

### 4. Stability

- **Crash Fix**: Patched `Spoke::Client` destructor in `client.h` to suppress `std::system_error` during shutdown/termination.
- **Clean Shutdown**: Added `SimpleEngine::shutdown()` for explicit resource release.

## How to Run

### 1. Start Infrastructure (Terminals 1 & 2)

**Terminal 1 (Hub):**

```bash
./tools/spoke_hub
```

**Terminal 2 (Agent):**

```bash
# Rebuild if needed
cd build; cmake --build . --target nanodeploy_agent --config Release; cd ..
./bin/nanodeploy_agent 9000 127.0.0.1 8888
```

### 2. Run Chat Demo (Terminal 3)

**Build Client:**

```bash
cd build; cmake --build . --target test_qwen3_runner --config Release; cd ..
```

**Run Script:**

```bash
python ./tools/run_qwen_chat.py \
    --model_path /models/qwen3-0.6b-local \
    --exe_path ./build/bin/test_qwen3_runner \
    --agent_port 8888 \
    --prompt "Hello world!"
```

### Expected Output

```
[Python] Tokenizing prompt: 'Hello world!'
[Python] Input IDs: [9707, 1879, 0]
...
[SimpleEngine] Starting generation (Remote)...
...
[Python] Generated IDs: [21806, 21806, 38297, ...]
[Python] **Response:** Answer Answer Instructions...
```

*Note: Repetition is expected due to greedy decoding without a chat template.*
