# NanoOps

Operations CLI for NanoInfra distributed LLM inference.

## Installation

```bash
cd NanoOps
pip install -e .
```

## Quick Start

```bash
# Start a session
nanoctrl start --session-id my-session

# Set model configuration
nanoctrl set --session-id my-session \
  --model /models/llama2-7b \
  --prefill-attention-dp 8 --prefill-ffn-ep 8 \
  --decode-attention-tp 8 --decode-ffn-tp 8

# Spawn components
nanoctrl spawn prefill --session-id my-session
nanoctrl spawn decode --session-id my-session
nanoctrl spawn route --session-id my-session

# Wait for ready
nanoctrl wait --session-id my-session

# System is now ready!
```

## Commands

- `nanoctrl start` - Initialize a new session
- `nanoctrl set` - Set model configuration
- `nanoctrl spawn` - Spawn a component (route, prefill, decode)
- `nanoctrl wait` - Wait for all components to be ready
- `nanoctrl stop` - Stop session and cleanup
- `nanoctrl list` - List active sessions
