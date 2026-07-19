# Offline Inference

This guide covers direct Python inference through the scripts in [`examples/`](../examples). Offline mode is intended for development, debugging, correctness checks, and batch jobs that do not need a public HTTP API.

For production HTTP serving with `dlengine serve`, `dlengine-router`, and Claude Code, use the [Ray + DLEngine PD + Router quick start](../README.md#quick-start-ray--dlengine-pd--router).

## Offline versus serving mode

| Mode | Ray | dlslime-ctrl | dlengine-router | Public HTTP API |
| --- | --- | --- | --- | --- |
| Non-disaggregated offline | Required for GPU resource management and worker placement | Not required | Not required | No |
| PD-disaggregated offline | Required | Required for prefill/decode discovery and migration coordination | Not required | No |
| Production serving | Required | Required for discovery/control | Required | OpenAI + Anthropic |

Offline scripts create requests directly, drive the engine lifecycle, and print completions to stdout. They do not expose `/v1/chat/completions` or `/v1/messages`.

## Prerequisites

- Install DLEngine and its model dependencies.
- Make the model checkpoint available at the same path on all Ray nodes.
- Start a Ray cluster and verify its resources:

```bash
export HEAD_IP=<ray-head-ip>
export RAY_ADDRESS="${HEAD_IP}:7078"

# Head node
ray start --head \
  --node-ip-address "${HEAD_IP}" \
  --port 7078 \
  --dashboard-host 0.0.0.0

# Every worker node
ray start --address "${HEAD_IP}:7078"

ray status --address "${RAY_ADDRESS}"
```

## Non-disaggregated offline inference

[`examples/non_disagg.py`](../examples/non_disagg.py) runs prefill and decode in one logical engine. Ray still owns GPU resource allocation and worker placement, but no service registry or router is involved.

```bash
python examples/non_disagg.py \
  --model /models/Qwen3-235B-A22B \
  --ray_address "${RAY_ADDRESS}" \
  --attention_dp 8 \
  --ffn_ep 8 \
  --kvcache_block_size 256 \
  --prompt "What is 1+1?" \
  --max_tokens 128 \
  --temperature 0
```

The script:

1. Loads the tokenizer and applies its chat template when available.
2. Creates an `LLM` using the requested Ray/DP/EP configuration.
3. Encodes the prompt into a serialized `RequestIn`.
4. Calls the offline generation loop.
5. Decodes the returned token IDs and prints the completion.

Use `--ignore_eos` when a fixed number of output tokens is required. Models without a tokenizer chat template can provide a model-specific encoder through `--dsv4_encoding_dir`.

## PD-disaggregated offline inference

[`examples/disagg.py`](../examples/disagg.py) creates separate prefill and decode components but drives both from one Python process. It is useful for validating cache migration without running `dlengine-router`.

PD offline mode needs the same dlslime-ctrl control plane used by serving mode:

```bash
redis-server \
  --daemonize yes \
  --port 16379 \
  --bind 0.0.0.0 \
  --protected-mode no

export REDIS_PUBLIC_ADDRESS="${HEAD_IP}:16379"
dlslime-ctrl start --redis-url redis://127.0.0.1:16379

export CTRL_ADDRESS="http://${HEAD_IP}:4479"
```

Run the example:

```bash
python examples/disagg.py \
  --model /models/Qwen3-235B-A22B \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --executor_backend dlslime \
  --attention_dp 8 \
  --ffn_ep 8 \
  --kvcache_block_size 64 \
  --prompt "Explain prefill-decode disaggregation." \
  --max_tokens 256 \
  --temperature 0
```

The driver performs the PD lifecycle explicitly:

1. Creates independent `prefill` and `decode` `LLMComponent` actors.
2. Sends the serialized request to prefill.
3. Receives one or more `RequestMigrate` payloads.
4. Sends those payloads to decode, which retrieves the migrated KV state.
5. Generates the completion.
6. Frees the migrated sequence IDs from prefill.

There is no client-facing HTTP endpoint in this flow; dlslime-ctrl is present only for node discovery and migration/control metadata.

## Per-role overrides

Common options apply to both engines. Prefix an option with `--prefill.` or `--decode.` when the two roles need different configurations:

```bash
python examples/disagg.py \
  --model /models/Qwen3-235B-A22B \
  --ray_address "${RAY_ADDRESS}" \
  --ctrl_address "${CTRL_ADDRESS}" \
  --executor_backend dlslime \
  --prefill.attention_dp 8 \
  --prefill.ffn_ep 8 \
  --prefill.max_num_batched_tokens 16384 \
  --decode.attention_dp 8 \
  --decode.ffn_ep 8 \
  --decode.max_num_batched_tokens 2048 \
  --prompt "What is the difference between prefill and decode?" \
  --max_tokens 256
```

Both examples also accept `--config <yaml-file>` for reproducible configurations. Command-line values explicitly set under `--prefill.*` or `--decode.*` override their common defaults.

## When to use each path

- Use **non-disaggregated offline** for the shortest correctness/debug loop.
- Use **PD-disaggregated offline** to isolate prefill/decode migration and role-specific configuration.
- Use **production serving** when clients need OpenAI/Anthropic APIs, dynamic engine discovery, load balancing, streaming, or Claude Code integration.

For multi-node Redis security and containerized dlslime-ctrl deployment, see the [production quick start](../README.md#3-start-the-discovery-and-control-plane) and the official [DLSlime Docker guide](https://github.com/DeepLink-org/DLSlime/blob/main/docker/README.md).
