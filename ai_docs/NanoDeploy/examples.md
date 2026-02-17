# Example Scripts

NanoDeploy provides two example scripts for LLM inference, both using
[jsonargparse](https://github.com/omni-us/jsonargparse) with `add_class_arguments(Config)` to
expose all `Config` fields as CLI args (same pattern as `engine_server.py`).

## `non_disagg.py` — Non-Disaggregated Inference

Single-engine (hybrid prefill+decode) inference. All `Config` fields are
available as flat CLI args.

```bash
python examples/non_disagg.py \
    --model /models/deepseek-v3 \
    --attention_dp 8 --ffn_ep 8 \
    --loop_count 1 \
    --kvcache_block_size 64 \
    --max_tokens 64 --temperature 0.1 \
    --prompt "What is 1+1?"
```

Or via YAML config:

```bash
python examples/non_disagg.py --config config.yaml
```

## `disagg.py` — Disaggregated (Prefill-Decode) Inference

Two engines (prefill + decode) with automatic NanoCtrl peer discovery.
Uses scoped args for per-role config overrides.

### Shared vs Scoped Config

| Scope                  | How to set          | Examples                                 |
| ---------------------- | ------------------- | ---------------------------------------- |
| **Shared** (top-level) | `--model`, env vars | `--model /models/deepseek-v3`            |
| **Prefill-only**       | `--prefill.xxx`     | `--prefill.master_address 10.0.0.2:6006` |
| **Decode-only**        | `--decode.xxx`      | `--decode.loop_count 16`                 |

### Environment Variables

| Variable           | Description                    | Default          |
| ------------------ | ------------------------------ | ---------------- |
| `RAY_ADDRESS`      | Ray cluster address            | `127.0.0.1:6379` |
| `NANOCTRL_ADDRESS` | NanoCtrl control plane address | *(none)*         |

### Usage

```bash
export RAY_ADDRESS=10.102.97.179:7078
export NANOCTRL_ADDRESS=10.102.97.179:3000

python examples/disagg.py \
    --model /models/deepseek-v3 \
    --prefill.master_address 10.102.97.183:6006 \
    --prefill.attention_dp 8 --prefill.ffn_ep 8 \
    --decode.master_address 10.102.97.179:6006 \
    --decode.attention_dp 8 --decode.ffn_ep 8 \
    --decode.loop_count 16 \
    --max_tokens 256 --temperature 0.1 \
    --prompt "What is 1+1?"
```

### Design Notes

- `model` is shared: specified once at top-level, injected into both prefill and decode `Config`.
- `ray_address` and `nanoctrl_address` are read from environment variables (`RAY_ADDRESS` / `NANOCTRL_ADDRESS`), not CLI args, since they are typically cluster-wide settings.
- Per-role `Config` fields use `jsonargparse` nested groups (`nested_key="prefill"` / `nested_key="decode"`), giving natural `--prefill.xxx` / `--decode.xxx` scoping.
- Both scripts support `--config config.yaml` for file-based configuration via `ActionConfigFile`.

## History

These scripts were consolidated from four separate files:

| Old files                                       | Merged into     |
| ----------------------------------------------- | --------------- |
| `deepseek_v3_non_disagg.py`, `pd_non_disagg.py` | `non_disagg.py` |
| `deepseek_v3_disagg.py`, `pd_disagg.py`         | `disagg.py`     |

The old scripts had hardcoded model paths, addresses, and parallelism configs.
The new scripts parameterize everything via CLI / env vars / config files.
