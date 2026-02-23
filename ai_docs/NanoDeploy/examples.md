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
Common config is set at top-level; per-role overrides use `--prefill.xxx` / `--decode.xxx` scoping.

### Common Config + Per-Role Overlay

| Scope                  | How to set      | Examples                                                 |
| ---------------------- | --------------- | -------------------------------------------------------- |
| **Common** (top-level) | `--xxx`         | `--model /models/deepseek-v3`, `--kvcache_block_size 64` |
| **Prefill-only**       | `--prefill.xxx` | `--prefill.master_address 10.0.0.2:6006`                 |
| **Decode-only**        | `--decode.xxx`  | `--decode.loop_count 16`                                 |

Common values apply to both roles unless explicitly overridden by a scoped arg.

### Usage

```bash
python examples/disagg.py \
    --model /models/deepseek-v3 \
    --ray_address 10.102.97.179:7078 \
    --nanoctrl_address 10.102.97.179:3000 \
    --kvcache_block_size 64 \
    --attention_dp 8 --ffn_ep 8 \
    --prefill.master_address 10.102.97.183:6006 \
    --decode.master_address 10.102.97.179:6006 \
    --decode.loop_count 16 \
    --max_tokens 256 --temperature 0.1 \
    --prompt "What is 1+1?"
```

### Design Notes

- `model`, `ray_address`, `nanoctrl_address`, `kvcache_block_size`, etc. are **common config** — set once at top-level, applied to both roles.
- Per-role `--prefill.xxx` / `--decode.xxx` values **overlay** on top of common config. Only explicitly set values override; unset scoped args inherit from common.
- Both scripts support `--config config.yaml` for file-based configuration via `ActionConfigFile`.

## History

These scripts were consolidated from four separate files:

| Old files                                       | Merged into     |
| ----------------------------------------------- | --------------- |
| `deepseek_v3_non_disagg.py`, `pd_non_disagg.py` | `non_disagg.py` |
| `deepseek_v3_disagg.py`, `pd_disagg.py`         | `disagg.py`     |

The old scripts had hardcoded model paths, addresses, and parallelism configs.
The new scripts parameterize everything via CLI / env vars / config files.
