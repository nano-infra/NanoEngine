# NanoOps

Operations CLI for NanoInfra distributed LLM inference.

## Installation

```bash
cd NanoOps
pip install -e .
```

## Quick Start

```bash
# Create a session (optionally starts NanoCtrl)
nanoctrl create --session-id my-session

# Attach to the session so subsequent commands use it automatically
nanoctrl attach my-session

# Set model (parallelism is per-component at deploy time)
nanoctrl set --model /models/llama2-7b

# Deploy components (parallelism flags for prefill/decode)
nanoctrl deploy route
nanoctrl deploy prefill --attention-tp 1 --attention-dp 8 --ffn-ep 8
nanoctrl deploy decode  --attention-tp 1 --attention-dp 8 --ffn-ep 8

# Check status and endpoint
nanoctrl status
# System is ready when route is RUNNING and engines are registered.
```

## Environment (cloud-native)

Connection settings are read from environment variables when not passed via CLI:

- `NANOCTRL_REDIS_URL` - Redis URL (default: redis://localhost:6379)
- `RAY_ADDRESS` - Ray dashboard (default: http://localhost:8265)
- `NANOCTRL_ADDRESS` - NanoCtrl HTTP (default: http://localhost:3000)

## Commands

| Command                                             | Description                                        |
| --------------------------------------------------- | -------------------------------------------------- |
| `nanoctrl create --session-id <id>`                 | Create session; optionally start NanoCtrl          |
| `nanoctrl attach <id>`                              | Enter session shell (env + prompt); exit to detach |
| `nanoctrl set --model <path>`                       | Set model for session                              |
| `nanoctrl deploy route \| prefill \| decode [opts]` | Deploy a component via Ray                         |
| `nanoctrl status`                                   | Session and component status; route endpoint       |
| `nanoctrl stop`                                     | Stop session (Ray jobs + optional Redis cleanup)   |
| `nanoctrl list [--all]`                             | List sessions                                      |
| `nanoctrl job stop/rm/logs/status <job_id>`         | Manage Ray jobs for the session                    |
| `nanoctrl cleanup`                                  | Clean stale processes and Redis keys               |

## Design

For architecture, session model, Redis key layout, orchestrator, and backend clients, see **[DESIGN.md](DESIGN.md)**.
