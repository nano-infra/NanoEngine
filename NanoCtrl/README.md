# NanoCtrl Control Plane

Control plane server for DLSlime RDMA connection management.

## Prerequisites

- Redis server running
- Rust toolchain

## Building

```bash
cd NanoCtrl
cargo build --release
```

## Running

```bash
# Default: Redis at 127.0.0.1:6379
cargo run --release

# Or specify Redis URL via env (required for distributed deployment)
export REDIS_URL=redis://your-redis-host:6379
cargo run --release
```

The server will listen on `http://0.0.0.0:3000` by default.

**Distributed deployment**: When ModelRunner/PeerAgent runs on remote nodes (e.g. 10.102.97.183), they need to connect to Redis. If Redis runs on the master node, set:

```bash
export REDIS_URL=redis://127.0.0.1:6379   # NanoCtrl connects to local Redis
export REDIS_PUBLIC_ADDRESS=10.102.97.1   # IP that remote workers use to reach Redis (master node IP)
```

## API Endpoints

- `POST /start_peer_agent` - Register a peer agent
- `POST /query` - Query all registered peer agents
- `POST /init` - Initialize RDMA connection between two agents
- `POST /connect` - Establish RDMA connection
- `POST /register_mr` - Register a memory region
- `POST /get_mr_info` - Get remote memory region info
- `POST /get_endpoint_info` - Get remote endpoint info
- `POST /ack_init` - Acknowledge init completion
- `POST /ack_connect` - Acknowledge connect completion
- `POST /update_endpoint_info` - Update endpoint info

## Environment Variables

- `RUST_LOG` - Log level (default: `info`)
- `REDIS_URL` - Redis connection URL (default: `redis://127.0.0.1:6379`)
- `REDIS_PUBLIC_ADDRESS` - For distributed setup: IP:port that remote workers use to reach Redis (e.g. master node IP)

## Python Client

See `dlslime/peer_agent.py` for the Python client implementation.

Example usage:

```python
from dlslime import start_peer_agent

agent = start_peer_agent(
    alias="my_agent",
    server_url="http://127.0.0.1:3000",
    address="127.0.0.1:6379",
)
```
