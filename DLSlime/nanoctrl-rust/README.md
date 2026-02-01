# NanoCtrl Control Plane

Control plane server for DLSlime RDMA connection management.

## Prerequisites

- Redis server running (default: `127.0.0.1:6379`)
- Rust toolchain

## Building

```bash
cd nanoctrl-rust
cargo build --release
```

## Running

```bash
cargo run --release
```

The server will listen on `http://0.0.0.0:3000` by default.

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
