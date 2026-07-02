# AGENTS.md

## Cursor Cloud specific instructions

This is a **GPU-bound LLM inference monorepo** (see `README.md`). The two products are
`dlengine` (Python/C++/CUDA inference engine) and `dlengine-router` (Rust HTTP router).

### What can / cannot run on the cloud VM

- The cloud VM is **CPU-only, no NVIDIA GPU**. The `dlengine` Python/C++ engine requires
  NVIDIA Hopper (SM90+) GPUs + CUDA 12.8 to build (`nvcc`) and run, so it **cannot be built,
  tested, or served here**. Its pytest suite and the examples in `dlengine/examples/` all need
  a GPU. Use the `docker/Dockerfile` image on a GPU host for that work.
- `dlengine-router` (Rust) is the one component fully **buildable / testable / runnable on CPU**,
  so it is the default target for local development and verification in this environment.

### dlengine-router (Rust) — build / lint / test / run

Work from the `dlengine-router/` directory. Standard commands (mirrors `.pre-commit-config.yaml`):

- Build: `cargo build`
- Lint: `cargo fmt --check` and `cargo clippy -- -D warnings`
- Test: `cargo test`

Non-obvious gotchas:

- **`flatc` is required to build.** `build.rs` first looks for `flatc` on `PATH`; if absent it
  builds one from the `third_party/flatbuffers` submodule via CMake. Keep that submodule
  checked out (`git submodule update --init third_party/flatbuffers`). Do NOT rely on a distro
  `flatc` — the generated Rust is patched for the pinned `flatbuffers` crate version.
- The crate has **both a lib and a bin target**; `cargo build`/`test` always compile the lib
  (`src/lib.rs`), which includes `sequence_utils.rs`. The FlatBuffers `SequenceArgs` initializer
  there must set every schema field (e.g. `affinity_key`) — appending a field to
  `dlengine-proto/sequence.fbs` without updating that initializer breaks the build.

### Running the router end-to-end (hello world, no GPU needed)

The router **exits at startup unless it can reach a `dlslime-ctrl` control plane**, which in turn
needs Redis. Bring the stack up in this order:

1. `redis-server --daemonize yes --bind 0.0.0.0 --port 6379 --protected-mode no`
   - **`--protected-mode no` matters:** `dlslime-ctrl` advertises the VM's non-loopback IP to the
     router, and default Redis protected mode rejects those connections, producing endless
     (non-fatal) pub/sub reconnect warnings in the router log.
2. `dlslime-ctrl server --redis-url redis://127.0.0.1:6379` (binary from `pip install dlslime-ctrl`;
   installs to `~/.local/bin`, listens on `:4479`).
3. `cd dlengine-router && ./target/debug/dlengine-router --config config.toml` (serves `:3001`;
   `config.toml` already points `ctrl_address` at `http://127.0.0.1:4479`).

Verify: `curl http://127.0.0.1:3001/health` → `OK`. A `POST /v1/chat/completions` returns
`404 "Model ... not found. Available: []"` — this is expected here because no GPU inference
engines can register; it still proves the OpenAI-compatible routing path executes end-to-end.
