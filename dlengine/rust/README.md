# dlengine Rust Runtime

This directory contains the Rust/PyO3 runtime extension used by `dlengine`.
It provides the Python-facing control plane: scheduler state, cache plans,
metrics, binary request serialization, and L3 block-management logic.

Python code imports the curated wrapper `dlengine._rust`. The native PyO3
extension is installed as `dlengine._dlengine_rust` and is intentionally kept
one layer below the package-facing API.

## Layout

- `_dlengine_rust/src/sequence.rs`: request sequence and sampling parameter
  types.
- `_dlengine_rust/src/scheduler/`: scheduler, resource allocation, lifecycle,
  routing, metrics, and session-state cache logic.
- `_dlengine_rust/src/config/`: cache-plan specs shared by engine and
  context setup.
- `_dlengine_rust/src/proto/`: binary wire format helpers and batch metadata
  preparation for model runners.
- `_dlengine_rust/src/l3.rs`: L3 prefix-cache block manager internals.
- `dlengine/_dlengine_rust.pyi`: Python type stubs for IDE completion and type
  checkers.

## Build

From the `dlengine` package root:

```bash
maturin develop
```

The package is configured as a mixed Python/Rust project in `pyproject.toml`,
with the extension module installed as `dlengine._dlengine_rust`.

## Quick Checks

```bash
cargo check
pytest -q tests/test_rust_strong_types.py
python tests/test_l3_block_manager.py
```

The 3FS-backed L3 integration test is environment-dependent and skips when no
3FS mount is available:

```bash
python tests/test_l3_integration.py
```
