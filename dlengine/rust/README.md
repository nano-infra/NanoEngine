# dlengine Rust Runtime

This directory contains the Rust/PyO3 runtime extension used by `dlengine`.
It provides the Python-facing control plane that used to live behind the
`dlengine._cpp` pybind11 module: sequences, scheduler state, cache plans,
metrics, binary request serialization, and L3 block-management hooks.

Python imports continue to use `dlengine._cpp` for compatibility, but that
module now re-exports the Rust extension. New code may import from
`dlengine._rust` or `dlengine._dlengine_rust` directly when it wants to be
explicit.

## Layout

- `_dlengine_rust/src/sequence.rs`: request sequence and sampling parameter
  types.
- `_dlengine_rust/src/scheduler/`: scheduler, resource allocation, lifecycle,
  routing, metrics, and session-state cache logic.
- `_dlengine_rust/src/cache_plan.rs`: cache-plan specs shared by engine and
  context setup.
- `_dlengine_rust/src/stubs/`: binary wire format helpers and batch metadata
  preparation for model runners.
- `_dlengine_rust/src/l3.rs`: L3 prefix-cache block manager compatibility API.
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
