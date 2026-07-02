# dlengine Rust Extension

This directory contains the first Rust/PyO3 control-plane extension for
`dlengine`.

The intended migration shape is incremental:

1. Keep existing C++/CUDA kernels and model runner code in place.
2. Move Python-facing control-plane structs into Rust first.
3. Move scheduler/cache/metrics logic behind the same narrow Python API.
4. Delete the matching pybind11 surface once each piece is no longer used.

Build locally with maturin:

```bash
cd dlengine
maturin develop
```
