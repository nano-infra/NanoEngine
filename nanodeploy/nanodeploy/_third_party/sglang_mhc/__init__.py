"""Vendored sglang DSV4 hyper-connection (HC) kernels.

Source: https://github.com/sgl-project/sglang
        python/sglang/srt/layers/mhc.py

Why vendored: the upstream module ships as part of the sglang Python
package which we don't depend on (we only have ``tilelang`` installed).
This subpackage strips imports to exactly what we need at runtime:
torch + tilelang + the vendored ``is_arch_support_pdl`` helper from our
sibling ``sglang_jit_kernel/`` slice.

License: Apache-2.0 (see sibling sglang_jit_kernel/LICENSE).
"""

from .mhc import hc_split_sinkhorn, mhc_post, mhc_pre  # noqa: F401
