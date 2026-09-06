"""Optional-kernel capability probes for the GDN components.

Centralises every ``try/except`` kernel import used by the GDN prefill,
decode, and convolution components, plus the derived ``_HAS_*`` flags. The
component mixins reference these callables as *module attributes*
(``kernels.chunk_gated_delta_rule`` etc.) so that tests can monkeypatch a
single module.
"""

from dlengine.logging import get_logger

logger = get_logger()

# --- Triton helper kernels (with pure-torch fallbacks resolved by callers) ----
try:
    from dlengine.runtime.kernel.triton.generic.repeat_interleave import (
        can_use_repeat_interleave_from_prefix_triton,
        repeat_interleave_from_prefix_triton,
    )
except ImportError:
    can_use_repeat_interleave_from_prefix_triton = None
    repeat_interleave_from_prefix_triton = None

try:
    from dlengine.runtime.kernel.triton.generic.repeat_heads import (
        can_use_repeat_heads_triton,
        repeat_heads_triton,
    )
except ImportError:
    can_use_repeat_heads_triton = None
    repeat_heads_triton = None

try:
    from dlengine.runtime.kernel.triton.generic.ragged_layout import (
        can_use_ragged_to_padded_triton,
        ragged_to_padded_triton,
    )
except ImportError:
    can_use_ragged_to_padded_triton = None
    ragged_to_padded_triton = None

# --- FlashInfer GDN prefill/decode kernels ------------------------------------
try:
    from flashinfer import chunk_gated_delta_rule

    HAS_FLASHINFER_GDN_PREFILL = callable(chunk_gated_delta_rule)
except ImportError:
    chunk_gated_delta_rule = None
    HAS_FLASHINFER_GDN_PREFILL = False
    logger.warning(
        "flashinfer GDN kernels not available. GatedDeltaNet will use naive fallback."
    )

try:
    from flashinfer.gdn_decode import (
        gated_delta_rule_decode_pretranspose,
        run_pretranspose_decode as _run_pretranspose_decode,
    )

    HAS_FLASHINFER_GDN_PRETRANSPOSE = callable(
        gated_delta_rule_decode_pretranspose
    ) and callable(_run_pretranspose_decode)
except ImportError:
    gated_delta_rule_decode_pretranspose = None
    _run_pretranspose_decode = None
    HAS_FLASHINFER_GDN_PRETRANSPOSE = False

try:
    from flashinfer.gdn_decode import (
        gated_delta_rule_decode,
        run_nontranspose_decode as _run_nontranspose_decode,
    )

    HAS_FLASHINFER_GDN_NONTRANSPOSE = callable(gated_delta_rule_decode) and callable(
        _run_nontranspose_decode
    )
except ImportError:
    gated_delta_rule_decode = None
    _run_nontranspose_decode = None
    HAS_FLASHINFER_GDN_NONTRANSPOSE = False

# --- flash-linear-attention GDN kernels ---------------------------------------
try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as fla_fused_recurrent_gated_delta_rule,
    )

    HAS_FLA_GDN = True
except ImportError:
    fla_chunk_gated_delta_rule = None
    fla_fused_recurrent_gated_delta_rule = None
    HAS_FLA_GDN = False

# --- causal_conv1d optimized depthwise conv -----------------------------------
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states

    HAS_CAUSAL_CONV1D = True
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    causal_conv1d_varlen_states = None
    HAS_CAUSAL_CONV1D = False
