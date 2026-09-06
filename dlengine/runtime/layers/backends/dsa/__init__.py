"""DSA (DeepSeek/GLM sparse attention) backend family.

DSA composes an *indexer* (top-k key selection) with a *sparse MLA kernel*
(attend only the selected keys). It is dense MLA plus an index mask, so its
sparse kernels build on ``backends/mla/``.

Sub-packages:

- ``indexer`` : the NSA Lightning-Indexer (paged FP8, reference, pooled).
- ``state``   : ``IndexerTopKState`` — top-k selection shared across layers /
  PP boundaries / MTP iterations.
- ``sparse``  : sparse MLA prefill/decode kernels (added in a later step).
- ``dsa_attention`` : ``DsaAttention`` composing the above (added in a later
  step).
"""

from dlengine.runtime.layers.backends.dsa.state import (
    IndexerTopKState,
    _IndexerTopKState,
)

__all__ = ["IndexerTopKState", "_IndexerTopKState"]
