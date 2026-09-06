"""GatedDeltaNet / KDA (linear-attention) backend family.

- ``generic``    : GenericGatedDeltaNet    (composes the ``components/`` mixins)
- ``flashinfer`` : FlashInferGatedDeltaNet
- ``fla``        : FlaGatedDeltaNet
- ``torch``      : TorchGatedDeltaNet       (naive/debug)
- ``kda``        : FlashInferKda            (Kimi Delta Attention)

Shared components (conv / state / recurrence / output / kernels) live in
``components/``.
"""
