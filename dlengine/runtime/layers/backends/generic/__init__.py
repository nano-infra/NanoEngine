"""Generic (BF16 / reference) layer implementations.

This is the portable ``ref`` implementation family: correctness-first,
non-optimized-GPU implementations that do not require Hopper/Blackwell FP8 or
vendor kernels. Selection policy lives elsewhere; this package only owns the
implementations.
"""
