"""Hot weight-update helpers for ``ModelRunner``.

Single responsibility: apply a dict of HF-named full tensors into a live
``nn.Module``, in place. Decoupled from NanoInfra's engine + Ray plumbing so
the apply path can be unit-tested or reused elsewhere.

Key correctness contract:

- All copies go through ``param.data.copy_(...)`` (or the parameter's
  attached ``weight_loader`` callback, which itself ends in ``copy_``).
  This preserves the storage's address and keeps any captured CUDA graphs
  valid.
- For TP/EP-sharded layers, NanoInfra's parameter constructors at
  ``nanodeploy/backends/hopper/layers/linear.py`` attach a
  ``weight_loader`` callback that already knows this rank's slice. We
  simply call it instead of duplicating the slicing logic here.
- Tensors the model doesn't have a parameter for (e.g. metadata-only HF
  keys, or the model's MTP head weights when MTP is disabled) are silently
  skipped — the caller controls strict-ness.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("nanodeploy")


def apply_named_tensors_in_place(
    model: torch.nn.Module,
    named_tensors: dict[str, torch.Tensor],
    *,
    sync: bool = True,
) -> dict[str, int]:
    """Apply ``named_tensors`` to matching parameters of ``model`` in place.

    Returns a small stats dict the caller can log for visibility:
        loaded:           tensors successfully copied into the model
        skipped_unknown:  tensors whose name is not a parameter of ``model``
        used_loader_cb:   subset of ``loaded`` that went through the
                          per-parameter ``weight_loader`` callback (TP path)
        used_direct_copy: subset of ``loaded`` that fell back to plain
                          ``param.data.copy_`` (replicated / non-TP path)
    """
    n_loaded = n_skipped = n_loader = n_direct = 0
    for name, full in named_tensors.items():
        try:
            param = model.get_parameter(name)
        except AttributeError:
            n_skipped += 1
            continue
        loader = getattr(param, "weight_loader", None)
        if loader is not None:
            loader(param, full)
            n_loader += 1
        else:
            param.data.copy_(full, non_blocking=True)
            n_direct += 1
        n_loaded += 1
    if sync:
        torch.cuda.synchronize()
    stats = {
        "loaded": n_loaded,
        "skipped_unknown": n_skipped,
        "used_loader_cb": n_loader,
        "used_direct_copy": n_direct,
    }
    logger.info("apply_named_tensors_in_place: %s", stats)
    return stats
