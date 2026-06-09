"""Central control over ``torch.compile`` usage.

All nanodeploy call sites go through :func:`maybe_compile` instead of calling
``torch.compile`` directly. This gives a single, sound switch to run every
compiled path in plain eager mode — when disabled, the original callable is
returned *unwrapped*, so ``torch.compile`` (and therefore the inductor/triton
backend) is never invoked at all. This matters on platforms whose backend is
not adapted (e.g. PPU), where invoking the compiler raises ``BackendCompilerFailed``.

The switch can be set two ways (both checked at wrap time):
  * env var ``NANODEPLOY_DISABLE_COMPILE=1`` — useful before any config exists;
  * :func:`set_compile_disabled` — called by the worker from ``Config.disable_compile``.
"""

import os

import torch

# Initial value from the environment so it is honoured even outside the
# Ray-actor/config path (e.g. unit tests, standalone imports).
_COMPILE_DISABLED: bool = os.getenv("NANODEPLOY_DISABLE_COMPILE", "0") == "1"


def set_compile_disabled(disabled: bool) -> None:
    """Globally enable/disable nanodeploy's ``torch.compile`` wrappers.

    Must be called *before* compiled layers/functions are constructed or first
    invoked (the worker sets it before model construction). Lazily-compiled
    helpers read the flag on first call, so this also covers them.
    """
    global _COMPILE_DISABLED
    _COMPILE_DISABLED = bool(disabled)


def is_compile_disabled() -> bool:
    return _COMPILE_DISABLED


def maybe_compile(fn=None, **compile_kwargs):
    """``torch.compile`` that is a no-op when compilation is disabled.

    Usable as a direct call — ``maybe_compile(fn)`` /
    ``maybe_compile(fn, dynamic=False, fullgraph=True)`` — or as a decorator —
    ``@maybe_compile`` / ``@maybe_compile(dynamic=False)``.

    When disabled, returns ``fn`` unchanged so ``torch.compile`` is never called.
    """

    def _wrap(f):
        if _COMPILE_DISABLED:
            return f
        return torch.compile(f, **compile_kwargs)

    return _wrap if fn is None else _wrap(fn)
