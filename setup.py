"""Dynamic build config for the `nanoinfra` monorepo meta-package.

The optional-dependencies cannot be expressed statically in pyproject.toml
because the local subpackages must be referenced by *absolute* `file://`
URLs. Modern `packaging` (>=24) rejects relative direct references such as
``nanodeploy-kernel @ file:nanodeploy-kernel`` with "Invalid URL given", so we
compute absolute URLs based on this file's location at build time, keeping the
result portable across machines/checkout paths.
"""

import os
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).parent.resolve()


def _local(rel_path: str) -> str:
    # Path.as_uri() yields a valid absolute "file:///..." URL on all platforms.
    return (_ROOT / rel_path).as_uri()


_KERNEL = f"nanodeploy-kernel @ {_local('nanodeploy-kernel')}"
_NANODEPLOY = f"nanodeploy @ {_local('nanodeploy')}"
_NANODEPLOY_VL = f"nanodeploy[vl] @ {_local('nanodeploy')}"

setup(
    extras_require={
        "nanodeploy-kernel": [_KERNEL],
        "nanodeploy": [_KERNEL, _NANODEPLOY],
        # NanoDeployVL was folded into nanodeploy as the `nanodeploy.vl`
        # subpackage; this extra just pulls nanodeploy with its VL extras.
        "nanodeployvl": [_KERNEL, _NANODEPLOY_VL],
        "all": [_KERNEL, _NANODEPLOY_VL],
    },
)
