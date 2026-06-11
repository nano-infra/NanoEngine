"""Dynamic build config for the `nanoinfra` monorepo meta-package.

The optional-dependencies cannot be expressed statically in pyproject.toml
because the local subpackages must be referenced by *absolute* `file://`
URLs. Modern `packaging` (>=24) rejects relative direct references such as
``dlengine @ file:dlengine`` with "Invalid URL given", so we compute absolute
URLs based on this file's location at build time, keeping the result portable
across machines/checkout paths.
"""

import os
from pathlib import Path

from setuptools import setup

_ROOT = Path(__file__).parent.resolve()


def _local(rel_path: str) -> str:
    # Path.as_uri() yields a valid absolute "file:///..." URL on all platforms.
    return (_ROOT / rel_path).as_uri()


_DLENGINE = f"dlengine @ {_local('dlengine')}"
_DLENGINE_VL = f"dlengine[vl] @ {_local('dlengine')}"

setup(
    extras_require={
        "dlengine": [_DLENGINE],
        # DLEngineVL was folded into dlengine as the `dlengine.vl`
        # subpackage; this extra just pulls dlengine with its VL extras.
        "dlenginevl": [_DLENGINE_VL],
        "all": [_DLENGINE_VL],
    },
)
