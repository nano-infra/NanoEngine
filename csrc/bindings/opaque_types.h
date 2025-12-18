#pragma once
#include <pybind11/pybind11.h>
#include <unordered_map>
#include "sequence.h"

// Declare opaque map types *before* including <pybind11/stl.h> to prevent
// automatic conversion to Python dict copies.
// This header must be included in every binding file that uses these types
// with pybind11/stl.h, BEFORE including pybind11/stl.h.

PYBIND11_MAKE_OPAQUE(std::unordered_map<int, int>);
PYBIND11_MAKE_OPAQUE(nanodeploy::Sequence::BlockCtxMap);
