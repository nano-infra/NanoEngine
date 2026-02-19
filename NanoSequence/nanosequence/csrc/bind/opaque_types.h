#pragma once
#include "nanosequence/csrc/sequence/sequence.h"
#include <pybind11/pybind11.h>
#include <unordered_map>
#include <vector>

// Pybind11-specific wrapper types for opaque vector bindings.
// These exist solely so pybind11 can distinguish them from std::vector<int>
// and bind them as mutable reference-based containers instead of copy-converted ones.
struct BlockIdList: public std::vector<int> {
    using std::vector<int>::vector;
};
struct SpBlockTable: public std::vector<BlockIdList> {
    using std::vector<BlockIdList>::vector;
};

// Declare opaque map types *before* including <pybind11/stl.h> to prevent
// automatic conversion to Python dict copies.
// This header must be included in every binding file that uses these types
// with pybind11/stl.h, BEFORE including pybind11/stl.h.

PYBIND11_MAKE_OPAQUE(std::unordered_map<int, int>);
