#!/bin/bash
set -e

# Create build directory
BUILD_DIR=build
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure CMake
# We point to ../csrc where CMakeLists.txt is located
echo "Configuring CMake..."
cmake ../csrc \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")

# Build
echo "Building extension..."
cmake --build . -j$(nproc)

# Install (copies .so to nanodeploy/_cpp/)
echo "Installing extension..."
cmake --install .

cd ..
echo "Build completed successfully! C++ backend is ready."
