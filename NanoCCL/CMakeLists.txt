cmake_minimum_required(VERSION 3.16)

project(nanoccl C CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# NanoCCL options for ops
nanoccl_option(BUILD_INTRA_OPS           "Build intra LL collective ops"    OFF)
nanoccl_option(BUILD_INTER_OPS           "Build inter LL collective ops"    OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(NANOCCL_INSTALL_PATH "lib" CACHE STRING "Library installation directory")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (BUILD_INTRA_OPS OR BUILD_INTER_OPS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "90")
    find_package(CUDAToolkit REQUIRED)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/torch.cmake)
endif()

# --- Dependencies ---
# Define NanoInfra root path (for third_party when building from NanoInfra)
if(NOT DEFINED NANOINFRA_DIR)
  set(NANOINFRA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
message(STATUS "NanoInfra Root Path: ${NANOINFRA_DIR}")

add_subdirectory(${NANOINFRA_DIR}/NanoCommon NanoCommon)

# Include NanoCommon for logging
include_directories(${CMAKE_CURRENT_SOURCE_DIR} nanoccl/csrc ${NANOINFRA_DIR}/NanoCommon/include)

if (BUILD_INTRA_OPS OR BUILD_INTER_OPS)
    add_subdirectory(nanoccl/csrc/ops)
endif()

# Python bindings
nanoccl_option(BUILD_PYTHON              "Build python wrapper"             OFF)

if (BUILD_PYTHON)
    add_subdirectory(nanoccl/csrc/python)
endif()

# --- Installation & Export ---

# 1. Install Headers
install(
    DIRECTORY nanoccl/csrc/
    DESTINATION include/nanoccl
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
    PATTERN "python" EXCLUDE  # Exclude python bindings source if present
)
