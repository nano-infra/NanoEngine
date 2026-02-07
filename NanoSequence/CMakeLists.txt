cmake_minimum_required(VERSION 3.16)

# Check if this is a standalone build or being included as subdirectory
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Standalone build
    project(_nanosequence CXX)
    set(NANOSEQUENCE_STANDALONE_BUILD ON)
else()
    # Included as subdirectory
    set(NANOSEQUENCE_STANDALONE_BUILD OFF)
endif()

# NanoSequence library
# Note: This CMakeLists.txt can be used standalone or as a subdirectory from NanoDeploy
# Dependencies (third_party, NanoCommon) are provided by the parent project when used as subdirectory

# CXX Configuration - must match parent project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set output directories for standalone build
if(NANOSEQUENCE_STANDALONE_BUILD)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# Define NanoInfra root path if not already defined
if(NOT DEFINED NANOINFRA_DIR)
    set(NANOINFRA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()

# Only add dependencies if they don't already exist
if(NOT TARGET flatbuffers)
    if(EXISTS ${NANOINFRA_DIR}/third_party/flatbuffers/CMakeLists.txt)
        add_subdirectory(${NANOINFRA_DIR}/third_party/flatbuffers third_party/flatbuffers)
    endif()
endif()

if(NOT TARGET nanocommon)
    if(EXISTS ${NANOINFRA_DIR}/NanoCommon/CMakeLists.txt)
        add_subdirectory(${NANOINFRA_DIR}/NanoCommon NanoCommon)
    endif()
endif()

# Include directories
include_directories(.)
add_subdirectory(nanosequence/csrc/metrics)
add_subdirectory(nanosequence/csrc/sequence)

# Add Python bindings if building standalone (not as subdirectory)
# When included as subdirectory, bindings are handled by parent project
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # This is a standalone build
    add_subdirectory(nanosequence/csrc/bind)
endif()
