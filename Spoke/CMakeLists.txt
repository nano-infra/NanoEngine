cmake_minimum_required(VERSION 3.10)
project(Spoke VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Set build output layout
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# RPATH configuration
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# --- Vendor DLSlime ---
set(BUILD_RDMA ON CACHE BOOL "Enable RDMA" FORCE)
set(BUILD_PYTHON OFF CACHE BOOL "Disable Python bindings" FORCE)
set(BUILD_TEST OFF CACHE BOOL "Disable Tests" FORCE)
set(BUILD_BENCH OFF CACHE BOOL "Disable Benchmarks" FORCE)
# Disable CUDA for DLSlime - use HostOnlySignal for RDMA synchronization
# This allows RDMA to work on CPU-only processes (e.g., EngineActor)
set(USE_CUDA OFF CACHE BOOL "DLSlime: Disable CUDA" FORCE)
if(NOT TARGET dlslime)
    add_subdirectory(third_party/DLSlime)
endif()

# Header files installation
file(GLOB SPOKE_HEADERS
    "spoke/csrc/*.h"
)
install(FILES ${SPOKE_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spoke)

add_subdirectory(spoke/csrc)
add_subdirectory(spoke/sdk)
add_subdirectory(spoke/tools)
add_subdirectory(examples)

# --- CMake Package Export ---
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})

configure_package_config_file(
    "cmake/SpokeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SpokeConfig.cmake"
    INSTALL_DESTINATION share/cmake/Spoke
    PATH_VARS INCLUDE_INSTALL_DIR BIN_INSTALL_DIR
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SpokeConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT SpokeTargets
    FILE SpokeTargets.cmake
    NAMESPACE Spoke::
    DESTINATION share/cmake/Spoke
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SpokeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SpokeConfigVersion.cmake"
    DESTINATION share/cmake/Spoke
)
