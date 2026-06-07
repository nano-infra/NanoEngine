cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt" OR NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers/CMakeLists.txt")
  message(FATAL_ERROR "Git submodules are missing. Please run: git submodule update --init --recursive")
endif()

add_subdirectory(pybind11)
add_subdirectory(flatbuffers)
