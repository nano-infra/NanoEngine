cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(flatbuffers)
