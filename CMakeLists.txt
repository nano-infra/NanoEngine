cmake_minimum_required(VERSION 3.16)

project(_nanodeploy_cpp C CXX)

# CXX Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DLSLIME_INSTALL_PATH "lib" CACHE STRING "Library installation directory")
set(NANODEPLOY_INSTALL_PATH "nanodeploy" CACHE STRING "Python module installation directory")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)
find_package(Spoke REQUIRED)

include_directories(csrc)
add_subdirectory(csrc)

add_subdirectory(tests)
