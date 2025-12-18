cmake_minimum_required(VERSION 3.18)
project(nanodeploy_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ===== Dependencies =====
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ===== Sources =====
set(CPP_SOURCES
    src/sequence_metric.cpp
    src/server_metric.cpp
    src/sequence.cpp
    src/block.cpp
    src/block_manager.cpp
)

set(BINDING_SOURCES
    bindings/module.cpp
    bindings/metric_binding.cpp
    bindings/sequence_binding.cpp
    bindings/block_manager_binding.cpp
)

# ===== Build pybind11 module =====
pybind11_add_module(_nanodeploy_cpp
    ${CPP_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(_nanodeploy_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ===== Installation =====
# Install to nanodeploy/_cpp directory
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/_cpp")

install(TARGETS _nanodeploy_cpp
    LIBRARY DESTINATION ${INSTALL_DIR}
    RUNTIME DESTINATION ${INSTALL_DIR}
)

# ===== Compile Options =====
if(MSVC)
    target_compile_options(_nanodeploy_cpp PRIVATE /O2 /W4 /EHsc)
else()
    target_compile_options(_nanodeploy_cpp PRIVATE -O3 -Wall -Wextra)
endif()
