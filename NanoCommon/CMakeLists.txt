cmake_minimum_required(VERSION 3.16)

# Define NanoCommon Target
project(nanocommon CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only or small static lib?
# Since it has some implementation in headers but maybe future .cpp, let's make it an INTERFACE or STATIC
# But jring is C-style.
# Use INTERFACE for now if it's all headers, but we might want a real lib for proper linking.

add_library(nanocommon INTERFACE)

target_include_directories(nanocommon INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(nanocommon INTERFACE
    dl # For backtrace/dlopen
)

# Installation (optional, if valid install prefix)
install(TARGETS nanocommon
    EXPORT nanocommonTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/nanocommon
    DESTINATION include
)
