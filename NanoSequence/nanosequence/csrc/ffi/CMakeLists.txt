set(FFI_SOURCES
    nanosequence_c.cpp
)

# Create a static library for the C FFI interface
add_library(
    nanosequence_ffi
    STATIC
    ${FFI_SOURCES}
)

target_link_libraries(nanosequence_ffi PRIVATE _nanosequence nanocommon)

target_compile_options(nanosequence_ffi PRIVATE -O3 -Wall -Wextra)

# Ensure C++20 standard
set_target_properties(nanosequence_ffi PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Include directories
target_include_directories(nanosequence_ffi PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/NanoSequence>
    $<INSTALL_INTERFACE:include>
)

# Install headers
install(
    FILES nanosequence_c.h
    DESTINATION include/nanosequence/csrc/ffi
)

# Install library
install(
    TARGETS nanosequence_ffi
    ARCHIVE DESTINATION lib
)
