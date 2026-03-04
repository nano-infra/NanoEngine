set(CPP_SOURCES
    sequence.cpp
    serialization.cpp
)

# --- FlatBuffers Generation with gen-object-api ---
# Note: FLATC_EXECUTABLE target is available from add_subdirectory(third_party/flatbuffers)
if(TARGET flatc)
    set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
else()
    # Fallback if found on system
    find_program(FLATC_EXECUTABLE flatc REQUIRED)
endif()

set(FBS_SOURCE "${NANOINFRA_DIR}/NanoSequence/proto/sequence.fbs")
set(GEN_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(GEN_HEADER "${GEN_HEADER_DIR}/sequence_generated.h")

add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --scoped-enums
            --gen-object-api
            -o ${GEN_HEADER_DIR}
            ${FBS_SOURCE}
    DEPENDS ${FBS_SOURCE}
            flatc
    COMMENT "Generating FlatBuffers header for Sequence with gen-object-api"
)

# --- FlatBuffers Generation for interface.fbs (lean engine-runner protocol) ---
set(INTERFACE_FBS_SOURCE "${NANOINFRA_DIR}/NanoSequence/proto/interface.fbs")
set(INTERFACE_GEN_HEADER "${GEN_HEADER_DIR}/interface_generated.h")

add_custom_command(
    OUTPUT ${INTERFACE_GEN_HEADER}
    COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --scoped-enums
            --gen-object-api
            -o ${GEN_HEADER_DIR}
            ${INTERFACE_FBS_SOURCE}
    DEPENDS ${INTERFACE_FBS_SOURCE}
            flatc
    COMMENT "Generating FlatBuffers header for interface (engine-runner protocol)"
)

# --- Target Definition ---
add_library(
    _nanosequence
    SHARED
    ${CPP_SOURCES}
    ${GEN_HEADER}
    ${INTERFACE_GEN_HEADER}
)

# Set output directory to match other NanoDeploy libraries
# This ensures the library is built in the same location as _nanodeploy_cpp
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_nanosequence PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

# Ensure C++20 standard (required for std::source_location)
set_target_properties(_nanosequence PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(_nanosequence PRIVATE -O3 -Wall -Wextra)

# Link flatbuffers (header only or static lib) and metrics
# Use PUBLIC for flatbuffers to propagate its include directories to consumers
target_link_libraries(_nanosequence PUBLIC flatbuffers PRIVATE nanocommon _nanosequence_metrics)

# Include generated header directory
target_include_directories(_nanosequence PRIVATE ${GEN_HEADER_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# Include directories for headers
# The PUBLIC include directory should point to NanoSequence root so that
# consumers can include headers as "nanosequence/csrc/sequence/sequence.h"
# Also include the generated header directory so consumers can access sequence_generated.h
# Also include flatbuffers include directory so generated headers can find flatbuffers/flatbuffers.h
target_include_directories(_nanosequence PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/NanoSequence>
    $<BUILD_INTERFACE:${GEN_HEADER_DIR}>
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/third_party/flatbuffers/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(
    _nanosequence
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

# Install to the same location as other NanoSequence libraries
# Use NANOSEQUENCE_INSTALL_PATH if defined (from pyproject.toml), otherwise default to nanosequence
if(DEFINED NANOSEQUENCE_INSTALL_PATH)
    set(NANOSEQUENCE_LIB_INSTALL_PATH ${NANOSEQUENCE_INSTALL_PATH})
else()
    # Default to nanosequence for standalone builds, or use NANODEPLOY_INSTALL_PATH if building as subdirectory
    if(DEFINED NANODEPLOY_INSTALL_PATH)
        set(NANOSEQUENCE_LIB_INSTALL_PATH ${NANODEPLOY_INSTALL_PATH})
    else()
        set(NANOSEQUENCE_LIB_INSTALL_PATH nanosequence)
    endif()
endif()

install(
    TARGETS
    _nanosequence
    LIBRARY DESTINATION ${NANOSEQUENCE_LIB_INSTALL_PATH}
)
