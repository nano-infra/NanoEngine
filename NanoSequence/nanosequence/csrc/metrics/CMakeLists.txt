set(CPP_SOURCES
    sequence_metric.cpp
)

add_library(
    _nanosequence_metrics
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_nanosequence_metrics PRIVATE nanocommon)

target_compile_options(_nanosequence_metrics PRIVATE -O3 -Wall -Wextra)

# Ensure C++20 standard
set_target_properties(_nanosequence_metrics PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

set_target_properties(
    _nanosequence_metrics
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

# Include directories for headers
# The PUBLIC include directory should point to NanoSequence root so that
# consumers can include headers as "nanosequence/csrc/metrics/sequence_metric.h"
target_include_directories(_nanosequence_metrics PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/NanoSequence>
    $<INSTALL_INTERFACE:include>
)

# Set output directory to match other NanoSequence libraries
if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_nanosequence_metrics PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

# Install to the same location as other NanoSequence libraries
# Use NANOSEQUENCE_INSTALL_PATH if defined (from pyproject.toml), otherwise default to nanosequence
if(DEFINED NANOSEQUENCE_INSTALL_PATH)
    set(NANOSEQUENCE_METRICS_INSTALL_PATH ${NANOSEQUENCE_INSTALL_PATH})
else()
    # Default to nanosequence for standalone builds, or use NANODEPLOY_INSTALL_PATH if building as subdirectory
    if(DEFINED NANODEPLOY_INSTALL_PATH)
        set(NANOSEQUENCE_METRICS_INSTALL_PATH ${NANODEPLOY_INSTALL_PATH})
    else()
        set(NANOSEQUENCE_METRICS_INSTALL_PATH nanosequence)
    endif()
endif()

install(
    TARGETS
    _nanosequence_metrics
    LIBRARY DESTINATION ${NANOSEQUENCE_METRICS_INSTALL_PATH}
)
