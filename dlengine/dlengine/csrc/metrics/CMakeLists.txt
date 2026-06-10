add_library(
    _dlengine_metrics
    SHARED
    server_metric.cpp
)

target_compile_options(_dlengine_metrics PRIVATE -O3 -Wall -Wextra)

set_target_properties(_dlengine_metrics PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(TARGETS _dlengine_metrics LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH})

# Sequence metrics library (C++ runtime; schemas live in top-level dlengine-proto/)
add_library(
    _nanosequence_metrics
    SHARED
    sequence_metric.cpp
)

target_compile_options(_nanosequence_metrics PRIVATE -O3 -Wall -Wextra)

set_target_properties(_nanosequence_metrics PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

# PUBLIC include: DLEngine root so consumers use "dlengine/csrc/metrics/sequence_metric.h"
target_include_directories(_nanosequence_metrics PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/DLEngine>
    $<INSTALL_INTERFACE:include>
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_nanosequence_metrics PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

install(TARGETS _nanosequence_metrics LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH})
