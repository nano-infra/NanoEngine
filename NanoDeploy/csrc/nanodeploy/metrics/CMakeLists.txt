set(CPP_SOURCES
    sequence_metric.cpp
    server_metric.cpp
)

add_library(
    _nanodeploy_metrics
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_nanodeploy_metrics PRIVATE nanocommon)

target_compile_options(_nanodeploy_metrics PRIVATE -O3 -Wall -Wextra)

set_target_properties(
    _nanodeploy_metrics
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_metrics
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
