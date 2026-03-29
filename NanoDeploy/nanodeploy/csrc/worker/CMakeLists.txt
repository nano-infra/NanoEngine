set(CPP_SOURCES
    model_runner_utils.cpp
)

add_library(
    _nanodeploy_worker
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_nanodeploy_worker PUBLIC _nanosequence)

target_compile_options(_nanodeploy_worker PRIVATE -O3 -Wall -Wextra)

set_target_properties(
    _nanodeploy_worker
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_worker
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
