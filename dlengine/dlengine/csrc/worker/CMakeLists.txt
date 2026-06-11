set(CPP_SOURCES
    model_runner_utils.cpp
)

add_library(
    _dlengine_worker
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_dlengine_worker PUBLIC _nanosequence)

target_compile_options(_dlengine_worker PRIVATE -O3 -Wall -Wextra)

set_target_properties(
    _dlengine_worker
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _dlengine_worker
    LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH}
)
