set(CPP_SOURCES
    sequence.cpp
    block.cpp
    block_manager.cpp
    sp_state_manager.cpp
)

add_library(
    _nanodeploy_engine
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_nanodeploy_engine PRIVATE -O3 -Wall -Wextra)

target_link_libraries(_nanodeploy_engine PRIVATE _nanodeploy_metrics)

set_target_properties(
    _nanodeploy_engine
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_engine
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
