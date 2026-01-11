set(CPP_SOURCES
    block.cpp
    block_manager.cpp
    load_statistics.cpp
    sp_size_policy.cpp
    sp_state_manager.cpp
    scheduler_utils.cpp
    scheduler.cpp
)

add_library(
    _nanodeploy_scheduler
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_nanodeploy_scheduler PRIVATE -O3 -Wall -Wextra)

target_link_libraries(_nanodeploy_scheduler PRIVATE _nanodeploy_metrics _nanodeploy_sequence)

set_target_properties(
    _nanodeploy_scheduler
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_scheduler
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
