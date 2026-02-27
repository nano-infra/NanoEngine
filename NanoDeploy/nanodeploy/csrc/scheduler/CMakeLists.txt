set(CPP_SOURCES
    block.cpp
    block_manager.cpp
    state_manager.cpp
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

# Link _nanosequence as PUBLIC to propagate include directories to consumers
# Other dependencies can be PRIVATE since scheduler's headers don't expose them
target_link_libraries(_nanodeploy_scheduler
    PRIVATE _nanodeploy_metrics
    PUBLIC _nanosequence _nanosequence_metrics nanocommon
)

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
