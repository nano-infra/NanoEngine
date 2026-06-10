set(CPP_SOURCES
    group_manager.cpp
    scheduler.cpp
)

add_library(
    _dlengine_scheduler
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_dlengine_scheduler PRIVATE -O3 -Wall -Wextra)

# Link _nanosequence as PUBLIC to propagate include directories to consumers
# Other dependencies can be PRIVATE since scheduler's headers don't expose them
target_link_libraries(_dlengine_scheduler
    PRIVATE _dlengine_metrics
    PUBLIC _nanosequence _nanosequence_metrics _dlengine_cache_manager
)

set_target_properties(
    _dlengine_scheduler
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _dlengine_scheduler
    LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH}
)
