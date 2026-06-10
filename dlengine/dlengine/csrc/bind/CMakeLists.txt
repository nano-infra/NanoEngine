set(PYBIND11_FINDPYTHON ON)

# find_package(pybind11 CONFIG REQUIRED)

set(BINDING_SOURCES
    module.cpp
    metric_binding.cpp
    sequence_metric_binding.cpp
    sequence_binding.cpp
    block_manager_binding.cpp
    group_manager_binding.cpp
    scheduler_binding.cpp
    model_runner_binding.cpp
)

# ===== Build pybind11 module =====
pybind11_add_module(_dlengine_cpp
    ${BINDING_SOURCES}
)

target_link_libraries(_dlengine_cpp PRIVATE _dlengine_scheduler _dlengine_cache_manager _dlengine_metrics _nanosequence _nanosequence_metrics _dlengine_worker _dlengine_engine)

set_target_properties(
    _dlengine_cpp
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _dlengine_cpp
    LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH}
)
