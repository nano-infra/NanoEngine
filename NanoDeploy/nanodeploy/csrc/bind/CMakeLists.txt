set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

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
pybind11_add_module(_nanodeploy_cpp
    ${BINDING_SOURCES}
)

target_link_libraries(_nanodeploy_cpp PRIVATE _nanodeploy_scheduler _nanodeploy_cache_manager _nanodeploy_metrics _nanosequence _nanosequence_metrics _nanodeploy_worker _nanodeploy_engine nanocommon)

set_target_properties(
    _nanodeploy_cpp
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_cpp
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
