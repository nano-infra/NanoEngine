set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

# NANOINFRA_DIR should be defined by parent CMakeLists.txt
if(NOT DEFINED NANOINFRA_DIR)
    # Fallback: assume we're in NanoInfra/NanoDeploy/nanodeploy/csrc/bind
    set(NANOINFRA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
endif()

set(BINDING_SOURCES
    module.cpp
    metric_binding.cpp
    block_manager_binding.cpp
    sp_state_manager_binding.cpp
    scheduler_binding.cpp
    model_runner_binding.cpp
    # Sequence bindings from NanoSequence
    ${NANOINFRA_DIR}/NanoSequence/nanosequence/csrc/bind/sequence_binding.cpp
    ${NANOINFRA_DIR}/NanoSequence/nanosequence/csrc/bind/metric_binding.cpp
)

# ===== Build pybind11 module =====
pybind11_add_module(_nanodeploy_cpp
    ${BINDING_SOURCES}
)

target_link_libraries(_nanodeploy_cpp PRIVATE _nanodeploy_scheduler _nanodeploy_metrics _nanosequence _nanosequence_metrics _nanodeploy_worker _nanodeploy_engine nanocommon)

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
