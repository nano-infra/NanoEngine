set(CPP_SOURCES
    dummy_runner.cpp
    model_runner_utils.cpp
    model_runner.cpp
    ../layers/flashinfer_handler.cu
)

set_property(SOURCE ../layers/flashinfer_handler.cu PROPERTY LANGUAGE CUDA)


add_library(
    _nanodeploy_worker
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_nanodeploy_worker
    PUBLIC
    _nanodeploy_sequence
    nanodeploy_core
    torch
    torch_cpu
    c10
)

target_compile_options(_nanodeploy_worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
target_compile_features(_nanodeploy_worker PRIVATE cxx_std_17)

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
