find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)

add_library(_nanodeploy_ops STATIC
    deep_ep_ops.cpp
    deep_gemm_ops.cpp
    moe_expert_ops.cpp
    flashinfer_ops.cu
)

set_property(SOURCE flashinfer_ops.cu PROPERTY LANGUAGE CUDA)

target_include_directories(_nanodeploy_ops PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepGemm/csrc"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc"
    "/mnt/nvme1n1/ml_research/majinming/install/include"
)

target_link_libraries(_nanodeploy_ops PUBLIC
    _nanodeploy_context
    nanodeploy_core
    Python::Python
    pybind11::embed
)

target_compile_options(_nanodeploy_ops PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>)
