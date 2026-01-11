set(CPP_SOURCES
    dummy_runner.cpp
    model_runner_utils.cpp
    model_runner.cpp
    deep_ep_runner.cpp
    deep_gemm_runner.cpp
    distributed.cpp
    ../layers/flashinfer_handler.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/deep_ep.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/internode.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/internode_ll.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/intranode.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/layout.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/runtime.cu
)

find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.__path__[0], end='')"
    OUTPUT_VARIABLE TORCH_ROOT
)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_ROOT}/lib")

set_property(SOURCE ../layers/flashinfer_handler.cu PROPERTY LANGUAGE CUDA)
set_property(SOURCE model_runner.cpp PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/internode.cu PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/internode_ll.cu PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/intranode.cu PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/layout.cu PROPERTY LANGUAGE CUDA)
set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc/kernels/runtime.cu PROPERTY LANGUAGE CUDA)



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
    ${TORCH_PYTHON_LIBRARY}
    pybind11::embed
    Python::Python
    cuda
    nvrtc
    /mnt/nvme1n1/ml_research/majinming/install/lib/libnvshmem_host.so
    /mnt/nvme1n1/ml_research/majinming/install/lib/libnvshmem_device.a
)

set_target_properties(_nanodeploy_worker PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(_nanodeploy_worker PUBLIC
    "/mnt/nvme1n1/ml_research/majinming/install/include"
)
target_include_directories(_nanodeploy_worker PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepGemm/csrc"

    "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/DeepEP/csrc"
)

target_compile_options(_nanodeploy_worker PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
