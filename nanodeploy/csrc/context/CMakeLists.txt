add_library(_nanodeploy_context STATIC
    distributed_context.cpp
    deep_ep_context.cpp
    attention_context.cpp
    deep_gemm_context.cpp
)

find_package(Python COMPONENTS Interpreter Development.Embed REQUIRED)

target_link_libraries(_nanodeploy_context PUBLIC
    nanodeploy_core
    Python::Python
    pybind11::embed
    # DeepEP, DeepGemm, FlashInfer are linked via nanodeploy_core or explicit here
    # nanodeploy_core links them publicly, so we should be fine if we link nanodeploy_core
)

# If DEEPSEEK_MOE is enabled, we need CUDA
if (NANODEPLOY_BUILD_CUDA)
    target_compile_definitions(_nanodeploy_context PUBLIC NANODEPLOY_BUILD_CUDA)
    if (DEEPSEEK_MOE)
        target_compile_definitions(_nanodeploy_context PUBLIC DEEPSEEK_MOE)
    endif()
endif()

target_include_directories(_nanodeploy_context PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# NVSHMEM include path (matching generic worker setup)
target_include_directories(_nanodeploy_context PUBLIC
    "/mnt/nvme1n1/ml_research/majinming/install/include"
)
