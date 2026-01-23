project(NanoDeployTests)

# Spoke is already linked via third_party/Spoke submodule
# find_package(Spoke REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

add_executable(test_dummy_runner test_dummy_runner.cpp)

# Link against NanoDeploy's libraries if needed, or just include headers
# For this test, we act as a Spoke client, so we primarily need Spoke::core
target_link_libraries(test_dummy_runner
    PRIVATE
    Spoke::core
    torch
    torch_cpu
    c10
    _nanodeploy_sequence
)

# Include directories for nanodeploy headers (e.g. for serialization, sequence)
target_include_directories(test_dummy_runner
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)

add_executable(test_model_runner test_model_runner.cpp)
target_link_libraries(test_model_runner
    PRIVATE
    Spoke::core
    torch
    torch_cpu
    c10
    _nanodeploy_sequence
)
target_include_directories(test_model_runner
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)


add_executable(test_slot_mapping test_slot_mapping.cpp)
target_link_libraries(test_slot_mapping
    PRIVATE
    Spoke::core
    torch
    torch_cpu
    c10
    _nanodeploy_sequence
    GTest::gtest_main
)
target_include_directories(test_slot_mapping
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)





if (NANODEPLOY_BUILD_CUDA)
add_executable(test_flashinfer_ops test_flashinfer_ops.cpp)
target_link_libraries(test_flashinfer_ops
    PUBLIC
    _nanodeploy_ops
    nanodeploy_core
    torch
    torch_cpu
    c10
    GTest::gtest_main
)
target_include_directories(test_flashinfer_ops
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
)

# CUDA Graph Compatibility Test (GTest)
add_executable(test_cuda_graph_compat cuda_graph_compat_test.cpp)
target_link_libraries(test_cuda_graph_compat
    PUBLIC
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
    GTest::gtest_main
)
target_include_directories(test_cuda_graph_compat
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
)

# FlashInfer + MoE CUDA Graph Test (GTest)
add_executable(test_cuda_graph_flashinfer_moe cuda_graph_flashinfer_moe_test.cpp)
target_link_libraries(test_cuda_graph_flashinfer_moe
    PUBLIC
    _nanodeploy_worker
    _nanodeploy_ops
    nanodeploy_core
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
    GTest::gtest_main
)
target_include_directories(test_cuda_graph_flashinfer_moe
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/FlashInfer/include
)
endif()

if (NANODEPLOY_BUILD_CUDA)
add_executable(test_rope_ops test_rope_ops.cpp)
target_link_libraries(test_rope_ops
    PRIVATE
    nanodeploy_core
    torch
    torch_cpu
    c10
)
target_include_directories(test_rope_ops
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
)

# DeepGEMM FP8 Test (GTest)
add_executable(test_deepgemm_fp8 test_deepgemm_fp8.cpp)
target_link_libraries(test_deepgemm_fp8
    PUBLIC
    _nanodeploy_ops
    nanodeploy_core
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
    CUDA::nvrtc
    CUDA::cuda_driver
    GTest::gtest_main
)
target_include_directories(test_deepgemm_fp8
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
)

# MoeExpertOps Test (GTest)
add_executable(test_moe_expert_ops test_moe_expert_ops.cpp)
target_link_libraries(test_moe_expert_ops
    PUBLIC
    _nanodeploy_ops
    nanodeploy_core
    torch
    torch_cpu
    torch_cuda
    c10
    c10_cuda
    GTest::gtest_main
    CUDA::nvrtc
    CUDA::cuda_driver
)
target_include_directories(test_moe_expert_ops
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../nanodeploy/csrc
)
endif()
