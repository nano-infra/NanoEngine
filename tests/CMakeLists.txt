project(NanoDeployTests)

# Spoke is already linked via third_party/Spoke submodule
# find_package(Spoke REQUIRED)
find_package(Torch REQUIRED)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)

add_executable(test_qwen3_runner test_qwen3_runner.cpp)
target_link_libraries(test_qwen3_runner
    PUBLIC
    _nanodeploy_worker
    _nanodeploy_sequence
    _nanodeploy_scheduler
    nanodeploy_core
    torch
    torch_cpu
    c10
    Spoke::core
)
target_include_directories(test_qwen3_runner
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
)

add_executable(test_flashinfer_ops test_flashinfer_ops.cpp)
target_link_libraries(test_flashinfer_ops
    PRIVATE
    _nanodeploy_worker
    _nanodeploy_sequence
    nanodeploy_core
    torch
    torch_cpu
    c10
)
target_include_directories(test_flashinfer_ops
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
)

add_executable(test_deep_ep_actor test_deep_ep_actor.cpp)
target_link_libraries(test_deep_ep_actor
    PRIVATE
    Spoke::core
    torch
    torch_cpu
    c10
    _nanodeploy_sequence
)
target_include_directories(test_deep_ep_actor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)

add_executable(test_deep_gemm_actor test_deep_gemm_actor.cpp)
target_link_libraries(test_deep_gemm_actor
    PRIVATE
    Spoke::core
    torch
    torch_cpu
    c10
    _nanodeploy_sequence
)
target_include_directories(test_deep_gemm_actor
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/Spoke
)

add_executable(test_qwen3_moe_runner test_qwen3_moe_runner.cpp)
target_link_libraries(test_qwen3_moe_runner
    PUBLIC
    _nanodeploy_worker
    _nanodeploy_sequence
    _nanodeploy_scheduler
    nanodeploy_core
    torch
    torch_cpu
    c10
    Spoke::core
)
target_include_directories(test_qwen3_moe_runner
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../csrc
)
