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
