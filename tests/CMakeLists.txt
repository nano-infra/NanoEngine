project(NanoDeployTests)

find_package(Spoke REQUIRED)
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
    ${CMAKE_SOURCE_DIR}/../csrc
)
