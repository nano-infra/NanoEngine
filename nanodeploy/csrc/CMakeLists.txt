# NanoDeploy C++ Core and Worker
add_library(nanodeploy_core STATIC
    core/config.h
    core/module.h
    core/weight_mapping.h
)

target_include_directories(nanodeploy_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link with third-party and core dependencies
set_target_properties(nanodeploy_core PROPERTIES LINKER_LANGUAGE CXX)

# Link with third-party and core dependencies
target_link_libraries(nanodeploy_core PUBLIC
    flashinfer
    deep_gemm
    deep_ep
    dlslime::dlslime  # standardized alias
    Spoke::core
    torch
    torch_cpu
    c10
)

add_subdirectory(sequence)
add_subdirectory(metrics)
add_subdirectory(scheduler)
add_subdirectory(context)
add_subdirectory(ops)
add_subdirectory(worker)
add_subdirectory(executor)
add_subdirectory(python)
