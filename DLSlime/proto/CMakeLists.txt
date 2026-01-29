# FlatBuffers generation for DLSlime

set(RDMA_PROTO "${CMAKE_CURRENT_SOURCE_DIR}/rdma.fbs")
set(FBS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

file(MAKE_DIRECTORY ${FBS_OUTPUT_DIR})

# Find flatc (target from third_party)
if(NOT TARGET flatc)
    message(FATAL_ERROR "flatc target not found. Ensure third_party/flatbuffers is included.")
endif()

add_custom_command(
    OUTPUT "${FBS_OUTPUT_DIR}/rdma_generated.h"
    COMMAND flatc --cpp -o "${FBS_OUTPUT_DIR}" ${RDMA_PROTO}
    DEPENDS flatc ${RDMA_PROTO}
    COMMENT "Generating C++ FlatBuffers from rdma.fbs"
)

add_library(dlslime_proto INTERFACE)
target_include_directories(dlslime_proto INTERFACE
    $<BUILD_INTERFACE:${FBS_OUTPUT_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/flatbuffers/include>
    $<INSTALL_INTERFACE:include/dlslime>
)
add_custom_target(gen_dlslime_proto DEPENDS "${FBS_OUTPUT_DIR}/rdma_generated.h")
add_dependencies(dlslime_proto gen_dlslime_proto)
install(
    TARGETS dlslime_proto
    EXPORT dlslimeTargets
    LIBRARY DESTINATION ${DLSLIME_INSTALL_PATH}
    ARCHIVE DESTINATION ${DLSLIME_INSTALL_PATH}
    RUNTIME DESTINATION ${DLSLIME_INSTALL_PATH}
)
