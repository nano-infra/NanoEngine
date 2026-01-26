cmake_minimum_required(VERSION 3.16)

project(_nanodeploy_cpp C CXX)

# CXX Configuration
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(DLSLIME_INSTALL_PATH "lib" CACHE STRING "Library installation directory")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(csrc)
add_subdirectory(../third_party third_party)
add_subdirectory(../NanoCommon NanoCommon)
add_subdirectory(csrc)

# Python Flatbuffers Generation
set(PROTO_FILE "${CMAKE_CURRENT_SOURCE_DIR}/proto/sequence.fbs")
set(FBS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nanodeploy/fbs")

# Create output directory
file(MAKE_DIRECTORY ${FBS_OUTPUT_DIR})
file(WRITE "${FBS_OUTPUT_DIR}/__init__.py" "")

# Find flatc (target from third_party)
if(NOT TARGET flatc)
    message(FATAL_ERROR "flatc target not found. Ensure third_party/flatbuffers is included.")
endif()

add_custom_command(
    OUTPUT "${FBS_OUTPUT_DIR}/StepOut.py" "${FBS_OUTPUT_DIR}/SequenceStatus.py" "${FBS_OUTPUT_DIR}/Sequence.py"
    COMMAND flatc --python -o "${CMAKE_CURRENT_SOURCE_DIR}" ${PROTO_FILE}
    DEPENDS flatc ${PROTO_FILE}
    COMMENT "Generating Python Flatbuffers from sequence.fbs"
)

# Add a custom target to ensure it runs
add_custom_target(gen_python_fbs ALL DEPENDS "${FBS_OUTPUT_DIR}/StepOut.py")
