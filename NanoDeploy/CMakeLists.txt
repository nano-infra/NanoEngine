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

# Define NanoInfra root path
if(NOT DEFINED NANOINFRA_DIR)
    set(NANOINFRA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
endif()
message(STATUS "NanoInfra Root Path: ${NANOINFRA_DIR}")

include_directories(.)
add_subdirectory(${NANOINFRA_DIR}/third_party third_party)
add_subdirectory(nanodeploy/csrc)

# Python Flatbuffers Generation
set(SEQUENCE_PROTO "${NANOINFRA_DIR}/NanoSequence/proto/sequence.fbs")
set(PACKET_PROTO "${NANOINFRA_DIR}/NanoSequence/proto/packet.fbs")
set(INTERFACE_PROTO "${NANOINFRA_DIR}/NanoSequence/proto/interface.fbs")
set(FBS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nanodeploy/fbs")

# Find flatc (target from third_party)
if(NOT TARGET flatc)
    message(FATAL_ERROR "flatc target not found. Ensure third_party/flatbuffers is included.")
endif()

# List all expected generated Python files from sequence.fbs + packet.fbs + interface.fbs
set(FBS_GENERATED_FILES
    # From sequence.fbs
    "${FBS_OUTPUT_DIR}/SequenceStatus.py"
    "${FBS_OUTPUT_DIR}/BlockLocation.py"
    "${FBS_OUTPUT_DIR}/IntList.py"
    "${FBS_OUTPUT_DIR}/BlockContext.py"
    "${FBS_OUTPUT_DIR}/SamplingParams.py"
    "${FBS_OUTPUT_DIR}/Sequence.py"
    "${FBS_OUTPUT_DIR}/SequenceList.py"
    "${FBS_OUTPUT_DIR}/StepOut.py"
    "${FBS_OUTPUT_DIR}/FreeSequences.py"
    # From packet.fbs
    "${FBS_OUTPUT_DIR}/Action.py"
    "${FBS_OUTPUT_DIR}/ZmqPacket.py"
    # From interface.fbs
    "${FBS_OUTPUT_DIR}/IntListI.py"
    "${FBS_OUTPUT_DIR}/CompressedBlockTableI.py"
    "${FBS_OUTPUT_DIR}/VisionSlotRef.py"
    "${FBS_OUTPUT_DIR}/SequenceInput.py"
    "${FBS_OUTPUT_DIR}/RunBatchInput.py"
    "${FBS_OUTPUT_DIR}/RunSequenceOutput.py"
    "${FBS_OUTPUT_DIR}/RunBatchOutput.py"
    "${FBS_OUTPUT_DIR}/BlockLocationPair.py"
    "${FBS_OUTPUT_DIR}/MigrateSequenceInput.py"
    "${FBS_OUTPUT_DIR}/MigrateBatchInput.py"
    "${FBS_OUTPUT_DIR}/__init__.py"
)

# flatc --python with namespace nanodeploy.fbs writes nanodeploy/__init__.py under -o.
# Generate into a build-dir staging tree and copy only nanodeploy/fbs/ into the source
# package so the real nanodeploy/__init__.py is never overwritten (no .fbs or import churn).
set(FBS_PYTHON_STAGING "${CMAKE_CURRENT_BINARY_DIR}/fbs_python_staging")

add_custom_command(
    OUTPUT ${FBS_GENERATED_FILES}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${FBS_PYTHON_STAGING}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${FBS_PYTHON_STAGING}"
    COMMAND $<TARGET_FILE:flatc> --python --gen-object-api -o "${FBS_PYTHON_STAGING}" ${SEQUENCE_PROTO}
    COMMAND $<TARGET_FILE:flatc> --python --gen-object-api -o "${FBS_PYTHON_STAGING}" ${PACKET_PROTO}
    COMMAND $<TARGET_FILE:flatc> --python --gen-object-api -o "${FBS_PYTHON_STAGING}" ${INTERFACE_PROTO}
    COMMAND ${CMAKE_COMMAND} -E remove_directory "${FBS_OUTPUT_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${FBS_PYTHON_STAGING}/nanodeploy/fbs" "${FBS_OUTPUT_DIR}"
    DEPENDS flatc ${SEQUENCE_PROTO} ${PACKET_PROTO} ${INTERFACE_PROTO}
    COMMENT "Generating Python FlatBuffers bindings from sequence.fbs, packet.fbs, and interface.fbs"
)

# Create a custom target that depends on the generated files
add_custom_target(generate_flatbuffers ALL DEPENDS ${FBS_GENERATED_FILES})
