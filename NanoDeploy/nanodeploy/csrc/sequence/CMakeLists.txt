set(CPP_SOURCES
    sequence.cpp
    serialization.cpp
)

if(TARGET flatc)
    set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
else()
    find_program(FLATC_EXECUTABLE flatc REQUIRED)
endif()

set(FBS_SOURCE "${NANOINFRA_DIR}/NanoSequence/proto/sequence.fbs")
set(GEN_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(GEN_HEADER "${GEN_HEADER_DIR}/sequence_generated.h")

add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --scoped-enums
            --gen-object-api
            -o ${GEN_HEADER_DIR}
            ${FBS_SOURCE}
    DEPENDS ${FBS_SOURCE}
            flatc
    COMMENT "Generating FlatBuffers header for Sequence"
)

set(INTERFACE_FBS_SOURCE "${NANOINFRA_DIR}/NanoSequence/proto/interface.fbs")
set(INTERFACE_GEN_HEADER "${GEN_HEADER_DIR}/interface_generated.h")

add_custom_command(
    OUTPUT ${INTERFACE_GEN_HEADER}
    COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --scoped-enums
            --gen-object-api
            -o ${GEN_HEADER_DIR}
            ${INTERFACE_FBS_SOURCE}
    DEPENDS ${INTERFACE_FBS_SOURCE}
            flatc
    COMMENT "Generating FlatBuffers header for interface (engine-runner protocol)"
)

add_library(
    _nanosequence
    SHARED
    ${CPP_SOURCES}
    ${GEN_HEADER}
    ${INTERFACE_GEN_HEADER}
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_nanosequence PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

set_target_properties(_nanosequence PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

target_compile_options(_nanosequence PRIVATE -O3 -Wall -Wextra)

target_link_libraries(_nanosequence PUBLIC flatbuffers PRIVATE nanocommon _nanosequence_metrics)

target_include_directories(_nanosequence PRIVATE ${GEN_HEADER_DIR})

# PUBLIC include: NanoDeploy root so consumers can use "nanodeploy/csrc/sequence/sequence.h"
# Also expose generated headers and flatbuffers includes
target_include_directories(_nanosequence PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/NanoDeploy>
    $<BUILD_INTERFACE:${GEN_HEADER_DIR}>
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/third_party/flatbuffers/include>
    $<INSTALL_INTERFACE:include>
)

install(
    TARGETS _nanosequence
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
