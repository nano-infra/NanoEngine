set(CPP_SOURCES
    sequence.cpp
    serialization.cpp
)

# --- FlatBuffers Generation ---
# Note: FLATC_EXECUTABLE target is available from add_subdirectory(third_party/flatbuffers)
if(TARGET flatc)
    set(FLATC_EXECUTABLE $<TARGET_FILE:flatc>)
else()
    # Fallback if found on system
    find_program(FLATC_EXECUTABLE flatc REQUIRED)
endif()

set(FBS_SOURCE "${CMAKE_SOURCE_DIR}/proto/sequence.fbs")
set(GEN_HEADER_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(GEN_HEADER "${GEN_HEADER_DIR}/sequence_generated.h")

add_custom_command(
    OUTPUT ${GEN_HEADER}
    COMMAND ${FLATC_EXECUTABLE}
            --cpp
            --scoped-enums
            -o ${GEN_HEADER_DIR}
            ${FBS_SOURCE}
    DEPENDS ${FBS_SOURCE}
            flatc
    COMMENT "Generating FlatBuffers header for Sequence"
)

# --- Target Definition ---
add_library(
    _nanodeploy_sequence
    SHARED
    ${CPP_SOURCES}
    ${GEN_HEADER}
)

target_compile_options(_nanodeploy_sequence PRIVATE -O3 -Wall -Wextra)

# Link flatbuffers (header only or static lib)
target_link_libraries(_nanodeploy_sequence PRIVATE flatbuffers nanocommon)

# Include generated header directory
target_include_directories(_nanodeploy_sequence PRIVATE ${GEN_HEADER_DIR})

set_target_properties(
    _nanodeploy_sequence
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_sequence
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
