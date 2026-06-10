# Engine serialization library
set(CPP_SOURCES
    serialization.cpp
)

add_library(
    _dlengine_engine
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_dlengine_engine PUBLIC _nanosequence)

target_compile_options(_dlengine_engine PRIVATE -O3 -Wall -Wextra)

# Include the engine header directory
target_include_directories(_dlengine_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

set_target_properties(
    _dlengine_engine
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _dlengine_engine
    LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH}
)
