# Engine serialization library
set(CPP_SOURCES
    serialization.cpp
)

add_library(
    _nanodeploy_engine
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(_nanodeploy_engine PUBLIC _nanosequence)

target_compile_options(_nanodeploy_engine PRIVATE -O3 -Wall -Wextra)

# Include the engine header directory
target_include_directories(_nanodeploy_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

set_target_properties(
    _nanodeploy_engine
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_engine
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
