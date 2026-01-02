set(CPP_SOURCES
    sequence.cpp
    serialization.cpp
)

add_library(
    _nanodeploy_sequence
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_nanodeploy_sequence PRIVATE -O3 -Wall -Wextra)

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
