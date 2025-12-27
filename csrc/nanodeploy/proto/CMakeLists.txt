set(CPP_SOURCES
    serialization.cpp
)

add_library(
    _nanodeploy_proto
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_nanodeploy_proto PRIVATE -O3 -Wall -Wextra)

set_target_properties(
    _nanodeploy_proto
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_proto
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
