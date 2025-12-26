set(CPP_SOURCES
    rpc_endpoint.cpp
)

add_library(
    _nanodeploy_rpc
    SHARED
    ${CPP_SOURCES}
)

target_link_libraries(
    _nanodeploy_rpc
    ${slime_LIBRARIES}
)

target_compile_options(_nanodeploy_rpc PRIVATE -O3 -Wall -Wextra)

set_target_properties(
    _nanodeploy_rpc
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

install(
    TARGETS
    _nanodeploy_rpc
    LIBRARY DESTINATION ${NANODEPLOY_INSTALL_PATH}
)
