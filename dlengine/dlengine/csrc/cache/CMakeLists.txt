set(CPP_SOURCES
    block.cpp
    block_manager.cpp
    gdn_state_manager.cpp
    compressed_block_manager.cpp
)

add_library(
    _dlengine_cache_manager
    SHARED
    ${CPP_SOURCES}
)

target_compile_options(_dlengine_cache_manager PRIVATE -O3 -Wall -Wextra)

set_target_properties(_dlengine_cache_manager PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

target_link_libraries(_dlengine_cache_manager PUBLIC _nanosequence)

# PUBLIC include: DLEngine root so consumers use "dlengine/csrc/cache_manager/*.h"
target_include_directories(_dlengine_cache_manager PUBLIC
    $<BUILD_INTERFACE:${NANOINFRA_DIR}/DLEngine>
    $<INSTALL_INTERFACE:include>
)

if(DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(_dlengine_cache_manager PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    )
endif()

install(
    TARGETS _dlengine_cache_manager
    LIBRARY DESTINATION ${DLENGINE_INSTALL_PATH}
)
