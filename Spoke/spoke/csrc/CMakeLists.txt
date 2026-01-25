add_library(spoke
    SHARED
    agent.cpp
    actor.cpp
)

set_target_properties(spoke PROPERTIES EXPORT_NAME core)
add_library(Spoke::core ALIAS spoke)

target_include_directories(spoke PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

# Direct Link to Internal Target (dlslime::dlslime)
# The add_subdirectory above makes this target available
message(STATUS "Spoke: Linking internal DLSlime")
target_link_libraries(spoke PUBLIC dlslime::dlslime pthread)
target_compile_definitions(spoke PUBLIC SPOKE_HAS_RDMA)

install(TARGETS spoke
    EXPORT SpokeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
