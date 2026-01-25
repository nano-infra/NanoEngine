add_library(spoke_daemon SHARED daemon.cpp daemon_main.cpp)
target_link_libraries(spoke_daemon PUBLIC spoke)

# Standalone agent binary
add_executable(spoke_agent daemon_main.cpp)
target_link_libraries(spoke_agent PRIVATE spoke_daemon)

set_target_properties(spoke_agent PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(spoke_daemon PROPERTIES EXPORT_NAME daemon)
add_library(Spoke::daemon ALIAS spoke_daemon)

target_include_directories(spoke_daemon PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
)

install(TARGETS spoke_daemon spoke_agent
    EXPORT SpokeTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
