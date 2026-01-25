add_executable(spoke_hub hub_main.cpp)

set_target_properties(spoke_hub PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools"
    EXPORT_NAME hub
)

add_executable(Spoke::hub ALIAS spoke_hub)

target_link_libraries(spoke_hub PUBLIC spoke)

install(TARGETS spoke_hub
    EXPORT SpokeTargets
    RUNTIME DESTINATION bin
)
