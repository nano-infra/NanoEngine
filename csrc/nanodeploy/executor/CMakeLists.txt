include(GNUInstallDirs)
message(${Spoke_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})
include_directories(${Spoke_INCLUDE_DIRS})

set(CPP_SOURCES
    spoke_executor.cpp
)

add_executable(nanodeploy_agent ${CPP_SOURCES})


# Link Torch to nanodeploy_agent
target_link_libraries(nanodeploy_agent PUBLIC _nanodeploy_sequence torch torch_cpu c10 Spoke::daemon ${TORCH_LIBRARIES})

# Set RPATH so it can find lib_nanodeploy_sequence.so in the parent directory relative to bin/
set_target_properties(nanodeploy_agent PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}/.."
)

# Determine installation directory for tools (User requested python bin)
# We can use CMAKE_INSTALL_BINDIR relative to CMAKE_INSTALL_PREFIX which is usually set by pip/setup.py
# But pip install usually installs python scripts to bin, C extensions to site-packages.
# CMAKE_INSTALL_BINDIR is standard for binaries.

install(TARGETS nanodeploy_agent
    DESTINATION ${NANODEPLOY_INSTALL_PATH}/bin
)

# Install and rename spoke_hub to nanodeploy_hub
# Since spoke_hub is an imported target or external tool, we need to find it first.
# Spoke_HUB_EXE variable is provided by SpokeConfig.cmake
if(EXISTS "${Spoke_HUB_EXE}")
    install(PROGRAMS "${Spoke_HUB_EXE}"
        DESTINATION ${NANODEPLOY_INSTALL_PATH}/bin
        RENAME nanodeploy_hub
    )
endif()
