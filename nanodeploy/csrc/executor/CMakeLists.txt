include(GNUInstallDirs)
message(STATUS "Spoke Includes: ${Spoke_INCLUDE_DIRS}")
include_directories(${Torch_INCLUDE_DIRS})
include_directories(${Spoke_INCLUDE_DIRS})

set(CPP_SOURCES
    spoke_executor.cpp
)

add_executable(nanodeploy_agent ${CPP_SOURCES})


# Link Torch to nanodeploy_agent
target_link_libraries(nanodeploy_agent PUBLIC _nanodeploy_worker _nanodeploy_sequence _nanodeploy_scheduler torch torch_cpu c10 Spoke::daemon ${TORCH_LIBRARIES})

# Set RPATH so it can find lib_nanodeploy_sequence.so in the parent directory relative to bin/
# Also find PyTorch library path to fix MKL loading issues
set(_torch_rpath "")
if(TARGET torch_cpu)
    get_target_property(_torch_lib_loc torch_cpu IMPORTED_LOCATION)
    if(NOT _torch_lib_loc)
         get_target_property(_torch_lib_loc torch_cpu IMPORTED_LOCATION_RELEASE)
    endif()

    if(_torch_lib_loc)
        get_filename_component(_torch_lib_dir "${_torch_lib_loc}" DIRECTORY)
        set(_torch_rpath "${_torch_lib_dir}")
        message(STATUS "Appending Torch Lib Dir to RPATH: ${_torch_lib_dir}")
    endif()
endif()

set_target_properties(nanodeploy_agent PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}/..:${_torch_rpath}"
)

# Determine installation directory for tools (User requested python bin)
# We can use CMAKE_INSTALL_BINDIR relative to CMAKE_INSTALL_PREFIX which is usually set by pip/setup.py
# But pip install usually installs python scripts to bin, C extensions to site-packages.
# CMAKE_INSTALL_BINDIR is standard for binaries.

install(TARGETS nanodeploy_agent
    DESTINATION ${NANODEPLOY_INSTALL_PATH}/bin
)

if(TARGET Spoke::hub)
    # 1. Resolve alias if necessary (e.g., local build)
    set(_hub_target Spoke::hub)
    get_target_property(_aliased Spoke::hub ALIASED_TARGET)
    if(_aliased)
        set(_hub_target ${_aliased})
    endif()

    # 2. If it's a local build target (not imported), allow add_dependencies
    get_target_property(_is_imported ${_hub_target} IMPORTED)
    if(NOT _is_imported)
        # Force it to build alongside nanodeploy_agent
        add_dependencies(nanodeploy_agent ${_hub_target})
    endif()

    # 3. Install as a PROGRAM (supports RENAME and finding TARGET_FILE)
    install(PROGRAMS $<TARGET_FILE:Spoke::hub>
        DESTINATION ${NANODEPLOY_INSTALL_PATH}/bin
        RENAME nanodeploy_hub
    )
endif()
