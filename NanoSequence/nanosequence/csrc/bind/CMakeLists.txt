set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)

set(BINDING_SOURCES
    module.cpp
    metric_binding.cpp
    sequence_binding.cpp
)

# ===== Build pybind11 module =====
pybind11_add_module(_nanosequence_cpp
    ${BINDING_SOURCES}
)

# Define standalone build flag if building standalone
if(NANOSEQUENCE_STANDALONE_BUILD)
    target_compile_definitions(_nanosequence_cpp PRIVATE NANOSEQUENCE_STANDALONE)
endif()

target_link_libraries(_nanosequence_cpp PRIVATE _nanosequence _nanosequence_metrics nanocommon)

set_target_properties(
    _nanosequence_cpp
    PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\${ORIGIN}"
)

# Install to the same location as other NanoSequence libraries
if(DEFINED NANOSEQUENCE_INSTALL_PATH)
    set(NANOSEQUENCE_BIND_INSTALL_PATH ${NANOSEQUENCE_INSTALL_PATH})
else()
    set(NANOSEQUENCE_BIND_INSTALL_PATH nanosequence)
endif()

install(
    TARGETS
    _nanosequence_cpp
    LIBRARY DESTINATION ${NANOSEQUENCE_BIND_INSTALL_PATH}
)
