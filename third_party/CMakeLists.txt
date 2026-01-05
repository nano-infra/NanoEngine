# NanoDeploy Third-Party Dependencies

# FlashInfer (Header-only)
set(FLASHINFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/flashinfer")
if(EXISTS "${FLASHINFER_ROOT}/include")
    message(STATUS "Found FlashInfer at ${FLASHINFER_ROOT}")
    add_library(flashinfer INTERFACE)
    target_include_directories(flashinfer INTERFACE "${FLASHINFER_ROOT}/include")
else()
    message(WARNING "FlashInfer not found at ${FLASHINFER_ROOT}, please run: git submodule update --init --recursive")
endif()

# DeepGEMM (Headers + Template indexing)
set(DEEPGEMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/DeepGemm")
if(EXISTS "${DEEPGEMM_ROOT}/deep_gemm/include")
    message(STATUS "Found DeepGEMM at ${DEEPGEMM_ROOT}")
    add_library(deep_gemm INTERFACE)
    target_include_directories(deep_gemm INTERFACE
        "${DEEPGEMM_ROOT}"
        "${DEEPGEMM_ROOT}/deep_gemm/include"
        "${DEEPGEMM_ROOT}/third-party/cutlass/include"
        "${DEEPGEMM_ROOT}/third-party/fmt/include"
    )
else()
    message(WARNING "DeepGEMM headers not found at ${DEEPGEMM_ROOT}, please run: git submodule update --init --recursive")
endif()

# DeepEP (Headers + Source)
set(DEEPEP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/DeepEP")
if(EXISTS "${DEEPEP_ROOT}/csrc")
    message(STATUS "Found DeepEP at ${DEEPEP_ROOT}")
    add_library(deep_ep INTERFACE)
    target_include_directories(deep_ep INTERFACE
        "${DEEPEP_ROOT}"
        "${DEEPEP_ROOT}/csrc"
    )
else()
    message(WARNING "DeepEP headers not found at ${DEEPEP_ROOT}, please run: git submodule update --init --recursive")
endif()

# DLSlime (Submodule)
set(DLSLIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/DLSlime")
if(EXISTS "${DLSLIME_ROOT}/CMakeLists.txt")
    message(STATUS "Integrating DLSlime from ${DLSLIME_ROOT}")
    add_subdirectory("${DLSLIME_ROOT}" DLSlime EXCLUDE_FROM_ALL)
else()
    message(WARNING "DLSlime not found at ${DLSLIME_ROOT}, check submodules")
endif()

# Spoke (Submodule)
set(SPOKE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/Spoke")
if(EXISTS "${SPOKE_ROOT}/CMakeLists.txt")
    message(STATUS "Integrating Spoke from ${SPOKE_ROOT}")
    add_subdirectory("${SPOKE_ROOT}" Spoke EXCLUDE_FROM_ALL)
else()
    message(WARNING "Spoke not found at ${SPOKE_ROOT}, check submodules")
endif()

# pybind11 (Submodule)
set(PYBIND11_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/pybind11")
if(EXISTS "${PYBIND11_ROOT}/CMakeLists.txt")
    message(STATUS "Integrating pybind11 from ${PYBIND11_ROOT}")
    add_subdirectory("${PYBIND11_ROOT}" pybind11 EXCLUDE_FROM_ALL)
else()
    message(WARNING "pybind11 not found at ${PYBIND11_ROOT}, please run: git submodule add https://github.com/pybind/pybind11 third_party/pybind11")
endif()
