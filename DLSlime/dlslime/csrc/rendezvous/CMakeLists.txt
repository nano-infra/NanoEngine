# 与 RDMA 无关的 rendezvous 通用工具（如 client_addr_from_bind）
add_library(dlslime_rendezvous INTERFACE)
target_include_directories(dlslime_rendezvous INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:include/dlslime>
)
install(TARGETS dlslime_rendezvous EXPORT dlslimeTargets)
install(FILES client_addr.h DESTINATION ${DLSLIME_INSTALL_PATH}/../include/dlslime/rendezvous)
