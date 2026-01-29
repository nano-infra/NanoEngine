#pragma once

#include <string>

namespace dlslime {
namespace rendezvous {

/** bind 地址转为本机 client 连入地址（供本机连本机 broker）。
 *  0.0.0.0 表示 bind 到全部本地网卡，转成 client 时取回环 127.0.0.1；
 *  0.0.0.0:port -> 127.0.0.1:port，0.0.0.0 -> 127.0.0.1，其他原样。与 RDMA/ZMQ 无关。 */
inline std::string client_addr_from_bind(const std::string& bind_addr)
{
    if (bind_addr.compare(0, 8, "0.0.0.0:") == 0)
        return "127.0.0.1:" + bind_addr.substr(8);
    if (bind_addr == "0.0.0.0")
        return "127.0.0.1";
    return bind_addr;
}

/** 从 client_addr（如 "127.0.0.1:50051"）取默认 peer id（端口部分 "50051"）；无冒号则返回原串。 */
inline std::string default_peer_id_from_client_addr(const std::string& client_addr)
{
    size_t colon = client_addr.rfind(':');
    if (colon != std::string::npos && colon + 1 < client_addr.size())
        return client_addr.substr(colon + 1);
    return client_addr.empty() ? "0" : client_addr;
}

}  // namespace rendezvous
}  // namespace dlslime
