#pragma once

#include "nanocommon/json.hpp"

using json = nlohmann::json;

namespace dlslime {

class Buffer {
    void sync(const std::vector<int>& device_ids, json buffer_info);
}
