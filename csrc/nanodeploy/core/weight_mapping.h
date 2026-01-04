#pragma once

#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <vector>

namespace nanodeploy {
namespace core {

struct WeightMapping {
    std::string internal_name;
    std::string hf_name;
};

class WeightMappingConfig {
    std::vector<WeightMapping> mappings_;

public:
    static WeightMappingConfig load(const std::string& path)
    {
        std::ifstream f(path);
        if (!f.is_open()) {
            throw std::runtime_error("Failed to open weight map: " + path);
        }
        nlohmann::json j = nlohmann::json::parse(f);

        WeightMappingConfig config;
        for (auto it = j.begin(); it != j.end(); ++it) {
            config.mappings_.push_back({it.key(), it.value()});
        }
        return config;
    }

    std::string resolve(const std::string& internal_name, int layer = -1, int expert = -1) const
    {
        for (const auto& m : mappings_) {
            if (m.internal_name == internal_name) {
                std::string res = m.hf_name;
                if (layer != -1) {
                    res = std::regex_replace(res, std::regex("\\{layer\\}"), std::to_string(layer));
                }
                if (expert != -1) {
                    res = std::regex_replace(res, std::regex("\\{expert\\}"), std::to_string(expert));
                }
                return res;
            }
        }
        return internal_name;
    }
};

}  // namespace core
}  // namespace nanodeploy
