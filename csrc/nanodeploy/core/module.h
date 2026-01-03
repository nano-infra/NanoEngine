#pragma once

#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace nanodeploy {

struct WeightSpec {
    std::string          name;
    std::vector<int64_t> shape;
    torch::ScalarType    dtype;
};

class Module {
public:
    virtual ~Module() = default;

    // Return the weight specifications for this module
    virtual std::vector<WeightSpec> weight_specs(const std::string& prefix) const = 0;

    // Set weights from a map
    virtual void set_weights(const std::map<std::string, torch::Tensor>& weights, const std::string& prefix) = 0;
};

}  // namespace nanodeploy
