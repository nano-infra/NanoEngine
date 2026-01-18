#pragma once

#include <map>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace nanodeploy {
namespace core {

struct WeightSpec {
    std::string          name;
    std::vector<int64_t> shape;
    torch::ScalarType    dtype;
};

class Module {
public:
    virtual ~Module() = default;

    virtual std::vector<WeightSpec> weight_specs(const std::string& /*prefix*/) const
    {
        return {};
    }

    virtual void set_weights(const std::map<std::string, torch::Tensor>& /*weights*/, const std::string& /*prefix*/)
    {
        // Default implementation does nothing or warns
    }

    torch::Tensor register_parameter(const std::string& /*name*/, torch::Tensor tensor)
    {
        // Placeholder registry
        return tensor;
    }

    torch::Tensor register_buffer(const std::string& /*name*/, torch::Tensor tensor)
    {
        return tensor;
    }

    // Generic register_module to match PyTorch API in simple cases
    template<typename T>
    T register_module(const std::string& /*name*/, T module)
    {
        // In a full implementation, we'd store this for traversal.
        // For now, just pass through.
        return module;
    }
};

}  // namespace core
}  // namespace nanodeploy
