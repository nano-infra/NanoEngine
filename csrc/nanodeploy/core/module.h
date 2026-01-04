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

    virtual std::vector<WeightSpec> weight_specs(const std::string& prefix) const
    {
        return {};
    }

    virtual void set_weights(const std::map<std::string, torch::Tensor>& weights, const std::string& prefix)
    {
        // Default implementation does nothing or warns
    }

    // Helper for registering parameters (if not already existing in PyTorch Module style)
    // Since we are likely implementing proper weight management later, let's add a basic registry map
    // or just rely on manual member management for now.
    // BUT WAIT: Linear/RMSNorm are calling register_parameter. Where is that defined?
    // It must be in this class!

    torch::Tensor register_parameter(const std::string& name, torch::Tensor tensor)
    {
        // Placeholder registry
        return tensor;
    }

    torch::Tensor register_buffer(const std::string& name, torch::Tensor tensor)
    {
        return tensor;
    }
};

}  // namespace core
}  // namespace nanodeploy
