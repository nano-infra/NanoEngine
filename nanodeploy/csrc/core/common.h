#pragma once
#include <torch/torch.h>

namespace nanodeploy {

enum class QuantType {
    FP16,
    BF16,
    FP8_E4M3,
    W8A8
};

}  // namespace nanodeploy
