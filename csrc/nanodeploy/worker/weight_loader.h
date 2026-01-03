#pragma once

#include <fcntl.h>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/mman.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

namespace nanodeploy {

class SafeTensorLoader {
    int                                   fd_;
    size_t                                size_;
    void*                                 addr_;
    nlohmann::json                        metadata_;
    std::map<std::string, nlohmann::json> tensors_;

public:
    SafeTensorLoader(const std::string& path)
    {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            throw std::runtime_error("Failed to open: " + path);

        size_ = lseek(fd_, 0, SEEK_END);
        addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED)
            throw std::runtime_error("mmap failed");

        uint64_t header_size;
        memcpy(&header_size, addr_, sizeof(uint64_t));

        std::string header_str((char*)addr_ + sizeof(uint64_t), header_size);
        metadata_ = nlohmann::json::parse(header_str);

        for (auto it = metadata_.begin(); it != metadata_.end(); ++it) {
            if (it.key() == "__metadata__")
                continue;
            tensors_[it.key()] = it.value();
        }
    }

    ~SafeTensorLoader()
    {
        if (addr_ != MAP_FAILED)
            munmap(addr_, size_);
        if (fd_ >= 0)
            close(fd_);
    }

    torch::Tensor load(const std::string& name, torch::Device device = torch::kCPU)
    {
        if (tensors_.find(name) == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }

        auto                  info         = tensors_[name];
        std::vector<int64_t>  shape        = info["shape"].get<std::vector<int64_t>>();
        std::vector<uint64_t> data_offsets = info["data_offsets"].get<std::vector<uint64_t>>();
        std::string           dtype_str    = info["dtype"];

        torch::ScalarType dtype;
        if (dtype_str == "F16")
            dtype = torch::kFloat16;
        else if (dtype_str == "BF16")
            dtype = torch::kBFloat16;
        else if (dtype_str == "F32")
            dtype = torch::kFloat32;
        else if (dtype_str == "I64")
            dtype = torch::kInt64;
        else
            throw std::runtime_error("Unsupported dtype: " + dtype_str);

        uint64_t header_size;
        memcpy(&header_size, addr_, sizeof(uint64_t));
        void* data_ptr = (char*)addr_ + sizeof(uint64_t) + header_size + data_offsets[0];

        auto          options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        torch::Tensor t       = torch::from_blob(data_ptr, shape, options);

        return t.to(device).clone();  // Clone to own the memory if mmap is released or if moving to GPU
    }
};

}  // namespace nanodeploy
