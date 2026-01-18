#pragma once

#include <fcntl.h>
#include <fstream>
#include <map>
#include <string>
#include <sys/mman.h>
#include <torch/torch.h>
#include <unistd.h>
#include <vector>

#include "nanodeploy/csrc/json.hpp"
#include "nanodeploy/csrc/logging.h"

namespace nanodeploy {

class SafeTensorLoader {
    int                                   fd_;
    size_t                                size_;
    void*                                 addr_;
    uint64_t                              header_size_;  // Cache header size
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

        // Read header size once
        memcpy(&header_size_, addr_, sizeof(uint64_t));

        std::string header_str((char*)addr_ + sizeof(uint64_t), header_size_);
        metadata_ = nlohmann::json::parse(header_str);

        for (auto it = metadata_.begin(); it != metadata_.end(); ++it) {
            if (it.key() == "__metadata__")
                continue;
            tensors_[it.key()] = it.value();
        }
        NANODEPLOY_LOG_DEBUG("[SafeTensorLoader] Constructed. Addr: ", addr_);
    }

    // Move Constructor
    SafeTensorLoader(SafeTensorLoader&& other) noexcept:
        fd_(other.fd_),
        size_(other.size_),
        addr_(other.addr_),
        header_size_(other.header_size_),
        metadata_(std::move(other.metadata_)),
        tensors_(std::move(other.tensors_))
    {
        other.fd_   = -1;
        other.addr_ = MAP_FAILED;
        NANODEPLOY_LOG_DEBUG("[SafeTensorLoader] Moved. Addr: ", addr_);
    }

    // Move Assignment
    SafeTensorLoader& operator=(SafeTensorLoader&& other) noexcept
    {
        if (this != &other) {
            if (addr_ != MAP_FAILED)
                munmap(addr_, size_);
            if (fd_ >= 0)
                close(fd_);

            fd_          = other.fd_;
            size_        = other.size_;
            addr_        = other.addr_;
            header_size_ = other.header_size_;
            metadata_    = std::move(other.metadata_);
            tensors_     = std::move(other.tensors_);

            other.fd_   = -1;
            other.addr_ = MAP_FAILED;
            NANODEPLOY_LOG_DEBUG("[SafeTensorLoader] Move Assigned. Addr: ", addr_);
        }
        return *this;
    }

    // Delete Copy
    SafeTensorLoader(const SafeTensorLoader&)            = delete;
    SafeTensorLoader& operator=(const SafeTensorLoader&) = delete;

    ~SafeTensorLoader()
    {
        if (addr_ != MAP_FAILED) {
            NANODEPLOY_LOG_DEBUG("[SafeTensorLoader] Destructing... Unmapping addr: ", addr_);
            munmap(addr_, size_);
        }
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

        NANODEPLOY_LOG_DEBUG("[SafeTensorLoader] info: ", name, " | Dtype: ", dtype_str);

        torch::ScalarType dtype;
        int               element_size = 0;
        if (dtype_str == "F16") {
            dtype        = torch::kFloat16;
            element_size = 2;
        }
        else if (dtype_str == "BF16") {
            dtype        = torch::kBFloat16;
            element_size = 2;
        }
        else if (dtype_str == "F32") {
            dtype        = torch::kFloat32;
            element_size = 4;
        }
        else if (dtype_str == "I64") {
            dtype        = torch::kInt64;
            element_size = 8;
        }
        else if (dtype_str == "F8_E4M3") {
            dtype        = torch::kFloat8_e4m3fn;
            element_size = 1;
        }
        else
            throw std::runtime_error("Unsupported dtype: " + dtype_str);

        // Calculate pointers
        size_t data_start_offset     = sizeof(uint64_t) + header_size_;
        size_t tensor_offset_in_file = data_start_offset + data_offsets[0];

        // Calculate tensor size
        size_t num_elements = 1;
        for (auto s : shape)
            num_elements *= s;
        size_t tensor_bytes = num_elements * element_size;

        // Debug Prints
        NANODEPLOY_LOG_DEBUG("      Base Addr: ", addr_);
        NANODEPLOY_LOG_DEBUG("      Header Size: ", header_size_);
        NANODEPLOY_LOG_DEBUG("      Data Start Offset: ", data_start_offset);
        NANODEPLOY_LOG_DEBUG("      Tensor Offset (rel to data): ", data_offsets[0]);
        NANODEPLOY_LOG_DEBUG("      Final File Offset: ", tensor_offset_in_file);
        NANODEPLOY_LOG_DEBUG("      Tensor Bytes: ", tensor_bytes);
        NANODEPLOY_LOG_DEBUG("      Total File Size: ", size_);

        if (tensor_offset_in_file + tensor_bytes > size_) {
            NANODEPLOY_LOG_ERROR("CRITICAL ERROR: OOB Access! ", (tensor_offset_in_file + tensor_bytes), " > ", size_);
            throw std::runtime_error("Tensor OOB");
        }

        void* data_ptr = (char*)addr_ + tensor_offset_in_file;

        NANODEPLOY_LOG_DEBUG("      Creating from blob at: ", data_ptr);
        auto          options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
        torch::Tensor t       = torch::from_blob(data_ptr, shape, options);

        return t.to(device);
    }
};

}  // namespace nanodeploy
