#pragma once

#include "sequence_core.h"

#include <cstdint>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// -------------------------------------------------------------------------
// Minimal XXH64 implementation (streaming) written from the public algorithm
// description. This is used to match Python's xxhash.xxh64().intdigest().
// -------------------------------------------------------------------------

namespace nanodeploy_hash {

static inline uint64_t rotl64(uint64_t x, int r)
{
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t read64le(const uint8_t* p)
{
    return (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24)
           | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
}

static inline uint32_t read32le(const uint8_t* p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline uint64_t round64(uint64_t acc, uint64_t input)
{
    // XXH64 round: acc += input * PRIME2; acc = rotl(acc,31); acc *= PRIME1
    constexpr uint64_t PRIME1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME2 = 14029467366897019727ULL;

    acc += input * PRIME2;
    acc = rotl64(acc, 31);
    acc *= PRIME1;
    return acc;
}

static inline uint64_t merge_round64(uint64_t acc, uint64_t val)
{
    constexpr uint64_t PRIME1 = 11400714785074694791ULL;
    constexpr uint64_t PRIME4 = 9650029242287828579ULL;

    acc ^= round64(0, val);
    acc = acc * PRIME1 + PRIME4;
    return acc;
}

static inline uint64_t avalanche64(uint64_t h)
{
    constexpr uint64_t PRIME2 = 14029467366897019727ULL;
    constexpr uint64_t PRIME3 = 1609587929392839161ULL;

    h ^= h >> 33;
    h *= PRIME2;
    h ^= h >> 29;
    h *= PRIME3;
    h ^= h >> 32;
    return h;
}

class XXH64 {
public:
    explicit XXH64(uint64_t seed = 0): seed_(seed)
    {
        reset(seed);
    }

    void reset(uint64_t seed)
    {
        seed_ = seed;
        total_len_ = 0;
        mem_size_ = 0;

        constexpr uint64_t PRIME1 = 11400714785074694791ULL;
        constexpr uint64_t PRIME2 = 14029467366897019727ULL;

        v1_ = seed_ + PRIME1 + PRIME2;
        v2_ = seed_ + PRIME2;
        v3_ = seed_ + 0;
        v4_ = seed_ - PRIME1;
    }

    void update(const void* data, size_t len)
    {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        total_len_ += len;

        // Fill existing partial buffer to 32 bytes
        if (mem_size_ + len < 32) {
            std::memcpy(mem_ + mem_size_, p, len);
            mem_size_ += (uint32_t)len;
            return;
        }

        size_t idx = 0;
        if (mem_size_ > 0) {
            size_t fill = 32 - mem_size_;
            std::memcpy(mem_ + mem_size_, p, fill);
            consume32(mem_);
            idx += fill;
            mem_size_ = 0;
        }

        // Process stripes
        const uint8_t* p2 = p + idx;
        const uint8_t* end = p + len;
        while (p2 + 32 <= end) {
            consume32(p2);
            p2 += 32;
        }

        // Buffer remaining
        size_t remaining = (size_t)(end - p2);
        if (remaining > 0) {
            std::memcpy(mem_, p2, remaining);
            mem_size_ = (uint32_t)remaining;
        }
    }

    uint64_t digest() const
    {
        constexpr uint64_t PRIME1 = 11400714785074694791ULL;
        constexpr uint64_t PRIME2 = 14029467366897019727ULL;
        constexpr uint64_t PRIME3 = 1609587929392839161ULL;
        constexpr uint64_t PRIME4 = 9650029242287828579ULL;
        constexpr uint64_t PRIME5 = 2870177450012600261ULL;

        uint64_t h;

        if (total_len_ >= 32) {
            h = rotl64(v1_, 1) + rotl64(v2_, 7) + rotl64(v3_, 12) + rotl64(v4_, 18);
            h = merge_round64(h, v1_);
            h = merge_round64(h, v2_);
            h = merge_round64(h, v3_);
            h = merge_round64(h, v4_);
        }
        else {
            h = seed_ + PRIME5;
        }

        h += (uint64_t)total_len_;

        const uint8_t* p = mem_;
        const uint8_t* end = mem_ + mem_size_;

        while (p + 8 <= end) {
            uint64_t k1 = round64(0, read64le(p));
            h ^= k1;
            h = rotl64(h, 27) * PRIME1 + PRIME4;
            p += 8;
        }

        while (p + 4 <= end) {
            h ^= (uint64_t)read32le(p) * PRIME1;
            h = rotl64(h, 23) * PRIME2 + PRIME3;
            p += 4;
        }

        while (p < end) {
            h ^= (uint64_t)(*p) * PRIME5;
            h = rotl64(h, 11) * PRIME1;
            ++p;
        }

        return avalanche64(h);
    }

private:
    void consume32(const uint8_t* p)
    {
        v1_ = round64(v1_, read64le(p + 0));
        v2_ = round64(v2_, read64le(p + 8));
        v3_ = round64(v3_, read64le(p + 16));
        v4_ = round64(v4_, read64le(p + 24));
    }

    uint64_t seed_ = 0;
    uint64_t v1_ = 0;
    uint64_t v2_ = 0;
    uint64_t v3_ = 0;
    uint64_t v4_ = 0;

    uint8_t  mem_[32] = {0};
    uint32_t mem_size_ = 0;
    size_t   total_len_ = 0;
};

} // namespace nanodeploy_hash

// -------------------------------------------------------------------------
// Block + BlockManager (C++)
// -------------------------------------------------------------------------

struct NDCacheBlock {
    int                      block_id = -1;
    int                      ref_count = 0;
    bool                     has_hash  = false;
    uint64_t                 hash      = 0;
    std::vector<int>         token_ids;

    explicit NDCacheBlock(int id = -1): block_id(id) {}

    void reset()
    {
        ref_count = 1;
        has_hash  = false;
        hash      = 0;
        token_ids.clear();
    }

    void update(uint64_t h, std::vector<int> tokens)
    {
        has_hash = true;
        hash     = h;
        token_ids = std::move(tokens);
    }

    py::object py_hash() const
    {
        if (!has_hash)
            return py::int_(-1);
        return py::int_(hash);
    }
};

class NDCacheBlockManager {
public:
    NDCacheBlockManager(std::optional<std::string> engine_id, int sp_idx, int num_blocks, int block_size):
        engine_id_(std::move(engine_id)), sp_idx_(sp_idx), block_size_(block_size)
    {
        if (num_blocks <= 0)
            throw std::invalid_argument("num_blocks must be > 0");
        if (block_size_ <= 0)
            throw std::invalid_argument("block_size must be > 0");

        blocks_.reserve((size_t)num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            blocks_.emplace_back(i);
            free_block_ids_.push_back(i);
        }
    }

    static uint64_t compute_hash(const std::vector<int>& token_ids, std::optional<uint64_t> prefix)
    {
        nanodeploy_hash::XXH64 h;
        if (prefix.has_value()) {
            uint64_t p = prefix.value();
            uint8_t  bytes[8];
            for (int i = 0; i < 8; ++i) {
                bytes[i] = (uint8_t)((p >> (i * 8)) & 0xFF);
            }
            h.update(bytes, 8);
        }

        // Python: np.array(token_ids).tobytes() -> int64 bytes on 64-bit
        // We must match that layout (little-endian 8 bytes per element).
        std::vector<uint64_t> tmp;
        tmp.reserve(token_ids.size());
        for (int t : token_ids) {
            tmp.push_back((uint64_t)(int64_t)t);
        }
        if (!tmp.empty()) {
            h.update(tmp.data(), tmp.size() * sizeof(uint64_t));
        }
        return h.digest();
    }

    std::vector<int> free_block_ids() const
    {
        return std::vector<int>(free_block_ids_.begin(), free_block_ids_.end());
    }

    bool can_allocate(Sequence& seq) const
    {
        return (int)free_block_ids_.size() >= seq.num_blocks(engine_id_, sp_idx_);
    }

    void allocate(Sequence& seq, int token_idx_from = -1, int token_idx_to = -1)
    {
        (void)token_idx_from;
        (void)token_idx_to;

        auto& table = seq.block_table(engine_id_, sp_idx_);
        if (!table.empty()) {
            throw std::runtime_error("BlockManager.allocate expects empty seq.block_table");
        }

        bool                    cache_miss = false;
        std::optional<uint64_t> h;

        int num_blocks = seq.num_blocks(engine_id_, sp_idx_);
        for (int i = 0; i < num_blocks; ++i) {
            std::vector<int> token_ids = slice_seq_block(seq, i);

            if ((int)token_ids.size() == block_size_) {
                h = compute_hash(token_ids, h);
            }
            else {
                h.reset();
            }

            int block_id = -1;
            if (h.has_value()) {
                auto it = hash_to_block_id_.find(h.value());
                if (it != hash_to_block_id_.end())
                    block_id = it->second;
            }

            if (block_id == -1 || blocks_[block_id].token_ids != token_ids) {
                cache_miss = true;
            }

            NDCacheBlock* block = nullptr;
            if (cache_miss) {
                block_id = pop_free_block_id();
                block = allocate_block(block_id);
            }
            else {
                if (used_block_ids_.count(block_id)) {
                    block = &blocks_[block_id];
                    block->ref_count += 1;
                }
                else {
                    block = allocate_block(block_id);
                }
            }

            if (h.has_value()) {
                block->update(h.value(), token_ids);
                hash_to_block_id_[h.value()] = block_id;
            }

            auto& ctx = seq.block_ctx(engine_id_);
            ctx.block_location.push_back({sp_idx_, block_id});
            table.push_back(block_id);
        }
    }

    void deallocate(Sequence& seq)
    {
        auto& table = seq.block_table(engine_id_, sp_idx_);
        for (auto it = table.rbegin(); it != table.rend(); ++it) {
            int block_id = *it;
            auto& block = blocks_.at((size_t)block_id);
            block.ref_count -= 1;
            if (block.ref_count == 0) {
                deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        table.clear();
    }

    bool can_append(Sequence& seq, int num_tokens = 1) const
    {
        auto& ctx = seq.block_ctx(engine_id_);
        int   before = (ctx.num_dispatched_tokens[sp_idx_] + block_size_ - 1) / block_size_;
        int   after  = (ctx.num_dispatched_tokens[sp_idx_] + num_tokens + block_size_ - 1) / block_size_;
        return (int)free_block_ids_.size() >= (after - before);
    }

    void may_append(Sequence& seq, int num_tokens = 1)
    {
        for (int idx = 0; idx < num_tokens; ++idx) {
            auto& table = seq.block_table(engine_id_, sp_idx_);
            if (table.empty())
                throw std::runtime_error("BlockManager.may_append called with empty block_table");

            auto& ctx = seq.block_ctx(engine_id_);

            int dispatched = ctx.num_dispatched_tokens[sp_idx_] + idx;

            // Mirror Python condition:
            // if (num_dispatched + idx) % block_size == 1: allocate new block
            if ((dispatched % block_size_) == 1) {
                int block_id = pop_free_block_id();
                ctx.block_location.push_back({sp_idx_, block_id});
                allocate_block(block_id);
                table.push_back(block_id);
            }
            else if (((ctx.num_dispatched_tokens[sp_idx_] + idx - 1) % block_size_) == 0) {
                // Python has commented-out hash update logic here; keep as no-op.
                continue;
            }
            else {
                continue;
            }
        }
    }

    // Debug/inspection (optional)
    std::optional<std::string> engine_id() const { return engine_id_; }
    int sp_idx() const { return sp_idx_; }
    int block_size() const { return block_size_; }

private:
    std::vector<int> slice_seq_block(const Sequence& seq, int block_index) const
    {
        // This mirrors Sequence.block() which slices by Sequence::block_size (256).
        const size_t start = (size_t)block_index * (size_t)Sequence::block_size;
        size_t       end   = (size_t)(block_index + 1) * (size_t)Sequence::block_size;
        if (end > seq.token_ids.size())
            end = seq.token_ids.size();
        if (start >= seq.token_ids.size())
            return {};

        std::vector<int> out;
        out.reserve(end - start);
        for (size_t i = start; i < end; ++i)
            out.push_back(seq.token_ids[i]);
        return out;
    }

    int pop_free_block_id()
    {
        if (free_block_ids_.empty())
            throw std::runtime_error("No free blocks available");
        return free_block_ids_.front();
    }

    NDCacheBlock* allocate_block(int block_id)
    {
        auto& block = blocks_.at((size_t)block_id);
        if (block.ref_count != 0)
            throw std::runtime_error("allocate_block expects ref_count == 0");

        block.reset();
        // remove first occurrence
        for (auto it = free_block_ids_.begin(); it != free_block_ids_.end(); ++it) {
            if (*it == block_id) {
                free_block_ids_.erase(it);
                break;
            }
        }
        used_block_ids_.insert(block_id);
        return &block;
    }

    void deallocate_block(int block_id)
    {
        auto& block = blocks_.at((size_t)block_id);
        if (block.ref_count != 0)
            throw std::runtime_error("deallocate_block expects ref_count == 0");

        used_block_ids_.erase(block_id);
        free_block_ids_.push_back(block_id);
    }

private:
    std::optional<std::string>               engine_id_;
    int                                      sp_idx_ = 0;
    int                                      block_size_ = 0;

    std::vector<NDCacheBlock>                blocks_;
    std::unordered_map<uint64_t, int>        hash_to_block_id_;
    std::deque<int>                          free_block_ids_;
    std::unordered_set<int>                  used_block_ids_;
};

// Binding hook (implemented in block_manager_binding.cpp)
void bind_block_manager(py::module& m);
