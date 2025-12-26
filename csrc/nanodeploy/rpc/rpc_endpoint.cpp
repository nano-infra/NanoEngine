#include "rpc_endpoint.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace nanodeploy {

namespace {

class Serializer {
public:
    std::vector<uint8_t> buffer;

    Serializer()
    {
        buffer.reserve(4096);
    }

    template<typename T>
    void write(const T& val)
    {
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&val);
        buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
    }

    void write_string(const std::string& str)
    {
        write<size_t>(str.size());
        if (!str.empty()) {
            buffer.insert(buffer.end(), str.begin(), str.end());
        }
    }

    template<typename T>
    void write_vector_basic(const std::vector<T>& vec)
    {
        write<size_t>(vec.size());
        if (!vec.empty()) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(vec.data());
            buffer.insert(buffer.end(), ptr, ptr + vec.size() * sizeof(T));
        }
    }

    void write_block_locations(const std::vector<std::pair<int, int>>& vec)
    {
        write<size_t>(vec.size());
        for (const auto& p : vec) {
            write<int>(p.first);
            write<int>(p.second);
        }
    }

    void write_sp_block_table(const std::vector<std::vector<int>>& map)
    {
        write<size_t>(map.size());
        for (const auto& kv : map) {
            write_vector_basic(kv);
        }
    }
};

class Deserializer {
public:
    const uint8_t* ptr;
    const uint8_t* end;

    Deserializer(void* data, size_t size)
    {
        if (!data)
            throw std::runtime_error("Deserializer received null pointer");
        ptr = static_cast<const uint8_t*>(data);
        end = ptr + size;
    }

    template<typename T>
    T read()
    {
        if (ptr + sizeof(T) > end)
            throw std::runtime_error("Buffer underflow reading primitive");
        T val;
        std::memcpy(&val, ptr, sizeof(T));
        ptr += sizeof(T);
        return val;
    }

    std::string read_string()
    {
        size_t len = read<size_t>();
        if (ptr + len > end)
            throw std::runtime_error("Buffer underflow reading string");
        std::string str(reinterpret_cast<const char*>(ptr), len);
        ptr += len;
        return str;
    }

    template<typename T>
    std::vector<T> read_vector_basic()
    {
        size_t len = read<size_t>();
        if (ptr + len * sizeof(T) > end)
            throw std::runtime_error("Buffer underflow reading vector");
        std::vector<T> vec(len);
        if (len > 0) {
            std::memcpy(vec.data(), ptr, len * sizeof(T));
            ptr += len * sizeof(T);
        }
        return vec;
    }

    std::vector<std::pair<int, int>> read_block_locations()
    {
        size_t                           len = read<size_t>();
        std::vector<std::pair<int, int>> vec;
        vec.reserve(len);
        for (size_t i = 0; i < len; ++i) {
            int first  = read<int>();
            int second = read<int>();
            vec.emplace_back(first, second);
        }
        return vec;
    }

    std::vector<std::vector<int>> read_sp_block_table()
    {
        size_t                        len = read<size_t>();
        std::vector<std::vector<int>> map;
        for (size_t i = 0; i < len; ++i) {
            auto             vec_base = read_vector_basic<int>();
            std::vector<int> val(vec_base.begin(), vec_base.end());
            map.push_back(std::move(val));
        }
        return map;
    }
};

void serialize_block_context(Serializer& s, const BlockContext& ctx)
{
    s.write_string(ctx.engine_id_);
    s.write<int>(ctx.dp_idx_);
    s.write<int>(ctx.master_sp_idx_);
    s.write<int>(ctx.attention_sp_);
    s.write<int>(ctx.attention_dp_);

    s.write_block_locations(ctx.block_location);
    s.write_sp_block_table(ctx.sp_block_table);
    s.write_vector_basic(ctx.num_dispatched_tokens);
}

void deserialize_block_context(Deserializer& d, BlockContext& ctx)
{
    ctx.engine_id_     = d.read_string();
    ctx.dp_idx_        = d.read<int>();
    ctx.master_sp_idx_ = d.read<int>();
    ctx.attention_sp_  = d.read<int>();
    ctx.attention_dp_  = d.read<int>();

    ctx.block_location        = d.read_block_locations();
    ctx.sp_block_table        = d.read_sp_block_table();
    ctx.num_dispatched_tokens = d.read_vector_basic<int>();
}

void serialize_sequence(Serializer& s, const Sequence& seq)
{
    s.write<uint64_t>(seq.seq_id);
    s.write<SequenceStatus>(seq.status);
    s.write<double>(seq.temperature);
    s.write<int>(seq.max_tokens);
    s.write<bool>(seq.ignore_eos);

    s.write<int>(seq.last_token);
    s.write<int>(seq.num_tokens);
    s.write<int>(seq.num_prompt_tokens);
    s.write<int>(seq.num_checkpointed_tokens);
    s.write<int>(seq.num_cached_tokens);

    s.write_vector_basic(seq.token_ids);

    for (const auto& ctx : seq.slots_) {
        serialize_block_context(s, ctx);
    }
}

std::shared_ptr<Sequence> deserialize_sequence(Deserializer& d)
{
    uint64_t       seq_id      = d.read<uint64_t>();
    SequenceStatus status      = d.read<SequenceStatus>();
    double         temperature = d.read<double>();
    int            max_tokens  = d.read<int>();
    bool           ignore_eos  = d.read<bool>();

    int last_token              = d.read<int>();
    int num_tokens              = d.read<int>();
    int num_prompt_tokens       = d.read<int>();
    int num_checkpointed_tokens = d.read<int>();
    int num_cached_tokens       = d.read<int>();

    std::vector<int> token_ids = d.read_vector_basic<int>();

    auto seq                     = std::make_shared<Sequence>(token_ids, temperature, max_tokens, ignore_eos);
    seq->seq_id                  = seq_id;
    seq->status                  = status;
    seq->last_token              = last_token;
    seq->num_tokens              = num_tokens;
    seq->num_prompt_tokens       = num_prompt_tokens;
    seq->num_checkpointed_tokens = num_checkpointed_tokens;
    seq->num_cached_tokens       = num_cached_tokens;

    for (size_t i = 0; i < (size_t)BlockContextSlot::_COUNT; ++i) {
        deserialize_block_context(d, seq->slots_[i]);
    }

    return seq;
}

}  // namespace

void RpcEndpoint::set_buffer(uint64_t ptr, size_t size)
{
    data_     = reinterpret_cast<void*>(ptr);
    size_     = size;
    own_data_ = false;
    buffer_.clear();
    buffer_.shrink_to_fit();
}

int32_t RpcEndpoint::feed_sequences(std::vector<std::shared_ptr<Sequence>> seqs)
{
    seqs_ = std::move(seqs);
    return 0;
}

int32_t RpcEndpoint::serialize_for_prefill()
{
    Serializer s;
    s.write<size_t>(seqs_.size());
    for (const auto& seq : seqs_) {
        serialize_sequence(s, *seq);
    }

    buffer_   = std::move(s.buffer);
    data_     = buffer_.data();
    size_     = buffer_.size();
    own_data_ = true;

    return 0;
}

int32_t RpcEndpoint::serialize_for_decode()
{
    return serialize_for_prefill();
}

int32_t RpcEndpoint::serialize_for_migrate()
{
    return serialize_for_prefill();
}

int32_t RpcEndpoint::deserialize_for_prefill()
{
    if (!data_ || size_ == 0)
        return -1;

    try {
        Deserializer d(data_, size_);
        size_t       count = d.read<size_t>();

        seqs_.clear();
        seqs_.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            seqs_.push_back(deserialize_sequence(d));
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Deserialization error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int32_t RpcEndpoint::deserialize_for_decode()
{
    return deserialize_for_prefill();
}

int32_t RpcEndpoint::deserialize_for_migrate()
{
    return deserialize_for_prefill();
}

}  // namespace nanodeploy
