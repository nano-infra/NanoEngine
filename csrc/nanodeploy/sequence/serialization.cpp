#include "nanodeploy/logging.h"

#include "serialization.h"

namespace nanodeploy {

namespace {

// --- Low-level atomic write operations (inlined for performance) ---
template<typename T>
inline void write_raw(uintptr_t base, size_t& off, size_t max_size, const T& val)
{
    if (off + sizeof(T) > max_size)
        NANODEPLOY_ABORT("Buffer Overflow");
    std::memcpy(reinterpret_cast<void*>(base + off), &val, sizeof(T));
    off += sizeof(T);
}

inline void write_bytes(uintptr_t base, size_t& off, size_t max_size, const void* src, size_t len)
{
    if (off + len > max_size)
        NANODEPLOY_ABORT("Buffer Overflow");
    if (len > 0) {
        std::memcpy(reinterpret_cast<void*>(base + off), src, len);
        off += len;
    }
}

template<typename T>
inline T read_raw(uintptr_t base, size_t& off, size_t max_size)
{
    if (off + sizeof(T) > max_size)
        throw std::runtime_error("Buffer Underflow");
    T val;
    std::memcpy(&val, reinterpret_cast<void*>(base + off), sizeof(T));
    off += sizeof(T);
    return val;
}

inline void read_bytes(uintptr_t base, size_t& off, size_t max_size, void* dst, size_t len)
{
    if (off + len > max_size)
        throw std::runtime_error("Buffer Underflow");
    if (len > 0) {
        std::memcpy(dst, reinterpret_cast<void*>(base + off), len);
        off += len;
    }
}

// ==================== 对象级逻辑实现 ====================

void serialize_block_context(uintptr_t      base,
                             size_t&        off,
                             size_t         max,
                             const BlockContext& ctx,
                             int            target_sp_rank = -1)
{
    // String
    size_t s_len = ctx.engine_id_.size();
    write_raw(base, off, max, s_len);
    write_bytes(base, off, max, ctx.engine_id_.data(), s_len);

    // Primitives
    write_raw(base, off, max, ctx.dp_idx_);
    write_raw(base, off, max, ctx.master_sp_idx_);
    write_raw(base, off, max, ctx.attention_sp_);
    write_raw(base, off, max, ctx.attention_dp_);

    // In decode optimize mode, keep the sequence skeleton intact and only trim
    // per-target heavy fields inside the block context.
    BlockContext::BlockLocationList filtered_locations;
    const bool                      trim_for_target = target_sp_rank >= 0;
    if (trim_for_target) {
        filtered_locations.reserve(ctx.block_location.size());
        for (const auto& loc : ctx.block_location) {
            if (loc.first == target_sp_rank) {
                filtered_locations.push_back(loc);
            }
        }
    }

    // Vector<pair<int, int>> - 直接块拷贝
    size_t loc_count = trim_for_target ? filtered_locations.size() : ctx.block_location.size();
    write_raw(base, off, max, loc_count);
    if (trim_for_target) {
        write_bytes(base, off, max, filtered_locations.data(), loc_count * sizeof(std::pair<int, int>));
    }
    else {
        write_bytes(base, off, max, ctx.block_location.data(), loc_count * sizeof(std::pair<int, int>));
    }

    // Vector<int>
    size_t disp_count = ctx.num_dispatched_tokens.size();
    write_raw(base, off, max, disp_count);
    write_bytes(base, off, max, ctx.num_dispatched_tokens.data(), disp_count * sizeof(int));

    // Nested Vector<Vector<int>>
    size_t table_size = ctx.sp_block_table.size();
    write_raw(base, off, max, table_size);
    for (size_t sp_idx = 0; sp_idx < table_size; ++sp_idx) {
        const auto& inner = ctx.sp_block_table[sp_idx];
        size_t      inner_sz =
            (trim_for_target && static_cast<int>(sp_idx) != target_sp_rank) ? 0 : inner.size();
        write_raw(base, off, max, inner_sz);
        write_bytes(base, off, max, inner.data(), inner_sz * sizeof(int));
    }
}

void deserialize_block_context(uintptr_t base, size_t& off, size_t max, BlockContext& ctx)
{
    size_t s_len = read_raw<size_t>(base, off, max);
    ctx.engine_id_.assign(reinterpret_cast<const char*>(base + off), s_len);
    off += s_len;

    ctx.dp_idx_        = read_raw<int>(base, off, max);
    ctx.master_sp_idx_ = read_raw<int>(base, off, max);
    ctx.attention_sp_  = read_raw<int>(base, off, max);
    ctx.attention_dp_  = read_raw<int>(base, off, max);

    size_t loc_count = read_raw<size_t>(base, off, max);
    ctx.block_location.resize(loc_count);
    read_bytes(base, off, max, ctx.block_location.data(), loc_count * sizeof(std::pair<int, int>));

    size_t disp_count = read_raw<size_t>(base, off, max);
    ctx.num_dispatched_tokens.resize(disp_count);
    read_bytes(base, off, max, ctx.num_dispatched_tokens.data(), disp_count * sizeof(int));

    size_t table_size = read_raw<size_t>(base, off, max);
    ctx.sp_block_table.resize(table_size);
    for (size_t i = 0; i < table_size; ++i) {
        size_t inner_sz = read_raw<size_t>(base, off, max);
        ctx.sp_block_table[i].resize(inner_sz);
        read_bytes(base, off, max, ctx.sp_block_table[i].data(), inner_sz * sizeof(int));
    }
}

}  // namespace

// ==================== Public API ====================

size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill,
                           int                                           sp_rank,
                           int                                           sp_size)
{
    size_t off = 0;

    const bool trim_decode_heavy_fields = !is_prefill && sp_rank >= 0 && sp_size > 0;

    // Always preserve the full decode sequence skeleton and ordering. In the
    // optimized decode path we only trim heavy per-sequence fields.
    std::vector<std::shared_ptr<Sequence>> serialized_seqs;
    serialized_seqs.reserve(seqs.size());
    for (const auto& seq_ptr : seqs) {
        if (seq_ptr) {
            serialized_seqs.push_back(seq_ptr);
        }
    }

    // 写入数量（保持原始 sequence skeleton）
    size_t count = serialized_seqs.size();
    write_raw(data_ptr, off, buffer_size, count);

    for (const auto& seq_ptr : serialized_seqs) {
        const auto& seq = *seq_ptr;

        write_raw(data_ptr, off, buffer_size, seq.seq_id);
        write_raw(data_ptr, off, buffer_size, seq.status);
        write_raw(data_ptr, off, buffer_size, seq.temperature);
        write_raw(data_ptr, off, buffer_size, seq.max_tokens);
        write_raw(data_ptr, off, buffer_size, seq.ignore_eos);
        write_raw(data_ptr, off, buffer_size, seq.last_token);
        write_raw(data_ptr, off, buffer_size, seq.num_tokens);
        write_raw(data_ptr, off, buffer_size, seq.num_prompt_tokens);
        write_raw(data_ptr, off, buffer_size, seq.num_checkpointed_tokens);
        write_raw(data_ptr, off, buffer_size, seq.num_cached_tokens);

        // token_ids
        if (is_prefill) {
            // Prefill 阶段：传输完整的 token_ids
            size_t tid_count = seq.token_ids.size();
            write_raw(data_ptr, off, buffer_size, tid_count);
            write_bytes(data_ptr, off, buffer_size, seq.token_ids.data(), tid_count * sizeof(int));
        }
        else {
            // Decode 阶段：不传输 token_ids，写入长度 0
            size_t tid_count = 0;
            write_raw(data_ptr, off, buffer_size, tid_count);
        }

        // Slots (BlockContexts)
        for (size_t i = 0; i < (size_t)BlockContextSlot::_COUNT; ++i) {
            serialize_block_context(data_ptr,
                                    off,
                                    buffer_size,
                                    seq.slots_[i],
                                    trim_decode_heavy_fields ? sp_rank : -1);
        }
    }
    return off;
}

std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len)
{
    size_t off   = 0;
    size_t count = read_raw<size_t>(data_ptr, off, data_len);

    std::vector<std::shared_ptr<Sequence>> seqs;
    seqs.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // 先读取基础字段以便构造
        uint64_t       seq_id = read_raw<uint64_t>(data_ptr, off, data_len);
        SequenceStatus status = read_raw<SequenceStatus>(data_ptr, off, data_len);
        double         temp   = read_raw<double>(data_ptr, off, data_len);
        int            max_t  = read_raw<int>(data_ptr, off, data_len);
        bool           eos    = read_raw<bool>(data_ptr, off, data_len);
        int            last   = read_raw<int>(data_ptr, off, data_len);
        int            num    = read_raw<int>(data_ptr, off, data_len);
        int            prompt = read_raw<int>(data_ptr, off, data_len);
        int            check  = read_raw<int>(data_ptr, off, data_len);
        int            cached = read_raw<int>(data_ptr, off, data_len);

        size_t           tid_count = read_raw<size_t>(data_ptr, off, data_len);
        std::vector<int> tids(tid_count);
        read_bytes(data_ptr, off, data_len, tids.data(), tid_count * sizeof(int));

        auto seq                     = std::make_shared<Sequence>(tids, temp, max_t, eos);
        seq->seq_id                  = seq_id;
        seq->status                  = status;
        seq->last_token              = last;
        seq->num_tokens              = num;
        seq->num_prompt_tokens       = prompt;
        seq->num_checkpointed_tokens = check;
        seq->num_cached_tokens       = cached;

        for (size_t j = 0; j < (size_t)BlockContextSlot::_COUNT; ++j) {
            deserialize_block_context(data_ptr, off, data_len, seq->slots_[j]);
        }
        seqs.push_back(seq);
    }
    return seqs;
}

}  // namespace nanodeploy
