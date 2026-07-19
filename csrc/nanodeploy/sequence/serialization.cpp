#include <algorithm>

#include "nanodeploy/logging.h"

#include "serialization.h"

namespace nanodeploy {

namespace {

// --- Low-level atomic write operations (inlined for performance) ---
template<typename T>
inline void write_raw(uintptr_t base, size_t& off, size_t max_size, const T& val)
{
    if (off > max_size || sizeof(T) > max_size - off)
        NANODEPLOY_ABORT("Buffer Overflow");
    std::memcpy(reinterpret_cast<void*>(base + off), &val, sizeof(T));
    off += sizeof(T);
}

inline void write_bytes(uintptr_t base, size_t& off, size_t max_size, const void* src, size_t len)
{
    if (off > max_size || len > max_size - off)
        NANODEPLOY_ABORT("Buffer Overflow");
    if (len > 0) {
        std::memcpy(reinterpret_cast<void*>(base + off), src, len);
        off += len;
    }
}

template<typename T>
inline T read_raw(uintptr_t base, size_t& off, size_t max_size)
{
    if (off > max_size || sizeof(T) > max_size - off)
        throw std::runtime_error("Buffer Underflow");
    T val;
    std::memcpy(&val, reinterpret_cast<void*>(base + off), sizeof(T));
    off += sizeof(T);
    return val;
}

inline void read_bytes(uintptr_t base, size_t& off, size_t max_size, void* dst, size_t len)
{
    if (off > max_size || len > max_size - off)
        throw std::runtime_error("Buffer Underflow");
    if (len > 0) {
        std::memcpy(dst, reinterpret_cast<void*>(base + off), len);
        off += len;
    }
}

size_t checked_payload_bytes(size_t count, size_t element_size, size_t off, size_t max, const char* field)
{
    if (off > max || (element_size != 0 && count > (max - off) / element_size)) {
        throw std::runtime_error(std::string("Invalid serialized ") + field + " length");
    }
    return count * element_size;
}

void validate_sequence_fields(SequenceStatus status,
                              int            assigned_dp,
                              int            max_tokens,
                              int            num_tokens,
                              int            num_prompt_tokens,
                              int            num_checkpointed_tokens,
                              int            num_cached_tokens)
{
    if (!is_valid_sequence_status(status)) {
        throw std::runtime_error("Invalid serialized SequenceStatus ordinal");
    }
    if (assigned_dp < -1) {
        throw std::runtime_error("Invalid serialized assigned_dp");
    }
    if (max_tokens < 0) {
        throw std::runtime_error("Invalid serialized max_tokens");
    }
    if (num_tokens < 0 || num_prompt_tokens < 0 || num_prompt_tokens > num_tokens
        || num_checkpointed_tokens < 0 || num_checkpointed_tokens > num_tokens || num_cached_tokens < 0
        || num_cached_tokens > num_tokens) {
        throw std::runtime_error("Invalid serialized Sequence token counters");
    }
}

void validate_block_context(const BlockContext& ctx)
{
    if (ctx.attention_sp_ <= 0 || ctx.attention_dp_ <= 0 || ctx.dp_idx_ < -1
        || ctx.dp_idx_ >= ctx.attention_dp_ || ctx.master_sp_idx_ < -1
        || ctx.master_sp_idx_ >= ctx.attention_sp_) {
        throw std::runtime_error("Invalid serialized BlockContext topology");
    }

    const size_t attention_sp = static_cast<size_t>(ctx.attention_sp_);
    const bool   uninitialized_vectors = ctx.sp_block_table.empty() && ctx.num_dispatched_tokens.empty();
    const bool   dimensioned_vectors =
        ctx.sp_block_table.size() == attention_sp && ctx.num_dispatched_tokens.size() == attention_sp;
    if (!uninitialized_vectors && !dimensioned_vectors) {
        throw std::runtime_error("Invalid serialized BlockContext vector dimensions");
    }

    if (ctx.pending_token_present_) {
        if (!dimensioned_vectors || ctx.pending_token_target_sp_ < 0
            || ctx.pending_token_target_sp_ >= ctx.attention_sp_
            || ctx.num_dispatched_tokens[ctx.pending_token_target_sp_] <= 0) {
            throw std::runtime_error("Invalid serialized pending-token target");
        }
    }
    else if (ctx.pending_token_target_sp_ != -1) {
        throw std::runtime_error("Invalid serialized pending-token state");
    }
    for (int dispatched_tokens : ctx.num_dispatched_tokens) {
        if (dispatched_tokens < 0) {
            throw std::runtime_error("Invalid serialized dispatched-token count");
        }
    }

    std::vector<std::pair<int, int>> table_locations;
    for (size_t rank = 0; rank < ctx.sp_block_table.size(); ++rank) {
        const auto& table = ctx.sp_block_table[rank];
        table_locations.reserve(table_locations.size() + table.size());
        for (int block_id : table) {
            if (block_id < 0) {
                throw std::runtime_error("Invalid serialized block-table ID");
            }
            table_locations.emplace_back(static_cast<int>(rank), block_id);
        }
    }

    std::vector<std::pair<int, int>> published_locations(ctx.block_location.begin(), ctx.block_location.end());
    for (const auto& [rank, block_id] : published_locations) {
        if (rank < 0 || rank >= ctx.attention_sp_ || block_id < 0) {
            throw std::runtime_error("Invalid serialized block location");
        }
    }
    std::sort(table_locations.begin(), table_locations.end());
    std::sort(published_locations.begin(), published_locations.end());
    if (std::adjacent_find(table_locations.begin(), table_locations.end()) != table_locations.end()
        || std::adjacent_find(published_locations.begin(), published_locations.end()) != published_locations.end()) {
        throw std::runtime_error("Serialized BlockContext contains duplicate block locations");
    }
    if (table_locations != published_locations) {
        throw std::runtime_error("Serialized block tables and locations disagree");
    }

    if (ctx.master_sp_idx_ == -1
        && (ctx.pending_token_present_ || !ctx.block_location.empty()
            || std::any_of(ctx.sp_block_table.begin(), ctx.sp_block_table.end(), [](const auto& table) {
                   return !table.empty();
               })
            || std::any_of(ctx.num_dispatched_tokens.begin(),
                           ctx.num_dispatched_tokens.end(),
                           [](int tokens) { return tokens != 0; }))) {
        throw std::runtime_error("Serialized inactive BlockContext retains live state");
    }
}

// ==================== 对象级逻辑实现 ====================

void serialize_block_context(uintptr_t base, size_t& off, size_t max, const BlockContext& ctx, int target_sp_rank = -1)
{
    validate_block_context(ctx);
    const bool initialized_or_used = ctx.dp_idx_ != -1 || !ctx.engine_id_.empty()
                                     || !ctx.sp_block_table.empty() || !ctx.num_dispatched_tokens.empty()
                                     || !ctx.block_location.empty() || ctx.pending_token_present_;
    if (target_sp_rank < -1
        || (target_sp_rank >= ctx.attention_sp_ && initialized_or_used)) {
        throw std::runtime_error("Serialized BlockContext target SP rank is outside its topology");
    }

    // String
    size_t s_len = ctx.engine_id_.size();
    write_raw(base, off, max, s_len);
    write_bytes(base, off, max, ctx.engine_id_.data(), s_len);

    // Primitives
    write_raw(base, off, max, ctx.dp_idx_);
    write_raw(base, off, max, ctx.master_sp_idx_);
    write_raw(base, off, max, ctx.attention_sp_);
    write_raw(base, off, max, ctx.attention_dp_);
    const uint8_t pending_token_present = ctx.pending_token_present_ ? 1 : 0;
    write_raw(base, off, max, pending_token_present);
    write_raw(base, off, max, ctx.pending_token_target_sp_);

    // In decode optimize mode, keep the sequence skeleton intact and only trim
    // per-target heavy fields inside the block context.
    BlockContext::BlockLocationList filtered_locations;
    const bool trim_for_target = target_sp_rank >= 0 && target_sp_rank < ctx.attention_sp_;
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
        const auto& inner    = ctx.sp_block_table[sp_idx];
        size_t      inner_sz = (trim_for_target && static_cast<int>(sp_idx) != target_sp_rank) ? 0 : inner.size();
        write_raw(base, off, max, inner_sz);
        write_bytes(base, off, max, inner.data(), inner_sz * sizeof(int));
    }
}

void deserialize_block_context(uintptr_t base, size_t& off, size_t max, BlockContext& ctx)
{
    size_t s_len = read_raw<size_t>(base, off, max);
    checked_payload_bytes(s_len, sizeof(char), off, max, "engine_id");
    ctx.engine_id_.assign(reinterpret_cast<const char*>(base + off), s_len);
    off += s_len;

    ctx.dp_idx_                  = read_raw<int>(base, off, max);
    ctx.master_sp_idx_           = read_raw<int>(base, off, max);
    ctx.attention_sp_            = read_raw<int>(base, off, max);
    ctx.attention_dp_            = read_raw<int>(base, off, max);
    const uint8_t pending_token_present = read_raw<uint8_t>(base, off, max);
    if (pending_token_present > 1) {
        throw std::runtime_error("Invalid serialized pending-token flag");
    }
    ctx.pending_token_present_   = pending_token_present != 0;
    ctx.pending_token_target_sp_ = read_raw<int>(base, off, max);

    size_t loc_count = read_raw<size_t>(base, off, max);
    const size_t loc_bytes =
        checked_payload_bytes(loc_count, sizeof(std::pair<int, int>), off, max, "block_location");
    ctx.block_location.resize(loc_count);
    read_bytes(base, off, max, ctx.block_location.data(), loc_bytes);

    size_t disp_count = read_raw<size_t>(base, off, max);
    const size_t disp_bytes = checked_payload_bytes(disp_count, sizeof(int), off, max, "num_dispatched_tokens");
    ctx.num_dispatched_tokens.resize(disp_count);
    read_bytes(base, off, max, ctx.num_dispatched_tokens.data(), disp_bytes);

    size_t table_size = read_raw<size_t>(base, off, max);
    checked_payload_bytes(table_size, sizeof(size_t), off, max, "sp_block_table");
    ctx.sp_block_table.resize(table_size);
    for (size_t i = 0; i < table_size; ++i) {
        size_t inner_sz = read_raw<size_t>(base, off, max);
        const size_t inner_bytes = checked_payload_bytes(inner_sz, sizeof(int), off, max, "block table");
        ctx.sp_block_table[i].resize(inner_sz);
        read_bytes(base, off, max, ctx.sp_block_table[i].data(), inner_bytes);
    }
    validate_block_context(ctx);
}

}  // namespace

// ==================== Public API ====================

void validate_serializable_block_context(const BlockContext& context)
{
    validate_block_context(context);
}

void validate_serializable_sequence_context_ownership(int                 assigned_dp,
                                                      SequenceStatus      status,
                                                      const BlockContext& active_context)
{
    if (assigned_dp == -1) {
        return;
    }
    const bool initialized_or_used = status != SequenceStatus::WAITING || active_context.dp_idx_ != -1
                                     || !active_context.engine_id_.empty() || !active_context.sp_block_table.empty()
                                     || !active_context.num_dispatched_tokens.empty()
                                     || !active_context.block_location.empty() || active_context.pending_token_present_;
    if (initialized_or_used && active_context.dp_idx_ != assigned_dp) {
        throw std::runtime_error(
            "Serialized assigned_dp disagrees with initialized ACTIVE BlockContext dp_idx");
    }
}

size_t serialize_sequences(uintptr_t                                     data_ptr,
                           size_t                                        buffer_size,
                           const std::vector<std::shared_ptr<Sequence>>& seqs,
                           bool                                          is_prefill,
                           int                                           sp_rank,
                           int                                           sp_size)
{
    size_t off = 0;

    write_raw(data_ptr, off, buffer_size, kSequenceSerializationMagic);
    write_raw(data_ptr, off, buffer_size, kSequenceSerializationVersion);

    const bool trim_decode_heavy_fields = !is_prefill && sp_rank >= 0 && sp_size > 0;
    if (!is_prefill
        && ((sp_rank == -1) != (sp_size == -1)
            || (sp_rank != -1 && (sp_size <= 0 || sp_rank < 0 || sp_rank >= sp_size)))) {
        throw std::runtime_error("Invalid optimized Decode serialization SP target");
    }

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

        validate_sequence_fields(seq.status,
                                 seq.assigned_dp,
                                 seq.max_tokens,
                                 seq.num_tokens,
                                 seq.num_prompt_tokens,
                                 seq.num_checkpointed_tokens,
                                 seq.num_cached_tokens);
        validate_serializable_sequence_context_ownership(
            seq.assigned_dp, seq.status, seq.slots_[(size_t)BlockContextSlot::ACTIVE]);
        if (is_prefill) {
            if (seq.token_ids.size() != static_cast<size_t>(seq.num_tokens)) {
                throw std::runtime_error("Prefill serialization requires the complete token history");
            }
            if (!seq.token_ids.empty() && seq.token_ids.back() != seq.last_token) {
                throw std::runtime_error("Prefill serialization token history disagrees with last_token");
            }
        }

        write_raw(data_ptr, off, buffer_size, seq.seq_id);
        const int32_t status_ordinal = static_cast<int32_t>(seq.status);
        write_raw(data_ptr, off, buffer_size, status_ordinal);
        write_raw(data_ptr, off, buffer_size, seq.assigned_dp);
        write_raw(data_ptr, off, buffer_size, seq.temperature);
        write_raw(data_ptr, off, buffer_size, seq.max_tokens);
        const uint8_t ignore_eos = seq.ignore_eos ? 1 : 0;
        write_raw(data_ptr, off, buffer_size, ignore_eos);
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
            serialize_block_context(data_ptr, off, buffer_size, seq.slots_[i], trim_decode_heavy_fields ? sp_rank : -1);
        }
    }
    return off;
}

std::vector<std::shared_ptr<Sequence>> deserialize_sequences(uintptr_t data_ptr, size_t data_len)
{
    size_t off = 0;
    const uint64_t magic = read_raw<uint64_t>(data_ptr, off, data_len);
    if (magic != kSequenceSerializationMagic) {
        throw std::runtime_error("Unsupported legacy or invalid Sequence raw payload magic");
    }
    const uint32_t version = read_raw<uint32_t>(data_ptr, off, data_len);
    if (version != kSequenceSerializationVersion) {
        throw std::runtime_error("Unsupported Sequence raw payload version");
    }

    size_t count = read_raw<size_t>(data_ptr, off, data_len);
    checked_payload_bytes(count, sizeof(uint64_t), off, data_len, "sequence count");

    std::vector<std::shared_ptr<Sequence>> seqs;
    seqs.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        // 先读取基础字段以便构造
        uint64_t       seq_id = read_raw<uint64_t>(data_ptr, off, data_len);
        const int32_t  status_ordinal = read_raw<int32_t>(data_ptr, off, data_len);
        SequenceStatus status = static_cast<SequenceStatus>(status_ordinal);
        int            assigned_dp = read_raw<int>(data_ptr, off, data_len);
        double         temp   = read_raw<double>(data_ptr, off, data_len);
        int            max_t  = read_raw<int>(data_ptr, off, data_len);
        const uint8_t  eos_raw = read_raw<uint8_t>(data_ptr, off, data_len);
        if (eos_raw > 1) {
            throw std::runtime_error("Invalid serialized ignore_eos flag");
        }
        bool           eos    = eos_raw != 0;
        int            last   = read_raw<int>(data_ptr, off, data_len);
        int            num    = read_raw<int>(data_ptr, off, data_len);
        int            prompt = read_raw<int>(data_ptr, off, data_len);
        int            check  = read_raw<int>(data_ptr, off, data_len);
        int            cached = read_raw<int>(data_ptr, off, data_len);

        validate_sequence_fields(status, assigned_dp, max_t, num, prompt, check, cached);

        size_t           tid_count = read_raw<size_t>(data_ptr, off, data_len);
        const size_t tid_bytes = checked_payload_bytes(tid_count, sizeof(int), off, data_len, "token_ids");
        if (tid_count != 0 && tid_count != static_cast<size_t>(num)) {
            throw std::runtime_error("Serialized token history is neither complete nor decode-trimmed");
        }
        std::vector<int> tids(tid_count);
        read_bytes(data_ptr, off, data_len, tids.data(), tid_bytes);
        if (!tids.empty() && tids.back() != last) {
            throw std::runtime_error("Serialized token history disagrees with last_token");
        }

        std::array<BlockContext, (size_t)BlockContextSlot::_COUNT> slots;
        for (size_t j = 0; j < (size_t)BlockContextSlot::_COUNT; ++j) {
            deserialize_block_context(data_ptr, off, data_len, slots[j]);
        }
        validate_serializable_sequence_context_ownership(
            assigned_dp, status, slots[(size_t)BlockContextSlot::ACTIVE]);

        auto seq                     = std::make_shared<Sequence>(tids, temp, max_t, eos);
        seq->restore_seq_id(seq_id);
        seq->status                  = status;
        seq->assigned_dp             = assigned_dp;
        seq->last_token              = last;
        seq->num_tokens              = num;
        seq->num_prompt_tokens       = prompt;
        seq->num_checkpointed_tokens = check;
        seq->num_cached_tokens       = cached;
        seq->slots_                  = std::move(slots);
        seqs.push_back(seq);
    }
    if (off != data_len) {
        throw std::runtime_error("Trailing bytes after Sequence raw payload");
    }
    return seqs;
}

}  // namespace nanodeploy
