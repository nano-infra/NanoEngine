#pragma once

#include <cstdint>
#include <string>

#include "dlengine/csrc/common/json.hpp"

namespace dlengine {

#define DLENGINE_CACHE_SPEC_INT_FIELD(name) int name = -1;
#define DLENGINE_CACHE_SPEC_JSON_FIELD(name) j[#name] = name;
#define DLENGINE_DEFINE_CACHE_SPEC_JSON(FIELDS)                                                                        \
    nlohmann::json to_json(bool enabled) const                                                                         \
    {                                                                                                                  \
        nlohmann::json j = {{"enabled", enabled}};                                                                     \
        FIELDS(DLENGINE_CACHE_SPEC_JSON_FIELD)                                                                         \
        return j;                                                                                                      \
    }

#define DLENGINE_GQA_CACHE_SPEC_FIELDS(X)                                                                              \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)                                                                                              \
    X(num_layers)                                                                                                      \
    X(num_kv_heads)                                                                                                    \
    X(head_dim)

#define DLENGINE_MLA_CACHE_SPEC_FIELDS(X)                                                                              \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)                                                                                              \
    X(num_layers)                                                                                                      \
    X(kv_lora_rank)                                                                                                    \
    X(qk_rope_head_dim)                                                                                                \
    X(head_dim)

#define DLENGINE_GDN_CACHE_SPEC_FIELDS(X)                                                                              \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)                                                                                              \
    X(state_slots)                                                                                                     \
    X(state_bytes)

#define DLENGINE_HCA_CACHE_SPEC_FIELDS(X)                                                                              \
    X(bytes_per_token)                                                                                                 \
    X(block_size_multiple)                                                                                             \
    X(compression_ratio)                                                                                               \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)

#define DLENGINE_CSA_CACHE_SPEC_FIELDS(X)                                                                              \
    X(compressor_head_dim)                                                                                             \
    X(compression_ratio)                                                                                               \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)

#define DLENGINE_INDEXER_CACHE_SPEC_FIELDS(X)                                                                          \
    X(num_pages)                                                                                                       \
    X(page_size)                                                                                                       \
    X(max_blocks_per_seq)                                                                                              \
    X(index_head_dim)                                                                                                  \
    X(bytes_per_token)

#define DLENGINE_HISPARSE_CACHE_SPEC_FIELDS(X)                                                                         \
    X(max_num_seqs)                                                                                                    \
    X(device_buffer_size)                                                                                              \
    X(host_to_device_ratio)                                                                                            \
    X(swap_in_block_size)                                                                                              \
    X(dummy_slot)

#define DLENGINE_CACHE_PLAN_COMPONENTS(X)                                                                              \
    X(Gqa, gqa, GqaCacheSpec, 0)                                                                                       \
    X(Mla, mla, MlaCacheSpec, 1)                                                                                       \
    X(Gdn, gdn, GdnCacheSpec, 2)                                                                                       \
    X(Hca, hca, HcaCacheSpec, 3)                                                                                       \
    X(Csa, csa, CsaCacheSpec, 4)                                                                                       \
    X(Indexer, indexer, IndexerCacheSpec, 5)                                                                           \
    X(Hisparse, hisparse, HiSparseCacheSpec, 6)

enum class CachePlanFlag : uint32_t {
#define DLENGINE_CACHE_PLAN_ENUM_ENTRY(flag, field, spec_type, bit) flag = 1u << bit,
    DLENGINE_CACHE_PLAN_COMPONENTS(DLENGINE_CACHE_PLAN_ENUM_ENTRY)
#undef DLENGINE_CACHE_PLAN_ENUM_ENTRY
};

inline uint32_t cache_plan_flag(CachePlanFlag flag)
{
    return static_cast<uint32_t>(flag);
}

inline const char* cache_plan_flag_name(CachePlanFlag flag)
{
    switch (flag) {
#define DLENGINE_CACHE_PLAN_FLAG_NAME_CASE(flag, field, spec_type, bit)                                                \
    case CachePlanFlag::flag:                                                                                          \
        return #field;
        DLENGINE_CACHE_PLAN_COMPONENTS(DLENGINE_CACHE_PLAN_FLAG_NAME_CASE)
#undef DLENGINE_CACHE_PLAN_FLAG_NAME_CASE
    }
    return "unknown";
}

struct GqaCacheSpec {
    DLENGINE_GQA_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_GQA_CACHE_SPEC_FIELDS)
};

struct MlaCacheSpec {
    DLENGINE_MLA_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_MLA_CACHE_SPEC_FIELDS)
};

struct GdnCacheSpec {
    DLENGINE_GDN_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_GDN_CACHE_SPEC_FIELDS)
};

struct HcaCacheSpec {
    DLENGINE_HCA_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_HCA_CACHE_SPEC_FIELDS)
};

struct CsaCacheSpec {
    DLENGINE_CSA_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_CSA_CACHE_SPEC_FIELDS)
};

struct IndexerCacheSpec {
    DLENGINE_INDEXER_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_INDEXER_CACHE_SPEC_FIELDS)
};

struct HiSparseCacheSpec {
    DLENGINE_HISPARSE_CACHE_SPEC_FIELDS(DLENGINE_CACHE_SPEC_INT_FIELD)
    DLENGINE_DEFINE_CACHE_SPEC_JSON(DLENGINE_HISPARSE_CACHE_SPEC_FIELDS)
};

struct CachePlan {
    uint32_t flags = 0;

#define DLENGINE_CACHE_PLAN_MEMBER(flag, field, spec_type, bit) spec_type field;
    DLENGINE_CACHE_PLAN_COMPONENTS(DLENGINE_CACHE_PLAN_MEMBER)
#undef DLENGINE_CACHE_PLAN_MEMBER

    bool has_flag(CachePlanFlag flag) const
    {
        return (flags & cache_plan_flag(flag)) != 0;
    }

    void set_flag(CachePlanFlag flag)
    {
        flags |= cache_plan_flag(flag);
    }

#define DLENGINE_CACHE_PLAN_HAS_METHOD(flag, field, spec_type, bit)                                                    \
    bool has_##field() const                                                                                           \
    {                                                                                                                  \
        return has_flag(CachePlanFlag::flag);                                                                          \
    }
    DLENGINE_CACHE_PLAN_COMPONENTS(DLENGINE_CACHE_PLAN_HAS_METHOD)
#undef DLENGINE_CACHE_PLAN_HAS_METHOD

    bool has_linear_attention() const
    {
        return has_gdn();
    }

    std::string cache_mode() const
    {
        if (has_hca() || has_csa()) {
            return "dsv4";
        }
        if (has_mla() || has_indexer() || has_hisparse()) {
            return "mla";
        }
        return "gqa";
    }

    nlohmann::json to_json() const
    {
        nlohmann::json enabled = nlohmann::json::array();
        nlohmann::json specs   = nlohmann::json::object();
#define DLENGINE_CACHE_PLAN_JSON_COMPONENT(flag, field, spec_type, bit)                                                \
    if (has_##field()) {                                                                                               \
        enabled.push_back(cache_plan_flag_name(CachePlanFlag::flag));                                                  \
        specs[#field] = field.to_json(true);                                                                           \
    }
        DLENGINE_CACHE_PLAN_COMPONENTS(DLENGINE_CACHE_PLAN_JSON_COMPONENT)
#undef DLENGINE_CACHE_PLAN_JSON_COMPONENT

        nlohmann::json plan = {
            {"mode", cache_mode()},
            {"flags", flags},
            {"enabled", enabled},
        };
        plan.update(specs);
        return plan;
    }

    std::string to_json_string(int indent = -1) const
    {
        return to_json().dump(indent);
    }
};

}  // namespace dlengine
