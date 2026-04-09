#include "serialization.h"

#include <flatbuffers/flatbuffers.h>

#include "interface_generated.h"

namespace nanodeploy {

flatbuffers::DetachedBuffer serialize_run_batch(const std::vector<Sequence*>& seqs, bool is_prefill)
{
    flatbuffers::FlatBufferBuilder builder(4096);

    std::vector<flatbuffers::Offset<fbs::SequenceInput>> seq_offsets;
    seq_offsets.reserve(seqs.size());

    for (auto* seq : seqs) {
        auto& ctx = seq->block_ctx(BlockContextSlot::ACTIVE);

        // group_block_table: send all allocated blocks including extras reserved
        // by may_append() for speculative (MTP) verification, which writes KV
        // at positions beyond the current num_tokens.
        std::vector<flatbuffers::Offset<fbs::IntListI>> group_bt_offsets;
        for (size_t sp = 0; sp < ctx.group_block_table.size(); ++sp) {
            std::vector<int> vals;
            if (ctx.group_block_table[sp]) {
                const auto& full = ctx.group_block_table[sp]->values;
                vals.assign(full.begin(), full.end());
            }
            auto vals_vec = builder.CreateVector(vals);
            group_bt_offsets.push_back(fbs::CreateIntListI(builder, vals_vec));
        }
        auto group_bt_vec = builder.CreateVector(group_bt_offsets);

        // num_dispatched_tokens
        auto ndt_vec = builder.CreateVector(ctx.num_dispatched_tokens);

        // token_ids (prefill only: uncached portion)
        flatbuffers::Offset<flatbuffers::Vector<int32_t>> token_ids_off = 0;
        if (is_prefill) {
            const auto& full_tokens = seq->token_ids();
            int         num_cached  = seq->num_cached_tokens();
            int         total       = seq->num_tokens();
            if (num_cached < total) {
                token_ids_off = builder.CreateVector(full_tokens.data() + num_cached, total - num_cached);
            }
            else {
                token_ids_off = builder.CreateVector(std::vector<int>{});
            }
        }

        // vision_slots (prefill only, EP mode)
        flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<fbs::VisionSlotRef>>> vision_slots_off = 0;
        if (is_prefill && !seq->data_->vision_slots.empty()) {
            std::vector<flatbuffers::Offset<fbs::VisionSlotRef>> vs_offsets;
            vs_offsets.reserve(seq->data_->vision_slots.size());
            for (const auto& vs : seq->data_->vision_slots) {
                if (!vs)
                    continue;
                auto eid_off = builder.CreateString(vs->encoder_engine_id);
                vs_offsets.push_back(fbs::CreateVisionSlotRef(
                    builder, eid_off, vs->slot_idx, vs->num_tokens, vs->hidden_size, vs->max_tokens_per_slot));
            }
            if (!vs_offsets.empty()) {
                vision_slots_off = builder.CreateVector(vs_offsets);
            }
        }

        fbs::SequenceInputBuilder si_builder(builder);
        si_builder.add_master_group_id(ctx.master_group_id);
        si_builder.add_num_tokens(seq->num_tokens());
        si_builder.add_num_cached_tokens(seq->num_cached_tokens());
        si_builder.add_num_prompt_tokens(seq->num_prompt_tokens());
        si_builder.add_last_token(seq->last_token());
        if (is_prefill && token_ids_off.o != 0) {
            si_builder.add_token_ids(token_ids_off);
        }
        si_builder.add_group_block_table(group_bt_vec);
        si_builder.add_num_dispatched_tokens(ndt_vec);
        si_builder.add_state_slot(ctx.state_slot);

        // temperature from sampling_params
        auto sp = seq->sampling_params();
        si_builder.add_temperature(sp.temperature);

        if (vision_slots_off.o != 0) {
            si_builder.add_vision_slots(vision_slots_off);
        }

        seq_offsets.push_back(si_builder.Finish());
    }

    auto seqs_vec = builder.CreateVector(seq_offsets);
    auto batch    = fbs::CreateRunBatchInput(builder, seqs_vec, is_prefill);
    builder.Finish(batch);

    return builder.Release();
}

flatbuffers::DetachedBuffer serialize_migrate_batch(const std::vector<Sequence*>& seqs)
{
    flatbuffers::FlatBufferBuilder builder(4096);

    std::vector<flatbuffers::Offset<fbs::MigrateSequenceInput>> seq_offsets;
    seq_offsets.reserve(seqs.size());

    for (auto* seq : seqs) {
        auto& migrate_ctx = seq->block_ctx(BlockContextSlot::MIGRATE);
        auto& active_ctx  = seq->block_ctx(BlockContextSlot::ACTIVE);

        auto engine_id_off = builder.CreateString(migrate_ctx.engine_id);

        // migrate_block_location
        std::vector<fbs::BlockLocationPair> migrate_bl;
        migrate_bl.reserve(migrate_ctx.block_location.size());
        for (const auto& bl : migrate_ctx.block_location) {
            migrate_bl.emplace_back(bl.first(), bl.second());
        }
        auto migrate_bl_vec = builder.CreateVectorOfStructs(migrate_bl);

        // active_block_location
        std::vector<fbs::BlockLocationPair> active_bl;
        active_bl.reserve(active_ctx.block_location.size());
        for (const auto& bl : active_ctx.block_location) {
            active_bl.emplace_back(bl.first(), bl.second());
        }
        auto active_bl_vec = builder.CreateVectorOfStructs(active_bl);

        fbs::MigrateSequenceInputBuilder msi_builder(builder);
        msi_builder.add_seq_id(seq->seq_id());
        msi_builder.add_migrate_engine_id(engine_id_off);
        msi_builder.add_migrate_num_kvcache_blocks(migrate_ctx.num_kvcache_blocks);
        msi_builder.add_migrate_group_size(migrate_ctx.group_size);
        msi_builder.add_migrate_dp_idx(migrate_ctx.dp_idx);
        msi_builder.add_migrate_block_location(migrate_bl_vec);
        msi_builder.add_migrate_state_slot(migrate_ctx.state_slot);
        msi_builder.add_active_block_location(active_bl_vec);
        msi_builder.add_active_state_slot(active_ctx.state_slot);

        seq_offsets.push_back(msi_builder.Finish());
    }

    auto seqs_vec = builder.CreateVector(seq_offsets);
    auto batch    = fbs::CreateMigrateBatchInput(builder, seqs_vec);
    builder.Finish(batch);

    return builder.Release();
}

}  // namespace nanodeploy
