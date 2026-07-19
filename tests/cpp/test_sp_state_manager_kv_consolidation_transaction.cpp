#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nanodeploy/scheduler/sp_state_manager.h"

namespace {

using nanodeploy::BlockContext;
using nanodeploy::BlockContextSlot;
using nanodeploy::Sequence;
using nanodeploy::SequenceStatus;
using nanodeploy::SPStateManager;
using Plan = SPStateManager::LSKVConsolidationPlan;

static_assert(noexcept(std::declval<SPStateManager&>().commit_kv_consolidation(
    std::declval<const std::shared_ptr<Plan>&>())));
static_assert(noexcept(std::declval<SPStateManager&>().abort_kv_consolidation(
    std::declval<const std::shared_ptr<Plan>&>())));

void check(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template<typename Fn>
void expect_throw(Fn&& fn, const std::string& message)
{
    try {
        std::forward<Fn>(fn)();
    }
    catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

std::unique_ptr<SPStateManager> make_manager()
{
    auto manager = std::make_unique<SPStateManager>("prepared-kv-consolidation-cpu",
                                                     3,
                                                     8,
                                                     4,
                                                     16,
                                                     1024,
                                                     16,
                                                     0.0,
                                                     1,
                                                     false,
                                                     "legacy",
                                                     8192,
                                                     2,
                                                     false,
                                                     "",
                                                     1.0,
                                                     0.0,
                                                     1.0,
                                                     0.0,
                                                     1.0,
                                                     0.0,
                                                     1.0,
                                                     0.0,
                                                     1,
                                                     1,
                                                     1,
                                                     false,
                                                     "RoundRobin",
                                                     false,
                                                     0);
    manager->set_dp_idx(0);
    return manager;
}

std::shared_ptr<Sequence> admit_sequence(SPStateManager& manager)
{
    std::vector<int> tokens(9);
    for (int idx = 0; idx < static_cast<int>(tokens.size()); ++idx) {
        tokens[idx] = 100 + idx;
    }
    auto sequence       = std::make_shared<Sequence>(tokens, 1.0, 64, true);
    sequence->assigned_dp = 0;
    sequence->active("prepared-kv-consolidation-cpu", 3, 1);

    BlockContext placement = sequence->block_ctx(BlockContextSlot::ACTIVE);
    placement.dp_idx_                  = 0;
    placement.master_sp_idx_           = 0;
    placement.pending_token_present_   = false;
    placement.pending_token_target_sp_ = -1;
    placement.block_location.clear();
    placement.sp_block_table.assign(3, {});
    placement.num_dispatched_tokens = {4, 1, 4};

    auto initial = manager.prepare_ls_initial_batch({sequence}, {placement});
    check(initial.validate_precommit_noexcept(), "initial consolidation fixture did not validate");
    initial.commit_noexcept();
    sequence->append_token(0, BlockContextSlot::ACTIVE, 0);
    sequence->mark_last_token_pending(BlockContextSlot::ACTIVE, 0);
    sequence->status = SequenceStatus::RUNNING;
    manager.running.push_back(sequence);
    return sequence;
}

bool same_context(const BlockContext& lhs, const BlockContext& rhs)
{
    return lhs.getstate() == rhs.getstate();
}

void test_prepared_abort_restores_exact_allocator_and_context()
{
    auto manager  = make_manager();
    auto sequence = admit_sequence(*manager);
    const auto context_before = sequence->block_ctx(BlockContextSlot::ACTIVE);
    const auto free0_before   = manager->block_manager.at(0)->free_block_ids();
    const auto free1_before   = manager->block_manager.at(1)->free_block_ids();
    const auto free2_before   = manager->block_manager.at(2)->free_block_ids();
    const auto source_blocks  = context_before.sp_block_table[2];

    auto plan = manager->plan_kv_consolidation(1, 7, 0, {sequence}, 2, {0, 1});
    check(plan->success && plan->state == Plan::State::RESERVED,
          "consolidation plan was not RESERVED");
    check(same_context(sequence->block_ctx(BlockContextSlot::ACTIVE), context_before),
          "reservation published an ACTIVE context");
    check(manager->block_manager.at(1)->num_free_blocks()
              == static_cast<int>(free1_before.size()) - 1,
          "exact-capacity destination was not prepared");
    check(manager->block_manager.at(2)->free_block_ids() == free2_before,
          "prepared source release became visible before commit");
    expect_throw([&] { manager->block_manager.at(2)->release_blocks(source_blocks); },
                 "legacy source release bypassed the prepared transaction");

    manager->abort_kv_consolidation(plan);
    manager->abort_kv_consolidation(plan);
    check(plan->state == Plan::State::ABORTED, "prepared abort was not idempotent");
    check(same_context(sequence->block_ctx(BlockContextSlot::ACTIVE), context_before),
          "abort changed the stable ACTIVE context");
    check(manager->block_manager.at(0)->free_block_ids() == free0_before
              && manager->block_manager.at(1)->free_block_ids() == free1_before
              && manager->block_manager.at(2)->free_block_ids() == free2_before,
          "abort did not restore exact free-list order");
}

void test_capacity_order_and_noexcept_publication()
{
    auto manager  = make_manager();
    auto sequence = admit_sequence(*manager);
    const auto free1_before  = manager->block_manager.at(1)->free_block_ids();
    const auto free2_before  = manager->block_manager.at(2)->free_block_ids();
    const auto source_blocks = sequence->block_ctx(BlockContextSlot::ACTIVE).sp_block_table[2];

    auto plan = manager->plan_kv_consolidation(2, 8, 0, {sequence}, 2, {0, 1});
    check(plan->success, "capacity-order plan was rejected");
    check(!plan->moves.empty(), "capacity-order plan emitted no physical moves");
    check(std::all_of(plan->moves.begin(), plan->moves.end(), [](const auto& move) {
              return move.dst_sp_rank == 1;
          }),
          "destination order did not prefer greater exact capacity over the current master");

    plan->state = Plan::State::DISPATCHED;
    check(manager->commit_kv_consolidation(plan), "DISPATCHED consolidation did not commit");
    check(plan->state == Plan::State::COMMITTED, "commit did not publish COMMITTED");
    const auto& context = sequence->block_ctx(BlockContextSlot::ACTIVE);
    check(context.num_dispatched_tokens == std::vector<int>({5, 5, 0})
              && context.sp_block_table[2].empty(),
          "no-throw publication installed the wrong ACTIVE placement");
    check(manager->block_manager.at(1)->num_free_blocks()
              == static_cast<int>(free1_before.size()) - 1,
          "destination prepared allocation was not committed");
    check(manager->block_manager.at(2)->num_free_blocks()
              == static_cast<int>(free2_before.size() + source_blocks.size()),
          "prepared source release was not committed");
    check(manager->master_seq_count(0) == 1 && manager->num_recv_seqs_per_sp(1) == 1
              && manager->num_recv_seqs_per_sp(2) == 0,
          "precomputed role-counter shadow was not published");
}

void test_dispatched_stale_plan_remains_fail_closed()
{
    auto manager  = make_manager();
    auto sequence = admit_sequence(*manager);
    const auto free1_before = manager->block_manager.at(1)->free_block_ids();

    auto plan = manager->plan_kv_consolidation(3, 9, 0, {sequence}, 2, {0, 1});
    check(plan->success, "stale-plan fixture was rejected");
    plan->state = Plan::State::DISPATCHED;
    sequence->block_ctx(BlockContextSlot::ACTIVE).pending_token_target_sp_ = 1;

    check(!manager->commit_kv_consolidation(plan), "stale DISPATCHED plan committed");
    manager->abort_kv_consolidation(plan);
    check(plan->state == Plan::State::DISPATCHED,
          "post-dispatch abort released an ambiguous transaction");
    check(manager->block_manager.at(1)->num_free_blocks()
              == static_cast<int>(free1_before.size()) - 1,
          "post-dispatch failure released the destination reservation");
}

void test_external_dispatched_plan_outlives_manager()
{
    std::shared_ptr<Plan>              plan;
    std::weak_ptr<nanodeploy::BlockManager> source_manager;
    {
        auto manager  = make_manager();
        auto sequence = admit_sequence(*manager);
        source_manager = manager->block_manager.at(2);
        plan = manager->plan_kv_consolidation(4, 10, 0, {sequence}, 2, {0, 1});
        check(plan->success, "external-lifetime fixture was rejected");
        plan->state = Plan::State::DISPATCHED;
    }

    check(!source_manager.expired(),
          "DISPATCHED plan did not keep prepared BlockManagers alive after manager teardown");
    plan.reset();
    check(source_manager.expired(),
          "DISPATCHED plan retained BlockManagers after its prepared owners were destroyed");
}

struct TestCase {
    const char*           name;
    std::function<void()> run;
};

const std::vector<TestCase> test_cases = {
    {"prepared_abort_restores_exact_allocator_and_context",
     test_prepared_abort_restores_exact_allocator_and_context},
    {"capacity_order_and_noexcept_publication", test_capacity_order_and_noexcept_publication},
    {"dispatched_stale_plan_remains_fail_closed", test_dispatched_stale_plan_remains_fail_closed},
    {"external_dispatched_plan_outlives_manager", test_external_dispatched_plan_outlives_manager},
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "expected exactly one test-case name\n";
        return 2;
    }
    const std::string selected = argv[1];
    for (const auto& test_case : test_cases) {
        if (selected != test_case.name) {
            continue;
        }
        try {
            test_case.run();
        }
        catch (const std::exception& error) {
            std::cerr << selected << ": " << error.what() << '\n';
            return 1;
        }
        return 0;
    }
    std::cerr << "unknown test case: " << selected << '\n';
    return 2;
}
