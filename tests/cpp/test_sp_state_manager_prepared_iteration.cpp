#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nanodeploy/scheduler/sp_state_manager.h"

namespace allocation_fault {
bool enabled = false;
}

void* operator new(std::size_t size)
{
    if (allocation_fault::enabled) {
        throw std::bad_alloc();
    }
    if (void* allocation = std::malloc(size == 0 ? 1 : size)) {
        return allocation;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void operator delete(void* allocation) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation) noexcept
{
    std::free(allocation);
}

void operator delete(void* allocation, std::size_t) noexcept
{
    std::free(allocation);
}

void operator delete[](void* allocation, std::size_t) noexcept
{
    std::free(allocation);
}

namespace {

using nanodeploy::BlockContext;
using nanodeploy::BlockContextSlot;
using nanodeploy::BlockManager;
using nanodeploy::Sequence;
using nanodeploy::SequenceStatus;
using nanodeploy::SPStateManager;
using IterationMutation = SPStateManager::PreparedLSIterationMasterPlan;
using RankRebalance     = BlockManager::PreparedBlockRebalance;

static_assert(!std::is_copy_constructible_v<IterationMutation>);
static_assert(!std::is_copy_assignable_v<IterationMutation>);
static_assert(std::is_nothrow_move_constructible_v<IterationMutation>);
static_assert(std::is_nothrow_move_assignable_v<IterationMutation>);
static_assert(std::is_nothrow_destructible_v<IterationMutation>);
static_assert(noexcept(std::declval<IterationMutation&>().commit_noexcept()));
static_assert(noexcept(std::declval<IterationMutation&>().abort_noexcept()));
static_assert(noexcept(std::declval<const SPStateManager&>().validate_ls_pool_step_composition_noexcept(
    static_cast<const SPStateManager::PreparedLSInitialBatch*>(nullptr),
    static_cast<const IterationMutation*>(nullptr))));
static_assert(!std::is_copy_constructible_v<RankRebalance>);
static_assert(!std::is_copy_assignable_v<RankRebalance>);
static_assert(std::is_nothrow_move_constructible_v<RankRebalance>);
static_assert(std::is_nothrow_move_assignable_v<RankRebalance>);
static_assert(std::is_nothrow_destructible_v<RankRebalance>);
static_assert(noexcept(std::declval<RankRebalance&>().commit_noexcept()));
static_assert(noexcept(std::declval<RankRebalance&>().abort_noexcept()));

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

std::unique_ptr<SPStateManager> make_manager(int blocks_per_rank)
{
    auto manager = std::make_unique<SPStateManager>("prepared-iteration-cpu",
                                                    2,
                                                    blocks_per_rank,
                                                    2,
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

std::shared_ptr<Sequence> make_sequence(int token_seed)
{
    auto sequence         = std::make_shared<Sequence>(std::vector<int>{token_seed, token_seed + 1}, 1.0, 64, true);
    sequence->assigned_dp = 0;
    sequence->active("prepared-iteration-cpu", 2, 1);
    return sequence;
}

void admit_with_pending_dummy(SPStateManager& manager, const std::shared_ptr<Sequence>& sequence, int master)
{
    BlockContext placement             = sequence->block_ctx(BlockContextSlot::ACTIVE);
    placement.dp_idx_                  = 0;
    placement.master_sp_idx_           = master;
    placement.pending_token_present_   = false;
    placement.pending_token_target_sp_ = -1;
    placement.block_location.clear();
    placement.sp_block_table.assign(2, {});
    placement.num_dispatched_tokens.assign(2, 0);
    placement.num_dispatched_tokens[master] = sequence->num_tokens;

    auto initial = manager.prepare_ls_initial_batch({sequence}, {placement});
    check(initial.validate_precommit_noexcept(), "initial setup failed precommit validation");
    initial.commit_noexcept();
    sequence->append_token(0, BlockContextSlot::ACTIVE, master);
    sequence->mark_last_token_pending(BlockContextSlot::ACTIVE, master);
    sequence->status = SequenceStatus::RUNNING;
    manager.running.push_back(sequence);
}

SPStateManager::LSDecodeMasterPlan make_plan(const std::vector<int>& targets)
{
    SPStateManager::LSDecodeMasterPlan plan;
    plan.success               = true;
    plan.allocation            = {0, 1};
    plan.sequence_master_ranks = targets;
    std::vector<int> load(2, 0);
    for (int target : targets) {
        load.at(static_cast<size_t>(target))++;
    }
    for (int rank = 0; rank < 2; ++rank) {
        if (load[rank] > 0) {
            plan.master_ranks.push_back(rank);
            plan.master_batch_sizes.push_back(load[rank]);
        }
    }
    return plan;
}

bool same_context(const BlockContext& lhs, const BlockContext& rhs)
{
    return lhs.getstate() == rhs.getstate();
}

void test_rank_local_same_rank_disjoint_transfer()
{
    BlockManager manager("rank-rebalance", 0, 2, 2);
    const auto   owned = manager.reserve_blocks(2);
    check(manager.num_free_blocks() == 0, "rank setup is not full");

    {
        auto transfer = manager.prepare_rebalance({owned[0]}, 1);
        check(transfer.allocation_block_ids() == std::vector<int>({owned[0]}),
              "same-rank transfer did not reuse the released ownership");
        check(manager.num_free_blocks() == 0 && manager.blocks()[owned[0]].ref_count == 1,
              "same-rank prepare published an intermediate free state");
    }
    check(manager.num_free_blocks() == 0 && manager.blocks()[owned[0]].ref_count == 1,
          "same-rank transfer destructor did not restore exact ownership");

    auto committed = manager.prepare_rebalance({owned[0]}, 1);
    committed.commit_noexcept();
    committed.commit_noexcept();
    committed.abort_noexcept();
    check(manager.num_free_blocks() == 0 && manager.blocks()[owned[0]].ref_count == 1,
          "idempotent same-rank transfer changed committed ownership");
    manager.release_blocks(owned);
}

void test_full_rank_swap_commits_atomically()
{
    auto manager = make_manager(3);  // one permanent dummy + two usable blocks per rank
    auto first   = make_sequence(100);
    auto second  = make_sequence(200);
    admit_with_pending_dummy(*manager, first, 0);
    admit_with_pending_dummy(*manager, second, 1);
    check(manager->block_manager.at(0)->num_free_blocks() == 0 && manager->block_manager.at(1)->num_free_blocks() == 0,
          "full-rank swap setup retained unexpected free headroom");

    const auto first_before    = first->block_ctx(BlockContextSlot::ACTIVE);
    const auto second_before   = second->block_ctx(BlockContextSlot::ACTIVE);
    const int  first_history   = first_before.sp_block_table[0][0];
    const int  first_frontier  = first_before.sp_block_table[0][1];
    const int  second_history  = second_before.sp_block_table[1][0];
    const int  second_frontier = second_before.sp_block_table[1][1];

    auto transaction = manager->prepare_iteration_master_plan({first, second}, make_plan({1, 0}));
    check(transaction.validate_precommit_noexcept(), "fresh full-rank swap did not validate");
    check(same_context(first->block_ctx(BlockContextSlot::ACTIVE), first_before)
              && same_context(second->block_ctx(BlockContextSlot::ACTIVE), second_before),
          "swap prepare modified a stable ACTIVE context");
    check(manager->block_manager.at(0)->num_free_blocks() == 0 && manager->block_manager.at(1)->num_free_blocks() == 0,
          "swap prepare exposed a pending-only block as free");
    expect_throw([&] { manager->block_manager.at(0)->release_blocks({first_frontier}); },
                 "legacy release bypassed a prepared iteration transfer");

    transaction.commit_noexcept();
    transaction.commit_noexcept();
    transaction.abort_noexcept();
    check(transaction.state() == IterationMutation::State::COMMITTED, "full-rank transaction did not remain COMMITTED");

    const auto& first_after  = first->block_ctx(BlockContextSlot::ACTIVE);
    const auto& second_after = second->block_ctx(BlockContextSlot::ACTIVE);
    check(first_after.master_sp_idx_ == 1 && first_after.pending_token_target_sp_ == 1
              && first_after.num_dispatched_tokens == std::vector<int>({2, 1}),
          "first sequence did not publish its new pending frontier");
    check(second_after.master_sp_idx_ == 0 && second_after.pending_token_target_sp_ == 0
              && second_after.num_dispatched_tokens == std::vector<int>({1, 2}),
          "second sequence did not publish its new pending frontier");
    check(first_after.sp_block_table[0] == BlockContext::BlockIdList({first_history})
              && first_after.sp_block_table[1] == BlockContext::BlockIdList({second_frontier}),
          "first sequence did not preserve history and receive the rank-local transfer");
    check(second_after.sp_block_table[1] == BlockContext::BlockIdList({second_history})
              && second_after.sp_block_table[0] == BlockContext::BlockIdList({first_frontier}),
          "second sequence did not preserve history and receive the rank-local transfer");
    check(manager->master_seq_count(0) == 1 && manager->master_seq_count(1) == 1
              && manager->num_recv_seqs_per_sp(0) == 1 && manager->num_recv_seqs_per_sp(1) == 1,
          "full-rank swap published incorrect role metadata");
    check(manager->block_manager.at(0)->num_free_blocks() == 0 && manager->block_manager.at(1)->num_free_blocks() == 0,
          "full-rank commit lost or duplicated physical ownership");
}

void test_abort_restores_exact_context_allocator_and_counters()
{
    auto manager  = make_manager(4);
    auto sequence = make_sequence(300);
    admit_with_pending_dummy(*manager, sequence, 0);

    const auto context_before = sequence->block_ctx(BlockContextSlot::ACTIVE);
    const auto free0_before   = manager->block_manager.at(0)->free_block_ids();
    const auto free1_before   = manager->block_manager.at(1)->free_block_ids();
    const int  running_before = manager->num_running_tokens();
    const int  master0_before = manager->master_seq_count(0);

    auto transaction = manager->prepare_iteration_master_plan({sequence}, make_plan({1}));
    check(manager->block_manager.at(1)->num_free_blocks() == static_cast<int>(free1_before.size()) - 1,
          "prepare did not reserve the destination block");
    check(manager->block_manager.at(0)->free_block_ids() == free0_before,
          "prepare published the source pending-only release");
    check(same_context(sequence->block_ctx(BlockContextSlot::ACTIVE), context_before),
          "prepare changed the stable context before abort");

    transaction.abort_noexcept();
    transaction.abort_noexcept();
    transaction.commit_noexcept();
    check(transaction.state() == IterationMutation::State::ABORTED,
          "aborted iteration transaction changed terminal state");
    check(same_context(sequence->block_ctx(BlockContextSlot::ACTIVE), context_before),
          "abort did not restore the exact stable context");
    check(manager->block_manager.at(0)->free_block_ids() == free0_before
              && manager->block_manager.at(1)->free_block_ids() == free1_before,
          "abort did not restore exact free-list order");
    check(manager->num_running_tokens() == running_before && manager->master_seq_count(0) == master0_before
              && manager->master_seq_count(1) == 0,
          "abort changed precomputed counters");
}

void test_combined_disjoint_groups_publish_once()
{
    auto manager = make_manager(3);
    // These sequences model two independent scheduler groups whose canonical
    // allocations are disjoint. The adapter intentionally receives only their
    // concatenated request/assignment vectors and allocation union.
    auto group_zero = make_sequence(500);
    auto group_one  = make_sequence(600);
    admit_with_pending_dummy(*manager, group_zero, 0);
    admit_with_pending_dummy(*manager, group_one, 1);
    const auto zero_before = group_zero->block_ctx(BlockContextSlot::ACTIVE);
    const auto one_before  = group_one->block_ctx(BlockContextSlot::ACTIVE);

    auto combined = manager->prepare_iteration_master_plan({group_zero, group_one}, make_plan({0, 1}));
    check(combined.validate_precommit_noexcept(),
          "combined disjoint-group plan did not validate on one global counter snapshot");
    check(manager->block_manager.at(0)->num_free_blocks() == 0 && manager->block_manager.at(1)->num_free_blocks() == 0,
          "combined disjoint-group prepare required free headroom");
    combined.abort_noexcept();

    // The same disjoint identity assignments remain valid when the scheduler
    // concatenates independent groups in reverse order.
    auto reversed = manager->prepare_iteration_master_plan({group_one, group_zero}, make_plan({1, 0}));
    check(reversed.validate_precommit_noexcept(),
          "reverse-order disjoint-group plan did not validate");
    reversed.commit_noexcept();

    check(same_context(group_zero->block_ctx(BlockContextSlot::ACTIVE), zero_before)
              && same_context(group_one->block_ctx(BlockContextSlot::ACTIVE), one_before),
          "combined identity assignments changed disjoint stable frontiers");
    check(manager->master_seq_count(0) == 1 && manager->master_seq_count(1) == 1
              && manager->num_recv_seqs_per_sp(0) == 0 && manager->num_recv_seqs_per_sp(1) == 0,
          "combined publication did not atomically preserve global role counters");
}

void test_stale_validation_move_and_destructor_abort()
{
    auto manager  = make_manager(4);
    auto sequence = make_sequence(400);
    admit_with_pending_dummy(*manager, sequence, 0);
    const auto free0_before = manager->block_manager.at(0)->free_block_ids();
    const auto free1_before = manager->block_manager.at(1)->free_block_ids();

    {
        auto              source = manager->prepare_iteration_master_plan({sequence}, make_plan({1}));
        IterationMutation moved(std::move(source));
        check(source.state() == IterationMutation::State::ABORTED, "iteration move source remained PREPARED");

        sequence->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ = 1;
        check(!moved.validate_precommit_noexcept(), "stale ACTIVE context was not detected");
        sequence->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ = 0;
        check(moved.validate_precommit_noexcept(), "restored ACTIVE context did not revalidate");

        manager->add_running_tokens(0, 1);
        check(!moved.validate_precommit_noexcept(), "stale running-token counter was not detected");
        manager->add_running_tokens(0, -1);
        check(moved.validate_precommit_noexcept(), "restored counters did not revalidate");
    }

    check(manager->block_manager.at(0)->free_block_ids() == free0_before
              && manager->block_manager.at(1)->free_block_ids() == free1_before,
          "moved PREPARED destructor did not abort exact allocator state");
    check(sequence->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ == 0
              && sequence->block_ctx(BlockContextSlot::ACTIVE).pending_token_target_sp_ == 0,
          "moved PREPARED destructor published a shadow frontier");
}

void test_initial_iteration_composition_is_allocation_free_and_preserves_counters()
{
    auto manager  = make_manager(8);
    auto running  = make_sequence(700);
    auto admitted = make_sequence(800);
    admit_with_pending_dummy(*manager, running, 0);

    BlockContext placement             = admitted->block_ctx(BlockContextSlot::ACTIVE);
    placement.dp_idx_                  = 0;
    placement.master_sp_idx_           = 0;
    placement.pending_token_present_   = false;
    placement.pending_token_target_sp_ = -1;
    placement.block_location.clear();
    placement.sp_block_table.assign(2, {});
    placement.num_dispatched_tokens.assign(2, 0);
    placement.num_dispatched_tokens[0] = admitted->num_tokens;

    auto initial   = manager->prepare_ls_initial_batch({admitted}, {placement});
    auto iteration = manager->prepare_iteration_master_plan({running}, make_plan({1}));
    check(initial.validate_precommit_noexcept() && iteration.validate_precommit_noexcept(),
          "composed transactions were not individually fresh");

    allocation_fault::enabled    = true;
    const bool composition_valid = manager->validate_ls_pool_step_composition_noexcept(&initial, &iteration);
    if (composition_valid) {
        initial.commit_noexcept();
        iteration.commit_noexcept();
    }
    allocation_fault::enabled = false;

    check(composition_valid, "allocation-free pool-step composition validation failed");
    check(initial.state() == SPStateManager::PreparedLSInitialBatch::State::COMMITTED
              && iteration.state() == IterationMutation::State::COMMITTED,
          "composed no-throw publication did not commit both components");
    check(manager->master_seq_count(0) == 1 && manager->master_seq_count(1) == 1,
          "iteration delta publication overwrote the admitted master counter");
    check(manager->num_running_seqs() == 2, "initial/iteration composition lost the admitted running-sequence counter");
    check(running->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ == 1
              && admitted->block_ctx(BlockContextSlot::ACTIVE).master_sp_idx_ == 0,
          "composed publication installed incorrect ACTIVE masters");
}

struct TestCase {
    const char*           name;
    std::function<void()> run;
};

const std::vector<TestCase> test_cases = {
    {"rank_local_same_rank_disjoint_transfer", test_rank_local_same_rank_disjoint_transfer},
    {"full_rank_swap_commits_atomically", test_full_rank_swap_commits_atomically},
    {"abort_restores_exact_context_allocator_and_counters", test_abort_restores_exact_context_allocator_and_counters},
    {"combined_disjoint_groups_publish_once", test_combined_disjoint_groups_publish_once},
    {"stale_validation_move_and_destructor_abort", test_stale_validation_move_and_destructor_abort},
    {"initial_iteration_composition_is_allocation_free_and_preserves_counters",
     test_initial_iteration_composition_is_allocation_free_and_preserves_counters},
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
