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
using InitialMutation = SPStateManager::PreparedLSInitialBatch;
using ReleaseMutation = SPStateManager::PreparedLSRelease;

static_assert(!std::is_copy_constructible_v<InitialMutation>);
static_assert(!std::is_copy_assignable_v<InitialMutation>);
static_assert(std::is_nothrow_move_constructible_v<InitialMutation>);
static_assert(std::is_nothrow_move_assignable_v<InitialMutation>);
static_assert(std::is_nothrow_destructible_v<InitialMutation>);
static_assert(noexcept(std::declval<InitialMutation&>().commit_noexcept()));
static_assert(noexcept(std::declval<InitialMutation&>().abort_noexcept()));
static_assert(!std::is_copy_constructible_v<ReleaseMutation>);
static_assert(!std::is_copy_assignable_v<ReleaseMutation>);
static_assert(std::is_nothrow_move_constructible_v<ReleaseMutation>);
static_assert(std::is_nothrow_move_assignable_v<ReleaseMutation>);
static_assert(std::is_nothrow_destructible_v<ReleaseMutation>);
static_assert(noexcept(std::declval<ReleaseMutation&>().commit_noexcept()));
static_assert(noexcept(std::declval<ReleaseMutation&>().abort_noexcept()));

void check(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::unique_ptr<SPStateManager> make_manager()
{
    auto manager = std::make_unique<SPStateManager>("prepared-sp-cpu",
                                                     2,
                                                     6,
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

std::shared_ptr<Sequence> make_sequence(int token_count)
{
    std::vector<int> tokens;
    tokens.reserve(token_count);
    for (int idx = 0; idx < token_count; ++idx) {
        tokens.push_back(100 + idx);
    }
    auto sequence = std::make_shared<Sequence>(tokens, 1.0, 64, true);
    sequence->assigned_dp = 0;
    sequence->active("prepared-sp-cpu", 2, 1);
    return sequence;
}

BlockContext placement_for(const std::shared_ptr<Sequence>& sequence, int master, int rank0_tokens)
{
    BlockContext placement = sequence->block_ctx(BlockContextSlot::ACTIVE);
    placement.dp_idx_       = 0;
    placement.master_sp_idx_ = master;
    placement.pending_token_present_   = false;
    placement.pending_token_target_sp_ = -1;
    placement.block_location.clear();
    placement.sp_block_table.assign(2, {});
    placement.num_dispatched_tokens = {rank0_tokens, sequence->num_tokens - rank0_tokens};
    return placement;
}

void test_initial_abort_restores_exact_state()
{
    auto manager  = make_manager();
    auto sequence = make_sequence(4);
    auto placement = placement_for(sequence, 0, 4);
    const auto free0_before = manager->block_manager.at(0)->free_block_ids();
    const auto free1_before = manager->block_manager.at(1)->free_block_ids();
    const auto stable       = sequence->block_ctx(BlockContextSlot::ACTIVE);

    auto mutation = manager->prepare_ls_initial_batch({sequence}, {placement});
    check(mutation.state() == InitialMutation::State::PREPARED, "initial mutation was not PREPARED");
    check(manager->block_manager.at(0)->num_free_blocks() == static_cast<int>(free0_before.size()) - 2,
          "master did not reserve prompt plus dummy headroom exactly");
    check(manager->block_manager.at(1)->free_block_ids() == free1_before,
          "unused rank changed during initial prepare");
    check(sequence->block_ctx(BlockContextSlot::ACTIVE).num_dispatched_tokens == stable.num_dispatched_tokens
              && sequence->block_ctx(BlockContextSlot::ACTIVE).block_location.empty(),
          "initial prepare modified stable ACTIVE context");
    check(manager->num_running_seqs() == 0 && manager->num_running_tokens() == 0,
          "initial prepare published counters");
    check(mutation.validate_precommit_noexcept(), "fresh initial mutation failed precommit validation");

    mutation.abort_noexcept();
    check(mutation.state() == InitialMutation::State::ABORTED, "initial abort did not publish ABORTED");
    check(manager->block_manager.at(0)->free_block_ids() == free0_before
              && manager->block_manager.at(1)->free_block_ids() == free1_before,
          "initial abort did not restore exact free lists");
    check(manager->num_running_seqs() == 0 && manager->num_running_tokens() == 0,
          "initial abort changed counters");
}

void test_initial_commit_and_prepared_release()
{
    auto manager   = make_manager();
    auto sequence  = make_sequence(7);
    auto placement = placement_for(sequence, 0, 4);
    const auto free0_before = manager->block_manager.at(0)->free_block_ids();
    const auto free1_before = manager->block_manager.at(1)->free_block_ids();

    auto initial = manager->prepare_ls_initial_batch({sequence}, {placement});
    initial.commit_noexcept();
    check(initial.state() == InitialMutation::State::COMMITTED, "initial commit did not publish COMMITTED");
    const auto& committed = sequence->block_ctx(BlockContextSlot::ACTIVE);
    check(committed.num_dispatched_tokens == std::vector<int>({4, 3}),
          "initial commit changed prompt-only dispatched counts");
    check(committed.sp_block_table[0].size() == 2 && committed.sp_block_table[1].size() == 1,
          "initial commit did not attach exact prompt/dummy block tables");
    check(committed.block_location.size() == 3, "initial commit attached incomplete block locations");
    check(manager->num_running_seqs() == 1 && manager->num_running_tokens() == 8,
          "initial commit did not include exactly one future dummy in running counters");
    check(manager->master_seq_count(0) == 1 && manager->num_recv_seqs_per_sp(1) == 1,
          "initial commit published wrong role counters");

    // The scheduler's no-throw publication tail performs only these logical
    // updates. No may_append/add_running_tokens call is permitted or needed.
    sequence->append_token(0, BlockContextSlot::ACTIVE, 0);
    sequence->mark_last_token_pending(BlockContextSlot::ACTIVE, 0);
    sequence->status = SequenceStatus::RUNNING;
    check(sequence->num_tokens == manager->num_running_tokens(),
          "dummy append double-counted or missed running-token accounting");

    auto release = manager->prepare_ls_release(sequence, BlockContextSlot::ACTIVE);
    check(release.validate_precommit_noexcept(), "fresh release failed precommit validation");
    check(manager->num_running_seqs() == 1 && manager->num_running_tokens() == 8,
          "release prepare published counter deltas");
    check(sequence->block_ctx(BlockContextSlot::ACTIVE).block_location.size() == 3,
          "release prepare modified stable ACTIVE context");
    release.commit_noexcept();
    check(release.state() == ReleaseMutation::State::COMMITTED, "release commit did not publish COMMITTED");
    const auto& empty = sequence->block_ctx(BlockContextSlot::ACTIVE);
    check(empty.block_location.empty() && empty.sp_block_table[0].empty() && empty.sp_block_table[1].empty(),
          "release commit did not install a fresh ACTIVE context");
    check(empty.dp_idx_ == 0, "release commit lost persistent DP assignment");
    check(empty.master_sp_idx_ == -1 && !empty.pending_token_present_
              && empty.pending_token_target_sp_ == -1
              && std::all_of(empty.num_dispatched_tokens.begin(), empty.num_dispatched_tokens.end(), [](int value) {
                     return value == 0;
                 }),
          "release commit left an active or pending Decode role in the empty context");
    check(manager->num_running_seqs() == 0 && manager->num_running_tokens() == 0
              && manager->master_seq_count(0) == 0 && manager->num_recv_seqs_per_sp(1) == 0,
          "release commit published wrong counter deltas");
    check(manager->block_manager.at(0)->free_block_ids().size() == free0_before.size()
              && manager->block_manager.at(1)->free_block_ids().size() == free1_before.size(),
          "release commit did not return every exact block");
}

void test_raii_move_and_stale_validation()
{
    auto manager   = make_manager();
    auto sequence  = make_sequence(3);
    auto placement = placement_for(sequence, 1, 0);
    const auto free_before = manager->block_manager.at(1)->free_block_ids();

    {
        auto source = manager->prepare_ls_initial_batch({sequence}, {placement});
        InitialMutation destination(std::move(source));
        check(source.state() == InitialMutation::State::ABORTED, "initial move source remained PREPARED");
        sequence->status = SequenceStatus::PAUSED_OFFLOAD;
        check(!destination.validate_precommit_noexcept(), "stale status was not detected before commit");
        sequence->status = SequenceStatus::WAITING;
        check(destination.validate_precommit_noexcept(), "restored stable state did not revalidate");
    }
    check(manager->block_manager.at(1)->free_block_ids() == free_before,
          "initial PREPARED destructor did not abort exactly");
    check(manager->num_running_seqs() == 0 && manager->num_running_tokens() == 0,
          "RAII initial abort changed counters");
}

void test_disjoint_initial_prepare_abort_then_commit()
{
    auto manager = make_manager();
    auto finished_during_bootstrap = make_sequence(4);
    auto survivor                  = make_sequence(3);
    auto finished_placement       = placement_for(finished_during_bootstrap, 0, 4);
    auto survivor_placement       = placement_for(survivor, 0, 3);

    // Pool-step preparation may hold multiple isolated reservations that all
    // snapshot the same stable counters. Aborting a bootstrap-finished member
    // must leave the survivor transaction valid and its physical IDs reserved.
    auto finished_reservation =
        manager->prepare_ls_initial_batch({finished_during_bootstrap}, {finished_placement});
    auto survivor_reservation = manager->prepare_ls_initial_batch({survivor}, {survivor_placement});
    check(finished_reservation.validate_precommit_noexcept()
              && survivor_reservation.validate_precommit_noexcept(),
          "disjoint initial reservations did not coexist on one stable baseline");

    finished_reservation.abort_noexcept();
    check(survivor_reservation.validate_precommit_noexcept(),
          "aborting a disjoint bootstrap-finished reservation invalidated survivors");
    survivor_reservation.commit_noexcept();
    check(manager->num_running_seqs() == 1 && manager->num_running_tokens() == 4,
          "survivor commit used another transaction's counter delta");
    const auto& context = survivor->block_ctx(BlockContextSlot::ACTIVE);
    check(context.sp_block_table[0].size() == 1 && context.sp_block_table[1].empty(),
          "survivor commit attached blocks from the aborted reservation");
}

struct TestCase {
    const char*           name;
    std::function<void()> run;
};

const std::vector<TestCase> test_cases = {
    {"initial_abort_restores_exact_state", test_initial_abort_restores_exact_state},
    {"initial_commit_and_prepared_release", test_initial_commit_and_prepared_release},
    {"raii_move_and_stale_validation", test_raii_move_and_stale_validation},
    {"disjoint_initial_prepare_abort_then_commit", test_disjoint_initial_prepare_abort_then_commit},
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
