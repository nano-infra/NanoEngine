#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "nanodeploy/scheduler/block_manager.h"

namespace {

using nanodeploy::Block;
using nanodeploy::BlockManager;
using Mutation = BlockManager::PreparedBlockMutation;

static_assert(!std::is_copy_constructible_v<Mutation>);
static_assert(!std::is_copy_assignable_v<Mutation>);
static_assert(std::is_nothrow_move_constructible_v<Mutation>);
static_assert(std::is_nothrow_move_assignable_v<Mutation>);
static_assert(std::is_nothrow_destructible_v<Mutation>);
static_assert(noexcept(std::declval<Mutation&>().commit_noexcept()));
static_assert(noexcept(std::declval<Mutation&>().abort_noexcept()));

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

void check_ids(const std::vector<int>& actual, const std::vector<int>& expected, const std::string& message)
{
    if (actual != expected) {
        throw std::runtime_error(message);
    }
}

Block& mutable_block(BlockManager& manager, int block_id)
{
    return const_cast<Block&>(manager.blocks().at(static_cast<size_t>(block_id)));
}

void test_allocate_abort_and_destructor()
{
    BlockManager manager("cpu-test", 0, 6, 4);
    const auto   original_free = manager.free_block_ids();

    auto mutation = manager.prepare_allocate_uncached(2);
    check(mutation.state() == Mutation::State::PREPARED, "allocation was not prepared");
    check_ids(mutation.block_ids(), {0, 1}, "count allocation did not follow free-list order");
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "prepared blocks remained visible as free");
    check(manager.blocks()[0].ref_count == 1 && manager.blocks()[1].ref_count == 1,
          "prepared allocation did not reserve ownership");

    mutation.abort_noexcept();
    check(mutation.state() == Mutation::State::ABORTED, "allocation abort did not publish ABORTED");
    check_ids(manager.free_block_ids(), original_free, "allocation abort did not restore free-list order");
    check(manager.blocks()[0].ref_count == 0 && manager.blocks()[1].ref_count == 0,
          "allocation abort did not restore refcounts");
    mutation.abort_noexcept();
    mutation.commit_noexcept();
    check_ids(manager.free_block_ids(), original_free, "idempotent abort changed allocator state");

    {
        auto auto_abort = manager.prepare_allocate_uncached(std::vector<int>{2, 0});
        check_ids(auto_abort.block_ids(), {2, 0}, "exact allocation did not preserve caller order");
        check_ids(manager.free_block_ids(), {1, 3, 4, 5}, "exact preparation removed wrong free IDs");
    }
    check_ids(manager.free_block_ids(), original_free, "PREPARED destructor did not restore exact state");
}

void test_allocate_commit()
{
    BlockManager manager("cpu-test", 0, 6, 4);

    auto mutation = manager.prepare_allocate_uncached(2);
    mutation.commit_noexcept();
    check(mutation.state() == Mutation::State::COMMITTED, "allocation commit did not publish COMMITTED");
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "allocation commit returned reserved IDs to free list");
    check(manager.blocks()[0].ref_count == 1 && manager.blocks()[1].ref_count == 1, "allocation commit lost ownership");

    mutation.commit_noexcept();
    mutation.abort_noexcept();
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "idempotent allocation commit changed state");

    manager.release_blocks({0, 1});
    check_ids(manager.free_block_ids(), {2, 3, 4, 5, 0, 1}, "committed IDs were not releasable");
}

void test_exact_allocate_snapshot_restore()
{
    BlockManager manager("cpu-test", 0, 6, 4);
    Block&       cached = mutable_block(manager, 3);
    cached.hash         = 991;
    cached.token_ids    = {7, 8, 9};

    auto mutation = manager.prepare_allocate_uncached(std::vector<int>{3, 1});
    check_ids(mutation.block_ids(), {3, 1}, "exact allocation changed ID order");
    check_ids(manager.free_block_ids(), {0, 2, 4, 5}, "exact allocation changed survivor order");
    check(manager.blocks()[3].ref_count == 1 && manager.blocks()[3].hash == -1 && manager.blocks()[3].token_ids.empty(),
          "exact allocation did not reset reserved block contents");

    mutation.abort_noexcept();
    check_ids(manager.free_block_ids(), {0, 1, 2, 3, 4, 5}, "exact abort did not restore free-list order");
    check(manager.blocks()[3].ref_count == 0 && manager.blocks()[3].hash == 991
              && manager.blocks()[3].token_ids == std::vector<int>({7, 8, 9}),
          "exact abort did not restore cached block contents");
}

void test_release_abort_and_destructor()
{
    BlockManager manager("cpu-test", 0, 6, 4);
    const auto   owned = manager.reserve_blocks(2);
    check_ids(owned, {0, 1}, "reserve_blocks setup returned unexpected IDs");

    auto mutation = manager.prepare_release({1, 0});
    check(mutation.state() == Mutation::State::PREPARED, "release was not prepared");
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "release prepare published free nodes early");
    check(manager.blocks()[0].ref_count == 1 && manager.blocks()[1].ref_count == 1,
          "release prepare changed ownership early");
    mutation.abort_noexcept();
    check(manager.blocks()[0].ref_count == 1 && manager.blocks()[1].ref_count == 1, "release abort changed ownership");
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "release abort changed free-list state");

    {
        auto auto_abort = manager.prepare_release({0, 1});
        check(auto_abort.state() == Mutation::State::PREPARED, "second release was not prepared");
    }
    check(manager.blocks()[0].ref_count == 1 && manager.blocks()[1].ref_count == 1,
          "release PREPARED destructor changed ownership");

    manager.release_blocks(owned);
    check_ids(manager.free_block_ids(), {2, 3, 4, 5, 0, 1}, "release abort left blocks guarded");
}

void test_release_commit_and_shared_reference()
{
    BlockManager manager("cpu-test", 0, 6, 4);
    manager.reserve_blocks(2);

    auto mutation = manager.prepare_release({1, 0});
    mutation.commit_noexcept();
    check(mutation.state() == Mutation::State::COMMITTED, "release commit did not publish COMMITTED");
    check(manager.blocks()[0].ref_count == 0 && manager.blocks()[1].ref_count == 0,
          "release commit did not drop ownership");
    check_ids(manager.free_block_ids(), {2, 3, 4, 5, 1, 0}, "release commit published nodes in wrong order");
    mutation.commit_noexcept();
    mutation.abort_noexcept();
    check_ids(manager.free_block_ids(), {2, 3, 4, 5, 1, 0}, "idempotent release commit changed state");

    BlockManager shared_manager("cpu-test", 0, 3, 4);
    shared_manager.reserve_blocks(1);
    mutable_block(shared_manager, 0).ref_count = 2;
    auto release_one                           = shared_manager.prepare_release({0});
    release_one.commit_noexcept();
    check(shared_manager.blocks()[0].ref_count == 1, "shared release dropped more than one reference");
    check_ids(shared_manager.free_block_ids(), {1, 2}, "shared block became free too early");

    auto release_two = shared_manager.prepare_release({0});
    release_two.commit_noexcept();
    check(shared_manager.blocks()[0].ref_count == 0, "last shared reference was not released");
    check_ids(shared_manager.free_block_ids(), {1, 2, 0}, "last shared release did not publish free node");
}

void test_move_ownership_and_disjoint_preparations()
{
    BlockManager manager("cpu-test", 0, 6, 4);

    auto first  = manager.prepare_allocate_uncached(std::vector<int>{0});
    auto second = manager.prepare_allocate_uncached(std::vector<int>{1});
    check_ids(manager.free_block_ids(), {2, 3, 4, 5}, "disjoint prepares removed wrong IDs");

    first = std::move(second);
    check(second.state() == Mutation::State::ABORTED, "move source remained PREPARED");
    check_ids(first.block_ids(), {1}, "move assignment did not transfer ownership");
    check_ids(manager.free_block_ids(), {0, 2, 3, 4, 5}, "move assignment did not abort destination first");
    first.commit_noexcept();
    manager.release_blocks({1});
    check_ids(manager.free_block_ids(), {0, 2, 3, 4, 5, 1}, "moved transaction did not commit cleanly");

    auto     source = manager.prepare_allocate_uncached(std::vector<int>{2});
    Mutation destination(std::move(source));
    check(source.state() == Mutation::State::ABORTED, "move constructor source remained PREPARED");
    destination.abort_noexcept();
    check_ids(manager.free_block_ids(), {0, 2, 3, 4, 5, 1}, "moved abort did not restore stable order");
}

void test_validation_and_prepared_guards()
{
    BlockManager manager("cpu-test", 0, 6, 4);
    const auto   original_free = manager.free_block_ids();

    expect_throw([&] { (void)manager.prepare_allocate_uncached(-1); }, "negative allocation count was accepted");
    expect_throw([&] { (void)manager.prepare_allocate_uncached(7); }, "oversized allocation count was accepted");
    expect_throw([&] { (void)manager.prepare_allocate_uncached(std::vector<int>{-1}); },
                 "negative exact ID was accepted");
    expect_throw([&] { (void)manager.prepare_allocate_uncached(std::vector<int>{6}); },
                 "out-of-range exact ID was accepted");
    expect_throw(
        [&] {
            (void)manager.prepare_allocate_uncached(std::vector<int>{2, 2});
        },
        "duplicate exact IDs were accepted");
    check_ids(manager.free_block_ids(), original_free, "failed allocation prepare changed allocator state");

    manager.reserve_blocks(1);
    expect_throw([&] { (void)manager.prepare_allocate_uncached(std::vector<int>{0}); },
                 "owned block was accepted for allocation");
    expect_throw([&] { (void)manager.prepare_release(std::vector<int>{1}); }, "free block was accepted for release");
    expect_throw([&] { (void)manager.prepare_release(std::vector<int>{0, 0}); }, "duplicate release IDs were accepted");
    expect_throw([&] { (void)manager.prepare_release(std::vector<int>{-1}); }, "negative release ID was accepted");
    expect_throw([&] { (void)manager.prepare_release(std::vector<int>{6}); }, "out-of-range release ID was accepted");

    auto release = manager.prepare_release({0});
    expect_throw([&] { manager.release_blocks({0}); }, "legacy release bypassed prepared ownership guard");
    expect_throw([&] { (void)manager.prepare_release(std::vector<int>{0}); },
                 "second prepared release acquired the same block");
    expect_throw([&] { (void)manager.prepare_allocate_uncached(std::vector<int>{0}); },
                 "prepared release block was accepted for allocation");
    check(manager.blocks()[0].ref_count == 1, "guard rejection changed prepared release refcount");
    release.abort_noexcept();

    auto allocation = manager.prepare_allocate_uncached(std::vector<int>{1});
    expect_throw([&] { manager.release_blocks({1}); }, "legacy release bypassed prepared allocation guard");
    expect_throw([&] { (void)manager.prepare_allocate_uncached(std::vector<int>{1}); },
                 "second prepared allocation acquired the same block");
    allocation.abort_noexcept();

    manager.release_blocks({0});
    check_ids(manager.free_block_ids(), {1, 2, 3, 4, 5, 0}, "guards left allocator in a non-retryable state");

    auto empty_allocate = manager.prepare_allocate_uncached(0);
    empty_allocate.commit_noexcept();
    auto empty_release = manager.prepare_release({});
    empty_release.abort_noexcept();
    check_ids(manager.free_block_ids(), {1, 2, 3, 4, 5, 0}, "empty mutations changed allocator state");
}

struct TestCase {
    const char*           name;
    std::function<void()> run;
};

const std::vector<TestCase> test_cases = {
    {"allocate_abort_and_destructor", test_allocate_abort_and_destructor},
    {"allocate_commit", test_allocate_commit},
    {"exact_allocate_snapshot_restore", test_exact_allocate_snapshot_restore},
    {"release_abort_and_destructor", test_release_abort_and_destructor},
    {"release_commit_and_shared_reference", test_release_commit_and_shared_reference},
    {"move_ownership_and_disjoint_preparations", test_move_ownership_and_disjoint_preparations},
    {"validation_and_prepared_guards", test_validation_and_prepared_guards},
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
