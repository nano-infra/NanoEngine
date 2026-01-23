#include "nanodeploy/csrc/sequence/sequence.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

using namespace nanodeploy;

// Mock block size (must match Sequence::block_size which is 64)
constexpr int BLOCK_SIZE = 64;

// Replicated Logic from ModelRunner::run for prefill
std::vector<int32_t> compute_prefill_slot_mapping(const std::shared_ptr<Sequence>& seq)
{
    std::vector<int32_t> slot_mapping_vec;
    auto&                blocks = seq->block_table();

    // Logic from ModelRunner::run
    for (int i = 0; i < seq->token_ids.size(); ++i) {
        if (blocks.empty()) {
            slot_mapping_vec.push_back(0);
            continue;
        }
        int block_idx_in_seq = i / BLOCK_SIZE;

        // Safety check simulated
        if (block_idx_in_seq >= (int)blocks.size()) {
            std::cerr << "Error: Block index out of bounds! Token " << i << " needs block index " << block_idx_in_seq
                      << " but only " << blocks.size() << " blocks available." << std::endl;
            // For test purposes, push -1 or similar to indicate failure
            slot_mapping_vec.push_back(-1);
            continue;
        }

        int block_id = blocks[block_idx_in_seq];
        int offset   = i % BLOCK_SIZE;
        slot_mapping_vec.push_back(block_id * BLOCK_SIZE + offset);
    }
    return slot_mapping_vec;
}

// Replicated Logic from ModelRunner::run for decode
std::vector<int32_t> compute_decode_slot_mapping(const std::shared_ptr<Sequence>& seq)
{
    std::vector<int32_t> slot_mapping_vec;
    auto&                blocks = seq->block_table();

    if (seq->token_ids.empty())
        return {};

    int seq_len = seq->num_tokens;
    int pos     = seq_len - 1;  // Last token is the new one for decode

    if (blocks.empty()) {
        slot_mapping_vec.push_back(0);
    }
    else {
        int block_in_seq = pos / BLOCK_SIZE;
        if (block_in_seq >= (int)blocks.size())
            block_in_seq = blocks.size() - 1;

        int block_idx = blocks[block_in_seq];
        int offset    = pos % BLOCK_SIZE;
        slot_mapping_vec.push_back(block_idx * BLOCK_SIZE + offset);
    }
    return slot_mapping_vec;
}

class SlotMappingTest: public ::testing::Test {
protected:
    void SetUp() override
    {
        // Common setup if needed
    }
};

TEST_F(SlotMappingTest, PrefillSimple)
{
    // 10 tokens
    std::vector<int> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto             seq    = std::make_shared<Sequence>(tokens);
    seq->active("test", 1, 1);  // Initialize context

    // Assign Block 100
    seq->block_table().push_back(100);

    auto mapping = compute_prefill_slot_mapping(seq);

    ASSERT_EQ(mapping.size(), 10);
    for (int i = 0; i < 10; ++i) {
        int expected = 100 * BLOCK_SIZE + i;
        EXPECT_EQ(mapping[i], expected) << "Mismatch at " << i;
    }
}

TEST_F(SlotMappingTest, PrefillMultiBlock)
{
    // 70 tokens (64 + 6)
    std::vector<int> tokens;
    for (int i = 0; i < 70; ++i)
        tokens.push_back(i);
    auto seq = std::make_shared<Sequence>(tokens);
    seq->active("test", 1, 1);  // Initialize context

    // Assign Block 100, 101
    seq->block_table().push_back(100);
    seq->block_table().push_back(101);

    auto mapping = compute_prefill_slot_mapping(seq);

    ASSERT_EQ(mapping.size(), 70);
    // First 64 tokens -> Block 100
    for (int i = 0; i < 64; ++i) {
        int expected = 100 * BLOCK_SIZE + i;
        EXPECT_EQ(mapping[i], expected);
    }
    // Next 6 tokens -> Block 101
    for (int i = 64; i < 70; ++i) {
        int expected = 101 * BLOCK_SIZE + (i - 64);
        EXPECT_EQ(mapping[i], expected);
    }
}

TEST_F(SlotMappingTest, DecodeSimple)
{
    // Initialize with 10 tokens (Prefill done)
    std::vector<int> tokens = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto             seq    = std::make_shared<Sequence>(tokens);
    seq->active("test", 1, 1);  // Initialize context
    seq->block_table().push_back(100);

    // Simulate append token (Decode step 1)
    seq->append_token(999);
    // Now seq len is 11. pos is 10.

    auto mapping = compute_decode_slot_mapping(seq);
    ASSERT_EQ(mapping.size(), 1);
    int expected = 100 * BLOCK_SIZE + 10;
    EXPECT_EQ(mapping[0], expected);
}

TEST_F(SlotMappingTest, DecodeBoundaryCrossing)
{
    // Initialize with 64 tokens (Full block)
    std::vector<int> tokens;
    for (int i = 0; i < 64; ++i)
        tokens.push_back(i);
    auto seq = std::make_shared<Sequence>(tokens);
    seq->active("test", 1, 1);  // Initialize context
    seq->block_table().push_back(100);

    // Append 65th token. Should need new block.
    // Simulate Scheduler adding a block
    seq->block_table().push_back(101);
    seq->append_token(999);

    // seq len 65. pos 64.
    auto mapping = compute_decode_slot_mapping(seq);

    ASSERT_EQ(mapping.size(), 1);
    // pos 64 -> block index 1, offset 0 -> Block 101, offset 0
    int expected = 101 * BLOCK_SIZE + 0;

    EXPECT_EQ(mapping[0], expected);
}

TEST_F(SlotMappingTest, PrefillMissingBlock)
{
    // 70 tokens but only 1 block provided
    std::vector<int> tokens;
    for (int i = 0; i < 70; ++i)
        tokens.push_back(i);
    auto seq = std::make_shared<Sequence>(tokens);
    seq->active("test", 1, 1);  // Initialize context

    // Only Block 100 (capacity 64)
    seq->block_table().push_back(100);  // Index 0

    // This should technically fail or warn if logic is correct
    auto mapping = compute_prefill_slot_mapping(seq);

    // We check if the logic detects the missing block
    ASSERT_GT(mapping.size(), 64);
    EXPECT_EQ(mapping[64], -1) << "Expected failure marker -1 for missing block";
}
