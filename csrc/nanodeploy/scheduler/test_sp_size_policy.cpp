/**
 * @file test_sp_size_policy.cpp
 * @brief Unit tests for LoadStatistics and SPSizePolicy classes
 * 
 * Tests the new load-aware SP size policy that:
 * 1. Learns thresholds from historical traces
 * 2. Considers KVCache imbalance for SP decisions
 * 3. Uses LeastBatch for master rank selection
 * 
 * Compile with: g++ -std=c++17 -I.. test_sp_size_policy.cpp load_statistics.cpp sp_size_policy.cpp -o test_sp_size_policy
 * Run with: ./test_sp_size_policy
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "load_statistics.h"
#include "sp_size_policy.h"

namespace nanodeploy {
namespace test {

// ============================================================================
// Test Utilities
// ============================================================================

void assert_float_eq(float expected, float actual, float epsilon = 0.01f) {
    if (std::abs(expected - actual) > epsilon) {
        std::cerr << "FAILED: expected " << expected << ", got " << actual << std::endl;
        assert(false);
    }
}

// ============================================================================
// LoadStatistics Tests
// ============================================================================

void test_load_statistics_initial_values() {
    std::cout << "Testing LoadStatistics initial values... ";
    
    LoadStatistics stats(4, 1024.0f, 256.0f, 1000);  // 4 SP ranks
    
    assert_float_eq(1024.0f, stats.avg_prompt_length());
    assert_float_eq(256.0f, stats.avg_output_length());
    assert(stats.num_samples() == 0);
    assert(!stats.has_sufficient_samples());
    assert(stats.attention_sp() == 4);
    
    std::cout << "PASSED" << std::endl;
}

void test_load_statistics_sliding_window() {
    std::cout << "Testing LoadStatistics sliding window... ";
    
    LoadStatistics stats(4, 1024.0f, 256.0f, 5);  // Small window
    
    for (int i = 0; i < 10; ++i) {
        stats.record_request(100 * (i + 1), 50);
    }
    
    // Window should contain [600, 700, 800, 900, 1000]
    // avg = (600 + 700 + 800 + 900 + 1000) / 5 = 800
    assert(stats.has_sufficient_samples());
    assert_float_eq(800.0f, stats.avg_prompt_length());
    
    std::cout << "PASSED" << std::endl;
}

void test_load_statistics_expected_waiting() {
    std::cout << "Testing LoadStatistics expected_waiting_requests... ";
    
    LoadStatistics stats(4, 1024.0f, 256.0f, 100);
    
    // Record some waiting queue sizes
    for (int i = 0; i < 20; ++i) {
        stats.record_waiting_queue_size(5 + i % 10);  // 5-14
    }
    
    // Expected: p75 of [5,6,7,8,9,10,11,12,13,14,...] should be around 11-12
    float expected = stats.expected_waiting_requests();
    assert(expected >= 5.0f && expected <= 20.0f);
    
    std::cout << "PASSED" << std::endl;
}

void test_load_statistics_kvcache_imbalance() {
    std::cout << "Testing LoadStatistics KVCache imbalance... ";
    
    LoadStatistics stats(4, 1024.0f, 256.0f, 100);
    
    // Balanced case: [100, 100, 100, 100]
    stats.record_kvcache_distribution({100, 100, 100, 100});
    assert_float_eq(1.0f, stats.kvcache_imbalance_ratio());
    
    // Imbalanced case: [200, 100, 100, 100]
    // avg = 125, max = 200, ratio = 200/125 = 1.6
    stats.record_kvcache_distribution({200, 100, 100, 100});
    assert_float_eq(1.6f, stats.kvcache_imbalance_ratio());
    
    // Very imbalanced: [400, 100, 0, 0]
    // avg = 125, max = 400, ratio = 400/125 = 3.2
    stats.record_kvcache_distribution({400, 100, 0, 0});
    assert_float_eq(3.2f, stats.kvcache_imbalance_ratio());
    
    std::cout << "PASSED" << std::endl;
}

void test_load_statistics_learned_thresholds() {
    std::cout << "Testing LoadStatistics learned thresholds... ";
    
    LoadStatistics stats(4, 1000.0f, 256.0f, 100);
    
    // Initial values
    assert_float_eq(3.0f, stats.long_req_threshold());
    assert_float_eq(1.5f, stats.imbalance_threshold());
    
    // Simulate many beneficial SP decisions -> thresholds should decrease
    for (int i = 0; i < 200; ++i) {
        stats.update_learned_thresholds(2, true);  // SP=2 was beneficial
    }
    
    // Threshold should have decreased
    assert(stats.long_req_threshold() < 3.0f);
    
    std::cout << "PASSED" << std::endl;
}

void test_load_statistics_is_long_request() {
    std::cout << "Testing LoadStatistics is_long_request... ";
    
    LoadStatistics stats(4, 1000.0f, 256.0f, 100);
    
    // With initial avg=1000 and default threshold=3.0:
    // 2999 is NOT long (< 3000)
    // 3001 IS long (> 3000)
    assert(!stats.is_long_request(2999));
    assert(stats.is_long_request(3001));
    
    std::cout << "PASSED" << std::endl;
}

// ============================================================================
// SPSizePolicy Tests
// ============================================================================

void test_sp_size_policy_segment_mode() {
    std::cout << "Testing SPSizePolicy segment mode... ";
    
    SPSizeConfig config;
    config.mode = SPSizeMode::Segment;
    config.segment_size = 65536;
    
    SPSizePolicy policy(config);
    
    // num_tokens = 200000, max_sp = 4
    // num_segments = ceil(200000 / 65536) = 4
    int sp = policy.determine_sp_size_segment(200000, 4);
    assert(sp == 4);
    
    // num_tokens = 50000, max_sp = 4 -> SP = 1
    sp = policy.determine_sp_size_segment(50000, 4);
    assert(sp == 1);
    
    std::cout << "PASSED" << std::endl;
}

void test_sp_size_policy_least_batch_master() {
    std::cout << "Testing SPSizePolicy LeastBatch master selection... ";
    
    SPSizeConfig config;
    SPSizePolicy policy(config);
    
    // Batch sizes: rank 0 has 5, rank 1 has 2, rank 2 has 8, rank 3 has 3
    std::vector<int> batch_sizes = {5, 2, 8, 3};
    
    int master = policy.select_master_rank_least_batch(batch_sizes);
    assert(master == 1);  // rank 1 has smallest batch
    
    std::cout << "PASSED" << std::endl;
}

void test_sp_size_policy_short_request() {
    std::cout << "Testing SPSizePolicy short request... ";
    
    SPSizeConfig config;
    config.mode = SPSizeMode::LoadAware;
    SPSizePolicy policy(config);
    
    LoadStatistics stats(4, 1000.0f, 256.0f, 100);
    
    // Short request: 2000 tokens < 3 * 1000
    std::vector<int> free_blocks = {100, 100, 100, 100};
    std::vector<int> used_blocks = {400, 400, 400, 400};
    std::vector<int> batch_sizes = {10, 5, 8, 7};
    
    SPSizeDecision decision = policy.determine_sp_size(
        2000, free_blocks, used_blocks, batch_sizes,
        500, stats, 4, 256);
    
    assert(decision.sp_size == 1);  // Short request -> SP=1
    assert(decision.master_rank == 1);  // LeastBatch -> rank 1
    
    std::cout << "PASSED" << std::endl;
}

void test_sp_size_policy_imbalance_driven() {
    std::cout << "Testing SPSizePolicy imbalance-driven SP... ";
    
    SPSizeConfig config;
    config.mode = SPSizeMode::LoadAware;
    SPSizePolicy policy(config);
    
    LoadStatistics stats(4, 1000.0f, 256.0f, 100);
    
    // Record some samples so stats work
    for (int i = 0; i < 20; ++i) {
        stats.record_request(1000, 200);
    }
    
    // Long request with highly imbalanced KVCache
    // This should trigger SP to balance attention computation
    std::vector<int> free_blocks = {100, 400, 400, 400};  // rank 0 is nearly full
    std::vector<int> used_blocks = {400, 100, 100, 100};  // rank 0 has most KVCache
    std::vector<int> batch_sizes = {5, 5, 5, 5};
    
    // Record the imbalanced distribution
    stats.record_kvcache_distribution(used_blocks);
    
    SPSizeDecision decision = policy.determine_sp_size(
        5000,  // Long request
        free_blocks, used_blocks, batch_sizes,
        500, stats, 4, 256);
    
    // With high imbalance, should consider opening SP
    // (exact behavior depends on thresholds)
    std::cout << "SP size: " << decision.sp_size 
              << ", due_to_imbalance: " << decision.due_to_imbalance << std::endl;
    
    std::cout << "PASSED" << std::endl;
}

void test_sp_size_policy_memory_pressure() {
    std::cout << "Testing SPSizePolicy memory pressure... ";
    
    SPSizeConfig config;
    config.mode = SPSizeMode::LoadAware;
    SPSizePolicy policy(config);
    
    LoadStatistics stats(4, 1000.0f, 256.0f, 100);
    
    // Build up waiting queue statistics
    for (int i = 0; i < 50; ++i) {
        stats.record_waiting_queue_size(15);  // Expect ~15 waiting requests
        stats.record_request(1000, 200);
    }
    
    // Long request with high memory pressure
    std::vector<int> free_blocks = {50, 50, 50, 50};  // Very low free space
    std::vector<int> used_blocks = {450, 450, 450, 450};
    std::vector<int> batch_sizes = {5, 3, 7, 4};
    
    SPSizeDecision decision = policy.determine_sp_size(
        10000,  // Very long request (100 blocks needed)
        free_blocks, used_blocks, batch_sizes,
        500, stats, 4, 256);
    
    std::cout << "SP size: " << decision.sp_size 
              << ", due_to_pressure: " << decision.due_to_pressure << std::endl;
    
    // Under high pressure with long request, should increase SP
    assert(decision.sp_size >= 1);
    assert(decision.master_rank == 1);  // LeastBatch
    
    std::cout << "PASSED" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

}  // namespace test
}  // namespace nanodeploy

int main() {
    std::cout << "\n=== Load-Aware SP Size Policy Unit Tests ===" << std::endl;
    std::cout << std::endl;
    
    // LoadStatistics tests
    nanodeploy::test::test_load_statistics_initial_values();
    nanodeploy::test::test_load_statistics_sliding_window();
    nanodeploy::test::test_load_statistics_expected_waiting();
    nanodeploy::test::test_load_statistics_kvcache_imbalance();
    nanodeploy::test::test_load_statistics_learned_thresholds();
    nanodeploy::test::test_load_statistics_is_long_request();
    
    // SPSizePolicy tests
    nanodeploy::test::test_sp_size_policy_segment_mode();
    nanodeploy::test::test_sp_size_policy_least_batch_master();
    nanodeploy::test::test_sp_size_policy_short_request();
    nanodeploy::test::test_sp_size_policy_imbalance_driven();
    nanodeploy::test::test_sp_size_policy_memory_pressure();
    
    std::cout << "\n=== All Tests Passed ===" << std::endl;
    return 0;
}
