#include <iostream>
#include <string>
#include <vector>

#include "nanodeploy/engine/simple_engine.h"
#include "nanodeploy/logging.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_json_path>" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    
    // Set up distributed environment (simulate 2 GPU if possible, or 1 GPU with EP=1)
    // For single GPU test, ensure config has num_experts but we might force EP=1
    
    nanodeploy::get_log_level() = 1;

    nanodeploy::SimpleEngine engine;
    try {
        // rank=0, world_size=1 (Single Card Test)
        // If testing EP, we'd need MPI or multi-process spawning. 
        // This test assumes single process for basic sanity.
        engine.init(config_path, 0, 1, 1, "127.0.0.1", 9000);
    } catch (const std::exception& e) {
        std::cerr << "Engine init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "MoE Runner Initialized." << std::endl;
    
    // Simple Generation
    std::vector<int> prompt_ids = {1, 2, 3, 4, 5}; 
    engine.add_request(prompt_ids, 10);

    while (!engine.is_finished()) {
        engine.step();
    }
    
    std::cout << "MoE Generation Finished." << std::endl;
    return 0;
}
