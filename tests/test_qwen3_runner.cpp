#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "nanodeploy/engine/simple_engine.h"
#include "nanodeploy/logging.h"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_json_path> [prompt_id_1 prompt_id_2 ...]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];

    // Parse Args
    std::string agent_ip   = "127.0.0.1";
    int         agent_port = 9000;
    // Default to INFO (1) for clean output. Use 2 for DEBUG.
    nanodeploy::get_log_level() = 1;
    std::vector<int> prompt_ids;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--agent_ip" && i + 1 < argc) {
            agent_ip = argv[++i];
        }
        else if (arg == "--agent_port" && i + 1 < argc) {
            agent_port = std::stoi(argv[++i]);
        }
        else if (arg == "--log_level" && i + 1 < argc) {
            nanodeploy::get_log_level() = std::stoi(argv[++i]);
        }
        else {
            // Assume it's a prompt ID
            try {
                prompt_ids.push_back(std::stoi(arg));
            }
            catch (...) {
                // Ignore invalid args
            }
        }
    }

    if (prompt_ids.empty()) {
        // Default "Hello world!" -> [9707, 1879, 0]
        prompt_ids = {9707, 1879, 0};
        std::cout << "No prompt provided, using default: Hello world! (IDs: 9707 1879 0)" << std::endl;
    }

    nanodeploy::SimpleEngine engine;
    try {
        engine.init(config_path, 1, 1, 1, agent_ip, agent_port);
    }
    catch (const std::exception& e) {
        std::cerr << "Engine init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Input IDs: ";
    for (int id : prompt_ids)
        std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "Starting generation (Step-by-Step interaction)..." << std::endl;

    // 1. Add Request
    engine.add_request(prompt_ids, 20);

    std::vector<int> output_ids;

    // 2. Step Loop
    int step_count = 0;
    while (!engine.is_finished()) {
        step_count++;
        // std::cout << "  [Test] Executing Step " << step_count << "..." << std::endl;

        engine.step();

        // Optional: Add sleep or interaction logic here
    }
    std::cout << "\nGeneration Finished in " << step_count << " steps." << std::endl;

    // Normal return allows proper destructor calls for resource cleanup
    // Client destructor will auto-release unreleased gang allocations
    return 0;
}
