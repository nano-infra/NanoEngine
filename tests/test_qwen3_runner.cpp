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
        engine.init(config_path, agent_ip, agent_port);
    }
    catch (const std::exception& e) {
        std::cerr << "Engine init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Input IDs: ";
    for (int id : prompt_ids)
        std::cout << id << " ";
    std::cout << std::endl;

    std::cout << "Generating response..." << std::endl;
    std::vector<int> output_ids = engine.generate(prompt_ids, 20);

    std::cout << "Generated IDs: ";
    for (int id : output_ids)
        std::cout << id << " ";
    std::cout << std::endl;

    // Save outputs to docs for reference
    fs::path docs_dir = fs::path("d:/src/NanoDeploy/docs");  // Or relative to current path if needed
    // Verify d:/ exists? On linux likely not.
    // The user environment says Windows, but paths are mixed.
    // The user uses /mnt/nvme... in the error log.
    // I should use fs::current_path() / "docs" or similar to be safe, OR just use the relative path "docs" since we are
    // running from build or root. The previous code had "d:/src/NanoDeploy/docs". I will use a relative path "../docs"
    // assuming build is in "build/". Or just "docs" in current working dir.

    // Actually the user's error says "/mnt/nvme1n1/.../tests/test_qwen3_runner.cpp", so it IS Linux environment despite
    // the user info saying Windows. Wait, the User Info block says: "The USER's OS version is windows." But the error
    // log is clearly Linux GCC output. This happens with WSL. I should use generic paths.

    fs::path      output_path = fs::current_path() / "qwen3_chat_output.md";
    std::ofstream out_file(output_path);
    if (out_file.is_open()) {
        out_file << "# Qwen3 Generation Output\n\n";
        out_file << "**Input IDs:** ";
        for (int id : prompt_ids)
            out_file << id << " ";
        out_file << "\n\n**Output IDs:** ";
        for (int id : output_ids)
            out_file << id << " ";
        out_file << "\n";
        out_file.close();
        std::cout << "Output saved to " << output_path << std::endl;
    }
    else {
        std::cerr << "Failed to save output to " << output_path << std::endl;
    }

    // Use exit(0) to skip potentially crashing destructors (like Spoke Client threads)
    // The OS will clean up resources.
    std::exit(0);
}
