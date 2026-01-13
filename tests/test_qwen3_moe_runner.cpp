#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nanodeploy/engine/simple_engine.h"
#include "nanodeploy/logging.h"
#include "nanodeploy/sequence/sequence.h"
#include "sequence/sequence.h"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_json_path> [prompt_id_1 prompt_id_2 ...]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];

    // Parse Args
    std::string agent_ip     = "127.0.0.1";
    int         agent_port   = 9000;
    int         attention_dp = 1;
    int         ffn_ep       = 1;

    nanodeploy::get_log_level() = 1;
    std::vector<int> prompt_ids;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--agent_ip" || arg == "--agent-ip") && i + 1 < argc) {
            agent_ip = argv[++i];
        }
        else if ((arg == "--agent_port" || arg == "--agent-port") && i + 1 < argc) {
            agent_port = std::stoi(argv[++i]);
        }
        else if ((arg == "--log_level" || arg == "--log-level") && i + 1 < argc) {
            nanodeploy::get_log_level() = std::stoi(argv[++i]);
        }
        else if ((arg == "--attention_dp" || arg == "--attention-dp") && i + 1 < argc) {
            attention_dp = std::stoi(argv[++i]);
        }
        else if ((arg == "--ffn_ep" || arg == "--ffn-ep") && i + 1 < argc) {
            ffn_ep = std::stoi(argv[++i]);
        }
        else {
            try {
                prompt_ids.push_back(std::stoi(arg));
            }
            catch (...) {
            }
        }
    }

    if (prompt_ids.empty()) {
        prompt_ids = {9707, 1879, 0};
        std::cout << "No prompt provided, using default: Hello world! (IDs: 9707 1879 0)" << std::endl;
    }

    nanodeploy::SimpleEngine engine;
    try {
        engine.init(config_path,
                    1,
                    1,
                    1,
                    agent_ip,
                    agent_port,
                    1,
                    attention_dp,
                    1,  // attention_tp, attention_dp, attention_sp
                    1,
                    1,
                    ffn_ep);  // ffn_tp, ffn_dp, ffn_ep
    }
    catch (const std::exception& e) {
        std::cerr << "Engine init failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "MoE Runner Initialized with Attention DP=" << attention_dp << ", FFN EP=" << ffn_ep << std::endl;
    std::cout << "Starting generation..." << std::endl;

    // Add requests
    std::vector<std::shared_ptr<nanodeploy::Sequence>> total_seqs;
    int                                                num_requests = 1;
    for (int i = 0; i < num_requests; ++i) {
        auto seq = engine.add_request(prompt_ids, 20);
        total_seqs.emplace_back(seq);
    }

    // Run generation
    int step_count = 0;
    while (!engine.is_finished()) {
        step_count++;
        engine.step();
    }
    std::cout << "Generation Finished in " << step_count << " steps." << std::endl;

    // Get results directly from sequences
    auto results = engine.get_finished_sequences();

    // Save outputs
    fs::path      output_path = fs::current_path() / "qwen3_moe_chat_output.md";
    std::ofstream out_file(output_path);
    if (out_file.is_open()) {
        out_file << "# Qwen3 MoE Generation Output\n\n";
        out_file << "**Configuration:** Attention DP=" << attention_dp << ", FFN EP=" << ffn_ep << "\n\n";
        out_file << "**Prompt IDs:** ";
        for (int id : prompt_ids)
            out_file << id << " ";
        out_file << "\n\n";

        for (const auto& seq : total_seqs) {
            out_file << "## Sequence " << seq->seq_id << "\n\n";
            out_file << "**Generated IDs:** ";
            for (int id : seq->token_ids)
                out_file << id << " ";
            out_file << "\n\n";

            // Also print to console
            std::cout << "Sequence " << seq->seq_id << ": ";
            for (int id : seq->token_ids)
                std::cout << id << " ";
            std::cout << std::endl;
        }

        out_file.close();
        std::cout << "Output saved to " << output_path << std::endl;
    }

    return 0;
}
