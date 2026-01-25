#pragma once
#include <string>

namespace spoke {

class Agent;

void register_to_hub(const std::string& hub_ip, int hub_port, int my_port);
void handle_client(int client_fd, Agent* agent);
int  run_daemon(int argc, char** argv);

}  // namespace spoke
