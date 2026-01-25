#include "daemon.h"
#include "spoke/csrc/actor.h"
#include <cstring>
#include <iostream>

int main(int argc, char** argv)
{
    // Worker Mode: --worker <type> <id> <rx_fd> <tx_fd>
    if (argc >= 6 && strcmp(argv[1], "--worker") == 0) {
        // [Fix] Disable buffering for stdout/stderr to ensure logs are captured fully,
        // especially when redirected to files in Spoke Agent.
        setbuf(stdout, NULL);
        setbuf(stderr, NULL);

        std::string type  = argv[2];
        std::string id    = argv[3];
        int         rx_fd = std::atoi(argv[4]);
        int         tx_fd = std::atoi(argv[5]);

        auto actor = spoke::ActorFactory::instance().create(type, id, rx_fd, tx_fd);
        if (actor) {
            actor->run();
        }
        else {
            std::cerr << "[Worker] Unknown actor type: " << type << std::endl;
            return 1;
        }
        return 0;
    }

    return spoke::run_daemon(argc, argv);
}
