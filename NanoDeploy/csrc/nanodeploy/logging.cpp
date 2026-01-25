#include "nanocommon/logging.h"

namespace nanoinfra {

static int global_log_level = 0;

int get_log_level()
{
    return global_log_level;
}

void set_log_level(int level)
{
    global_log_level = level;
}

}  // namespace nanoinfra
