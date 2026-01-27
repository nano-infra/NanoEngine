#include "spoke/csrc/client.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

struct ComplexTask {
  int id;
  std::string name;
  std::vector<double> values;
};

namespace spoke {
template <> struct Serializer<ComplexTask> {
  static std::string pack(const ComplexTask &t) {
    std::stringstream ss;
    ss << t.id << "|" << t.name << "|";
    for (double v : t.values)
      ss << v << ",";
    return ss.str();
  }
  static ComplexTask unpack(const std::string &s) {
    return {}; // Stub
  }

  // [New] Zero-Copy Interface
  static size_t size(const ComplexTask &t) {
    // Estimate size or serialize first
    // Simple inefficient way consistent with existing pack:
    return pack(t).size();
  }

  static void packTo(const ComplexTask &t, char *buf) {
    std::string s = pack(t);
    memcpy(buf, s.data(), s.size());
  }

  static ComplexTask unpackFrom(const char *buf, size_t size) {
    return unpack(std::string(buf, size));
  }
};
} // namespace spoke

const spoke::Action kProcessTask = (spoke::Action)0x60;

int main() {
  std::string hub_ip = "127.0.0.1";
  int hub_port = 8888;

  std::cout << "[Smoke v5] Connecting to Hub..." << std::endl;
  spoke::Client client(hub_ip, hub_port, true);

  std::cout << "[Smoke v5] Spawning AdvancedActor..." << std::endl;
  client.spawnRemote("AdvancedActor", "adv_1");
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  ComplexTask task;
  task.id = 999;
  task.name = "ProjectX";
  task.values = {1.1, 2.2, 3.3, 4.4};

  std::cout
      << "[Smoke v5] Calling remote with Complex Object (Custom Serializer)..."
      << std::endl;

  auto fut =
      client.callRemote<ComplexTask, std::string>("adv_1", kProcessTask, task);

  std::cout << "[Smoke v5] Result: " << fut.get() << std::endl;

  return 0;
}
