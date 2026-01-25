# Spoke: High-Performance Distributed Actor Framework

Spoke is a lightweight, low-latency distributed task execution framework designed for high-performance computing and LLM inference workloads. It serves as a C++ native, RDMA-optimized alternative to heavier frameworks like Ray, specifically tailored for the needs of LLM inference.

## 🚀 Key Features

- **Resource-Aware Gang Scheduling (New in V2)**:
  - **`gangAllocate` API**: Atomically allocate multiple actors (e.g., 8 GPUs on one node) to ensure locality.
  - **Two-Phase Launch**: Decouples resource reservation from process launch, allowing precise topology configuration.
  - **GPU Isolation**: Automatic `CUDA_VISIBLE_DEVICES` injection.
- **Actor-Based Model**: Intuitive unit of distributed execution. Define a class, register it as an Actor, and invoke its methods remotely.
- **Zero-Copy RDMA Serialization**: Deep integration with **DLSlime** enables a "True Zero-Copy" (Zero-ish) path, removing unnecessary heap allocations and memory copies during RDMA transfers.
- **Robust Process Management**:
  - **Fork-Exec Architecture**: Thread-safe process spawning that eliminates `malloc` lock deadlocks in multi-threaded environments.
  - **Automatic Cleanup**: Uses `PR_SET_PDEATHSIG` to ensure worker processes terminate when the parent agent dies.
- **Synchronous Spawn Protocol**: Guaranteed actor availability with an Ack-based spawning flow.
- **Bi-Directional Communication**: Efficient request-response cycles with RDMA-backed response paths.
- **Scalable Control Plane**: Lightweight TCP-based control plane with a high-performance binary protocol.

## 📚 Documentation

- [**V2 User Guide (Gang Scheduling)**](docs/spoke_v2_user_guide.md): Detailed guide on using the new allocation and launch APIs.

## 🛠️ Core Components

- **`spoke::Client`**: The primary user interface for spawning actors and making remote calls (`callRemote`).
- **`spoke::Actor`**: The base class for remote services. Use the `SPOKE_METHOD` macro to define remote procedures.
- **`spoke::Agent`**: The background process (Daemon) that manages actor lifecycles and provides the runtime environment.
- **`spoke::Hub`**: Central registry and scheduler for resource management.
- **`spoke::Serializer`**: A extensible serialization interface supporting zero-copy `PackTo` and `UnpackFrom` operations.

## 📦 Installation

Spoke uses CMake for its build system and can be installed as a system library or integrated via `add_subdirectory`.

```bash
mkdir build && cd build
cmake ..
make -j
sudo make install
```

To use Spoke in your project:

```cmake
find_package(Spoke REQUIRED)
target_link_libraries(my_app PUBLIC Spoke::core Spoke::agent)
```

## 💻 Usage Example

### 1. Gang Allocation (V2)

```cpp
#include "spoke/csrc/client.h"

int main() {
    spoke::Client client("127.0.0.1", 8888, true);

    // Allocate 1 Node with 8 GPUs
    spoke::ResourceSpec res; res.num_gpus = 1;
    auto alloc = client.gangAllocate(1, 8, res).get();

    // Launch Actors
    for(int i=0; i<alloc.num_members; ++i) {
        client.launchActor(alloc.ticket_id, i, "MyActor", "worker_" + std::to_string(i), "");
    }
}
```

### 2. Define an Actor

```cpp
#include "spoke/csrc/actor.h"

class MyActor : public spoke::Actor {
public:
    std::string handle_hello(spoke::MessageView msg) {
        return "Hello from Spoke!";
    }

    void registerMethods() override {
        registerMethod(0x01, [this](auto m) { return handle_hello(m); });
    }
};

// Register the actor type
SPOKE_REGISTER_ACTOR(MyActor, "MyActor");
```

### 3. Invoke Remotely (RDMA Zero-Copy)

```cpp
#include "spoke/csrc/client.h"

int main() {
    // Enable RDMA in the client
    spoke::Client client("127.0.0.1", 8888, true);

    client.spawnRemote("MyActor", "actor_instance_1");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Explicitly initialize RDMA for this actor
    if (!client.initRDMA("actor_instance_1")) {
        return 1;
    }

    ComplexTask task = {666, "ZeroCopyCheck", {1.0, 2.0}};

    // callRemote automatically uses the zero-copy path if RDMA is initialized
    auto future = client.callRemote<ComplexTask, std::string>("actor_instance_1", 0x60, task);
    std::cout << "Result: " << future.get() << std::endl;

    return 0;
}
```

## 🔬 Performance Comparison

| Feature           | Spoke              | Ray (C++ Core)       |
| :---------------- | :----------------- | :------------------- |
| **Binaray Size**  | \< 1MB             | > 100MB              |
| **Startup Time**  | ~10ms              | > 1s                 |
| **RDMA Support**  | Native (Zero-Copy) | Optional (Shm-based) |
| **Serialization** | Zero-Copy Internal | Protobuf/Arrow       |
| **Complexity**    | Minimal            | High                 |

## 📖 License

Spoke is released under the MIT License. See `LICENSE` for details.
