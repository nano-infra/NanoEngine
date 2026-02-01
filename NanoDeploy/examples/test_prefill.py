import argparse
import json
import time

import ray
from test_utils import TestWorker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-ranks", type=int, default=1, help="Number of ranks")
    parser.add_argument("--base-port", type=int, default=30000, help="Base port")
    args = parser.parse_args()

    # Cleanup ports before starting
    from test_utils import cleanup_ports

    base_port = args.base_port
    cleanup_ports(base_port, args.num_ranks)

    ray.init(ignore_reinit_error=True)

    num_ranks = args.num_ranks
    host = "127.0.0.1"  # Adjust for multi-node

    workers = [TestWorker.remote(i, host, base_port) for i in range(num_ranks)]

    # Gather addresses
    addrs = ray.get([w.get_addr.remote() for w in workers])

    print("\n" + "=" * 50)
    print("PREFILL NODES STARTED")
    print(json.dumps(addrs, indent=2))
    print("=" * 50)

    # Write addresses to a file for decode script to read
    with open("prefill_addrs.json", "w") as f:
        json.dump(addrs, f)

    print("\nAddresses saved to prefill_addrs.json. Keeping prefill alive...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()
