#!/usr/bin/env python3
"""Test script to simulate decode engine registration scenario."""

import json
import random
import sys
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import requests


def simulate_decode_engine_registration(
    server_url: str, engine_id: str, num_ranks: int = 8
):
    """Simulate decode engine agent registration exactly as in cache.py."""
    print(f"Simulating decode engine registration")
    print(f"  Server URL: {server_url}")
    print(f"  Engine ID: {engine_id}")
    print(f"  Number of ranks: {num_ranks}")
    print("-" * 60)

    # Test basic connectivity first
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        response.raise_for_status()
        print(f"✓ Server is reachable: {response.text.strip()}\n")
    except Exception as e:
        print(f"✗ Server is not reachable: {e}")
        return False

    def register_rank(rank: int):
        """Register a single rank, exactly as in cache.py."""
        agent_alias = f"{engine_id}:{rank}"

        # Convert nanoctrl_address to full URL if needed (as in cache.py)
        full_server_url = server_url
        if not full_server_url.startswith("http://") and not full_server_url.startswith(
            "https://"
        ):
            full_server_url = f"http://{full_server_url}"

        # Add small delay to avoid all workers registering simultaneously
        # This matches the logic in cache.py line 173
        delay = random.uniform(0.0, 0.5) * rank
        time.sleep(delay)

        print(f"[Rank {rank}] Attempting registration after {delay:.3f}s delay...")

        try:
            response = requests.post(
                f"{full_server_url}/start_peer_agent",
                json={
                    "alias": agent_alias,
                    "device": "mlx5_0",
                    "ib_port": 1,
                    "link_type": "RoCE",
                    "address": "10.102.97.179",  # This would be the actual IP
                },
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()

            print(f"[Rank {rank}] ✓ Registered successfully: {agent_alias}")
            if "redis_address" in result:
                print(f"[Rank {rank}]   Redis address: {result['redis_address']}")

            return {
                "success": True,
                "rank": rank,
                "alias": agent_alias,
                "result": result,
            }
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            print(f"[Rank {rank}] ✗ ConnectionError: {error_msg}")
            return {
                "success": False,
                "rank": rank,
                "alias": agent_alias,
                "error": f"ConnectionError: {error_msg}",
                "error_type": "ConnectionError",
            }
        except requests.exceptions.Timeout as e:
            error_msg = str(e)
            print(f"[Rank {rank}] ✗ Timeout: {error_msg}")
            return {
                "success": False,
                "rank": rank,
                "alias": agent_alias,
                "error": f"Timeout: {error_msg}",
                "error_type": "Timeout",
            }
        except Exception as e:
            error_msg = str(e)
            print(f"[Rank {rank}] ✗ Error: {type(e).__name__}: {error_msg}")
            return {
                "success": False,
                "rank": rank,
                "alias": agent_alias,
                "error": f"{type(e).__name__}: {error_msg}",
                "error_type": type(e).__name__,
            }

    # Register all ranks concurrently (as they would in Ray workers)
    print(f"\nRegistering {num_ranks} ranks concurrently...\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_ranks) as executor:
        futures = [executor.submit(register_rank, rank) for rank in range(num_ranks)]
        results = [future.result() for future in as_completed(futures)]

    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"Summary:")
    print(f"  Total ranks: {num_ranks}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")

    if failed:
        print(f"\nFailed registrations:")
        error_types = {}
        for r in failed:
            error_type = r.get("error_type", "Unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
            print(f"  - {r['alias']}: {r['error']}")

        print(f"\nError type distribution:")
        for error_type, count in error_types.items():
            print(f"  - {error_type}: {count}")

    return len(failed) == 0


if __name__ == "__main__":
    # Use the actual decode engine ID format from logs
    server_url = "http://10.102.97.179:3000"
    engine_id = "07f032de-9f00-4b65-b161-a15eff5a9b83"  # From the error logs
    num_ranks = 8

    # Parse command line arguments
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        if not server_url.startswith("http://") and not server_url.startswith(
            "https://"
        ):
            server_url = f"http://{server_url}"

    if len(sys.argv) > 2:
        engine_id = sys.argv[2]

    if len(sys.argv) > 3:
        num_ranks = int(sys.argv[3])

    print(f"Simulating decode engine registration scenario")
    print(f"Using engine ID from error logs: {engine_id}\n")

    success = simulate_decode_engine_registration(server_url, engine_id, num_ranks)
    sys.exit(0 if success else 1)
