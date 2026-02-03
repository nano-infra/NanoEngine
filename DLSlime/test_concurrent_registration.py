#!/usr/bin/env python3
"""Test script to simulate concurrent agent registrations."""

import json
import random
import sys
import time
from concurrent.futures import as_completed, ThreadPoolExecutor

import requests


def register_agent(server_url: str, alias: str, rank: int):
    """Register a single agent."""
    # Simulate the delay used in cache.py
    delay = random.uniform(0.0, 0.5) * rank
    time.sleep(delay)

    try:
        response = requests.post(
            f"{server_url}/start_peer_agent",
            json={
                "alias": alias,
                "device": "mlx5_0",
                "ib_port": 1,
                "link_type": "RoCE",
                "address": "10.102.97.179",
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "alias": alias,
            "rank": rank,
            "result": result,
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "alias": alias,
            "rank": rank,
            "error": f"ConnectionError: {e}",
        }
    except requests.exceptions.Timeout as e:
        return {
            "success": False,
            "alias": alias,
            "rank": rank,
            "error": f"Timeout: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "alias": alias,
            "rank": rank,
            "error": f"{type(e).__name__}: {e}",
        }


def test_concurrent_registration(server_url: str, engine_id: str, num_agents: int = 8):
    """Test concurrent agent registrations."""
    print(f"Testing concurrent registration at: {server_url}")
    print(f"Engine ID: {engine_id}, Number of agents: {num_agents}")
    print("-" * 60)

    # Test basic connectivity first
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        response.raise_for_status()
        print(f"✓ Server is reachable: {response.text.strip()}\n")
    except Exception as e:
        print(f"✗ Server is not reachable: {e}")
        return False

    # Register agents concurrently
    print(f"Registering {num_agents} agents concurrently...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_agents) as executor:
        futures = []
        for rank in range(num_agents):
            alias = f"{engine_id}:{rank}"
            future = executor.submit(register_agent, server_url, alias, rank)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"✓ {result['alias']} registered successfully")
            else:
                print(f"✗ {result['alias']} failed: {result['error']}")

    elapsed_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    successful = sum(1 for r in results if r["success"])
    failed = num_agents - successful
    print(f"Summary:")
    print(f"  Total: {num_agents}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Elapsed time: {elapsed_time:.2f}s")

    if failed > 0:
        print("\nFailed registrations:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['alias']}: {r['error']}")

    return failed == 0


if __name__ == "__main__":
    # Default values
    server_url = "http://10.102.97.179:3000"
    engine_id = "test_engine_concurrent"
    num_agents = 8

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
        num_agents = int(sys.argv[3])

    success = test_concurrent_registration(server_url, engine_id, num_agents)
    sys.exit(0 if success else 1)
