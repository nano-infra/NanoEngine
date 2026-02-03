#!/usr/bin/env python3
"""Test script to verify control plane connectivity."""

import json
import sys

import requests


def test_control_plane(server_url: str):
    """Test connection to control plane server."""
    print(f"Testing control plane at: {server_url}")
    print("-" * 60)

    # Test 1: Basic connectivity
    print("\n[Test 1] Testing basic connectivity (GET /)")
    try:
        response = requests.get(f"{server_url}/", timeout=5)
        response.raise_for_status()
        print(f"✓ Success: {response.text.strip()}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 2: Register a test agent
    print("\n[Test 2] Testing agent registration (POST /start_peer_agent)")
    test_alias = "test_agent_python"
    try:
        response = requests.post(
            f"{server_url}/start_peer_agent",
            json={
                "alias": test_alias,
                "device": "mlx5_0",
                "ib_port": 1,
                "link_type": "RoCE",
                "address": "10.102.97.179",
            },
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        print(f"✓ Success: {json.dumps(result, indent=2)}")

        if "redis_address" in result:
            print(f"  Redis address: {result['redis_address']}")

    except requests.exceptions.ConnectionError as e:
        print(f"✗ ConnectionError: {e}")
        print(
            f"  This usually means the server is not reachable or connection was refused."
        )
        return False
    except requests.exceptions.Timeout as e:
        print(f"✗ Timeout: {e}")
        print(f"  The server did not respond within the timeout period.")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTPError: {e}")
        print(f"  Status code: {response.status_code}")
        print(f"  Response: {response.text}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    return True


if __name__ == "__main__":
    # Default server URL
    server_url = "http://10.102.97.179:3000"

    # Allow override via command line
    if len(sys.argv) > 1:
        server_url = sys.argv[1]
        if not server_url.startswith("http://") and not server_url.startswith(
            "https://"
        ):
            server_url = f"http://{server_url}"

    success = test_control_plane(server_url)
    sys.exit(0 if success else 1)
