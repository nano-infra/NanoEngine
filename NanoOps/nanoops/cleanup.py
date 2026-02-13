"""Cleanup utilities for NanoOps."""

import logging
import os
import subprocess
from typing import List, Optional

import redis

logger = logging.getLogger(__name__)


def kill_processes(pattern: str) -> int:
    """Kill processes matching pattern.

    Args:
        pattern: Process pattern to match

    Returns:
        Number of processes killed
    """
    try:
        result = subprocess.run(
            ["pkill", "-9", "-f", pattern],
            capture_output=True,
            text=True,
        )
        return 0 if result.returncode == 0 else 0
    except Exception as e:
        logger.debug(f"Error killing processes {pattern}: {e}")
        return 0


def cleanup_redis_keys(
    redis_url: str,
    patterns: List[str],
) -> int:
    """Delete Redis keys matching patterns.

    Args:
        redis_url: Redis connection URL
        patterns: List of key patterns to delete

    Returns:
        Total number of keys deleted
    """
    try:
        # Parse Redis URL
        if redis_url.startswith("redis://"):
            redis_url = redis_url[8:]

        parts = redis_url.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 6379

        client = redis.Redis(host=host, port=port, decode_responses=True)

        total_deleted = 0
        for pattern in patterns:
            keys = list(client.scan_iter(match=pattern, count=100))
            if keys:
                # Delete in batches of 100
                for i in range(0, len(keys), 100):
                    batch = keys[i : i + 100]
                    deleted = client.delete(*batch)
                    total_deleted += deleted
                logger.info(f"Deleted {len(keys)} keys matching: {pattern}")
            else:
                logger.debug(f"No keys found matching: {pattern}")

        return total_deleted

    except Exception as e:
        logger.error(f"Error cleaning Redis keys: {e}")
        return 0


def cleanup_all(
    redis_url: Optional[str] = None,
    sessions: Optional[List[str]] = None,
    kill_processes_flag: bool = True,
) -> None:
    """Clean up all NanoOps resources.

    Args:
        redis_url: Redis connection URL. Falls back to ``NANOCTRL_REDIS_URL``
                   env var, then ``redis://localhost:6379``.
        sessions: List of session IDs to clean (default: common test sessions)
        kill_processes_flag: Whether to kill stale processes
    """
    if redis_url is None:
        redis_url = os.getenv("NANOCTRL_REDIS_URL", "redis://localhost:6379")
    logger.info("Starting NanoOps cleanup...")

    # Default sessions to clean
    if sessions is None:
        sessions = ["demo", "demo2", "test", "test1", "test2", "JimyMa"]

    # Step 1: Kill processes
    if kill_processes_flag:
        logger.info("Killing stale processes...")
        kill_processes("engine_server")
        kill_processes("nanoroute.*config")

    # Step 2: Clean Redis keys
    logger.info("Cleaning Redis keys...")
    patterns = []

    # Add session-specific patterns
    for session_id in sessions:
        patterns.append(f"{session_id}:*")

    # Add general patterns
    patterns.extend(
        [
            "*:agent:*",  # Stale peer_agent keys
            "engine:*",  # Unscoped engine keys
        ]
    )

    total_deleted = cleanup_redis_keys(redis_url, patterns)
    logger.info(f"Deleted {total_deleted} Redis keys")

    logger.info("Cleanup complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleanup_all()
