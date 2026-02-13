"""NanoCtrl HTTP client for NanoOps."""

import logging
import os
import subprocess
import time
from typing import Dict, List, Optional

import httpx

from .exceptions import NanoCtrlError
from .utils import check_port_listening, find_binary

logger = logging.getLogger(__name__)


class NanoCtrlClient:
    """HTTP client for NanoCtrl API."""

    def __init__(self, nanoctrl_address: str, redis_url: str):
        """Initialize NanoCtrl client.

        Args:
            nanoctrl_address: NanoCtrl HTTP address (e.g., http://localhost:3000)
                             If no protocol specified, http:// will be added
            redis_url: Redis connection URL
        """
        # Add http:// if no protocol specified
        if not nanoctrl_address.startswith(("http://", "https://")):
            nanoctrl_address = f"http://{nanoctrl_address}"
            logger.info(f"Added http:// to NanoCtrl address: {nanoctrl_address}")

        self.address = nanoctrl_address
        self.redis_url = redis_url
        self.client = httpx.Client(timeout=30.0)

        # Parse host and port from address
        from urllib.parse import urlparse

        parsed = urlparse(nanoctrl_address)
        self.host = parsed.hostname or "localhost"
        self.port = parsed.port or 3000

    def is_healthy(self) -> bool:
        """Check if NanoCtrl is running and healthy."""
        try:
            # Try to get Redis address (simple health check)
            # NanoCtrl expects JSON content-type
            response = self.client.post(
                f"{self.address}/get_redis_address",
                json={},  # Empty JSON body
            )
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"NanoCtrl health check failed: {e}")
            return False

    def ensure_running(
        self,
        auto_start: bool = True,
        config_path: Optional[str] = None,
        binary_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """Ensure NanoCtrl is running, start if needed.

        Args:
            auto_start: Auto-start NanoCtrl if not running
            config_path: Path to NanoCtrl config.toml
            binary_path: Path to NanoCtrl binary

        Returns:
            Dict with status info: {"status": "already_running"|"started", "address": ..., "pid": ...}

        Raises:
            NanoCtrlError: If NanoCtrl is not running and auto_start=False, or if start fails
        """
        # Check if already running
        if self.is_healthy():
            logger.info(f"NanoCtrl already running at {self.address}")
            return {"status": "already_running", "address": self.address}

        # Check if port is occupied by something else
        if check_port_listening(self.host, self.port):
            raise NanoCtrlError(
                f"Port {self.port} is occupied but NanoCtrl is not responding. "
                "Please check what's running on this port."
            )

        if not auto_start:
            raise NanoCtrlError(
                f"NanoCtrl is not running at {self.address}. "
                "Start it manually or enable auto-start."
            )

        # Start NanoCtrl
        logger.info(f"Starting NanoCtrl on {self.host}:{self.port}")
        return self._start_nanoctrl(config_path, binary_path)

    def _start_nanoctrl(
        self,
        config_path: Optional[str] = None,
        binary_path: Optional[str] = None,
    ) -> Dict[str, any]:
        """Start NanoCtrl as subprocess.

        Args:
            config_path: Path to config.toml (will generate if not provided)
            binary_path: Path to NanoCtrl binary

        Returns:
            Dict with status info

        Raises:
            NanoCtrlError: If start fails
        """
        # Find NanoCtrl binary
        if not binary_path:
            search_paths = [
                "../NanoCtrl/target/release",
                "../../NanoCtrl/target/release",  # If running from NanoOps/nanoops
                os.path.join(
                    os.getenv("NANOINFRA_ROOT", ""), "NanoCtrl/target/release"
                ),
            ]
            binary_path = find_binary("nanoctrl", search_paths)

        if not binary_path:
            raise NanoCtrlError(
                "NanoCtrl binary not found. Please build NanoCtrl first:\n"
                "  cd NanoCtrl && cargo build --release"
            )

        # Generate config if not provided
        if not config_path:
            config_path = self._generate_config()

        # Build environment: inherit current env + ensure NANOCTRL_REDIS_URL
        child_env = os.environ.copy()
        child_env["NANOCTRL_REDIS_URL"] = self.redis_url

        # Start as subprocess
        try:
            logger.info(f"Launching: {binary_path} --config {config_path}")

            process = subprocess.Popen(
                [binary_path, "--config", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,  # Detach from parent
                env=child_env,
            )

            # Wait for health check (30s timeout)
            start_time = time.time()
            while time.time() - start_time < 30:
                if self.is_healthy():
                    logger.info(f"NanoCtrl started successfully (PID: {process.pid})")
                    return {
                        "status": "started",
                        "address": self.address,
                        "pid": process.pid,
                        "config_path": config_path,
                    }
                time.sleep(1)

            # Startup failed
            process.kill()
            raise NanoCtrlError("NanoCtrl failed to start within 30 seconds")

        except Exception as e:
            raise NanoCtrlError(f"Failed to start NanoCtrl: {e}")

    def _generate_config(self) -> str:
        """Generate temporary config.toml for NanoCtrl.

        Returns:
            Path to generated config file
        """
        config_content = f"""[server]
host = "{self.host}"
port = {self.port}

[redis]
url = "{self.redis_url}"
"""

        config_path = f"/tmp/nanoctrl_{int(time.time())}.toml"

        with open(config_path, "w") as f:
            f.write(config_content)

        logger.debug(f"Generated NanoCtrl config at {config_path}")
        return config_path

    def list_engines(
        self, session_id: str, role: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """List engines for a session.

        Args:
            session_id: Session identifier (used as scope)
            role: Optional role filter (prefill, decode)

        Returns:
            List of engine info dicts

        Raises:
            NanoCtrlError: If request fails
        """
        try:
            # Pass scope (session_id) so NanoCtrl queries the correct
            # scoped Redis keys: {scope}:engine:*
            body = {"scope": session_id} if session_id else {}
            response = self.client.post(f"{self.address}/list_engines", json=body)

            if response.status_code != 200:
                raise NanoCtrlError(f"Failed to list engines: {response.text}")

            result = response.json()

            # Handle different response formats
            if isinstance(result, list):
                engines = result
            elif isinstance(result, dict):
                # Might be wrapped in a dict like {"engines": [...]}
                engines = result.get("engines", [])
            else:
                logger.warning(
                    f"Unexpected response format from list_engines: {type(result)}"
                )
                engines = []

            # Filter by role if specified
            if role:
                engines = [
                    e for e in engines if isinstance(e, dict) and e.get("role") == role
                ]

            return engines

        except httpx.HTTPError as e:
            raise NanoCtrlError(f"HTTP error listing engines: {e}")

    def get_engine_info(self, engine_id: str) -> Dict[str, any]:
        """Get specific engine information.

        Args:
            engine_id: Engine identifier

        Returns:
            Engine info dict

        Raises:
            NanoCtrlError: If request fails
        """
        try:
            response = self.client.post(
                f"{self.address}/get_engine_info", json={"engine_id": engine_id}
            )

            if response.status_code != 200:
                raise NanoCtrlError(f"Failed to get engine info: {response.text}")

            return response.json()

        except httpx.HTTPError as e:
            raise NanoCtrlError(f"HTTP error getting engine info: {e}")

    def get_redis_url(self) -> str:
        """Get Redis URL from NanoCtrl.

        Returns:
            Redis connection URL

        Raises:
            NanoCtrlError: If request fails
        """
        try:
            response = self.client.post(f"{self.address}/get_redis_address", json={})

            if response.status_code != 200:
                raise NanoCtrlError(f"Failed to get Redis address: {response.text}")

            return response.json().get("redis_url", "")

        except httpx.HTTPError as e:
            raise NanoCtrlError(f"HTTP error getting Redis address: {e}")

    def close(self):
        """Close HTTP client."""
        self.client.close()
