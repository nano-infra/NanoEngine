"""Ray job manager for NanoOps."""

import logging
import os
import shutil
import time
from typing import Optional

import ray
from ray.job_submission import JobStatus, JobSubmissionClient
from ray.util.placement_group import (
    placement_group,
    PlacementGroup,
    remove_placement_group,
)

from .exceptions import RayJobError

logger = logging.getLogger(__name__)


class RayJobManager:
    """Manages Ray job submission and placement groups."""

    def __init__(self, ray_address: str):
        """Initialize Ray job manager.

        Args:
            ray_address: Ray dashboard HTTP address (e.g., http://localhost:8265)
                        Can also accept ray:// format or just host:port, will be converted
        """
        # Normalize address format
        if ray_address.startswith("ray://"):
            # Extract host and port, use 8265 as default dashboard port
            from urllib.parse import urlparse

            parsed = urlparse(ray_address)
            host = parsed.hostname
            dashboard_port = 8265
            ray_address = f"http://{host}:{dashboard_port}"
            logger.info(f"Converted ray:// address to dashboard URL: {ray_address}")
        elif not ray_address.startswith(("http://", "https://")):
            # Just host:port format, add http://
            ray_address = f"http://{ray_address}"
            logger.info(f"Added http:// protocol: {ray_address}")

        self.address = ray_address

        # Initialize job submission client
        try:
            self.client = JobSubmissionClient(ray_address)
            logger.info(f"Connected to Ray dashboard at {ray_address}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ray dashboard: {e}")
            self.client = None

        # Don't initialize Ray here - will do it lazily when needed for placement groups

    def create_placement_group(
        self,
        session_id: str,
        component_type: str,
        num_gpus: int,
    ) -> str:
        """Create STRICT_PACK placement group for gang scheduling.

        Each bundle requests 1 GPU so that ``num_gpus`` bundles are created
        and packed onto the same node.

        Args:
            session_id: Session identifier
            component_type: Component type (prefill, decode)
            num_gpus: Total number of GPUs needed

        Returns:
            Placement group ID (hex string)

        Raises:
            RayJobError: If placement group creation fails
        """
        try:
            # Ensure Ray is initialized for placement group management
            if not ray.is_initialized():
                logger.info("Initializing Ray for placement group management...")
                # Temporarily unset RAY_ADDRESS: it may contain the dashboard
                # HTTP URL (e.g. http://host:8265) which ray.init() cannot
                # parse — it expects "auto" or "ray://host:port".
                saved_ray_addr = os.environ.pop("RAY_ADDRESS", None)
                try:
                    ray.init(address="auto", ignore_reinit_error=True)
                finally:
                    if saved_ray_addr is not None:
                        os.environ["RAY_ADDRESS"] = saved_ray_addr

            logger.info(
                f"Creating placement group for {component_type}: "
                f"{num_gpus} GPUs ({num_gpus} bundles, STRICT_PACK)"
            )

            pg = placement_group(
                bundles=[{"CPU": 0.1, "GPU": 1.0}] * num_gpus,
                strategy="STRICT_PACK",
                name=f"{session_id}_{component_type}_pg",
            )

            # Wait for placement group to be ready
            ray.get(pg.ready(), timeout=60)

            pg_id = pg.id.hex()
            logger.info(f"Placement group created: {pg_id}")

            return pg_id

        except Exception as e:
            raise RayJobError(f"Failed to create placement group: {e}")

    def submit_component_job(
        self,
        session_id: str,
        component_type: str,
        config_path: str,
        env_vars: dict,
        placement_group_id: Optional[str] = None,
        working_dir: Optional[str] = None,
        excludes: Optional[list] = None,
    ) -> str:
        """Submit Ray job for a component.

        Args:
            session_id: Session identifier
            component_type: Component type (route, prefill, decode)
            config_path: Path to component config file
            env_vars: Environment variables to inject
            placement_group_id: Optional placement group ID
            working_dir: Optional working directory for code upload

        Returns:
            Ray job ID

        Raises:
            RayJobError: If job submission fails
        """
        try:
            # Build entrypoint command
            if component_type == "route":
                nanoroute_bin = self._find_nanoroute_binary()
                entrypoint = f"{nanoroute_bin} --config {config_path}"
            elif component_type in ["prefill", "decode"]:
                entrypoint = f"python -m nanodeploy.server.engine_server --config {config_path} --log_level INFO"
            else:
                raise ValueError(f"Unknown component type: {component_type}")

            # Build runtime environment
            runtime_env = {"env_vars": env_vars}

            # Add placement group if provided
            if placement_group_id:
                runtime_env["placement_group_id"] = placement_group_id

            # Add working directory if provided (for code upload)
            if working_dir:
                runtime_env["working_dir"] = working_dir

            # Add excludes to reduce upload size
            if excludes:
                runtime_env["excludes"] = excludes

            # Generate unique job ID
            job_id = f"{session_id}_{component_type}_{int(time.time())}"

            logger.info(f"Submitting Ray job: {job_id}")
            logger.debug(f"Entrypoint: {entrypoint}")
            logger.debug(f"Runtime env: {runtime_env}")

            # Submit job
            submission_id = self.client.submit_job(
                entrypoint=entrypoint,
                runtime_env=runtime_env,
                job_id=job_id,
            )

            logger.info(f"Ray job submitted: {submission_id}")
            return submission_id

        except Exception as e:
            raise RayJobError(f"Failed to submit job: {e}")

    def get_job_status(self, job_id: str) -> str:
        """Get Ray job status.

        Args:
            job_id: Ray job ID

        Returns:
            Job status: PENDING | RUNNING | SUCCEEDED | FAILED | STOPPED

        Raises:
            RayJobError: If status check fails
        """
        try:
            status = self.client.get_job_status(job_id)
            return status.value if isinstance(status, JobStatus) else status
        except Exception as e:
            raise RayJobError(f"Failed to get job status: {e}")

    def get_job_logs(self, job_id: str) -> str:
        """Get Ray job logs.

        Args:
            job_id: Ray job ID

        Returns:
            Job logs
        """
        try:
            return self.client.get_job_logs(job_id)
        except Exception as e:
            logger.warning(f"Failed to get job logs: {e}")
            return ""

    def stop_job(self, job_id: str) -> None:
        """Stop a running job.

        Args:
            job_id: Ray job ID

        Raises:
            RayJobError: If stop fails
        """
        try:
            logger.info(f"Stopping Ray job: {job_id}")
            self.client.stop_job(job_id)
        except Exception as e:
            raise RayJobError(f"Failed to stop job: {e}")

    def cleanup_placement_group(self, pg_id: str) -> None:
        """Remove a placement group.

        Args:
            pg_id: Placement group ID (hex string)
        """
        try:
            logger.info(f"Removing placement group: {pg_id}")
            pg = PlacementGroup.from_hex(pg_id)
            remove_placement_group(pg)
            logger.info("Placement group removed")
        except Exception as e:
            logger.warning(f"Failed to remove placement group {pg_id}: {e}")

    def is_connected(self) -> bool:
        """Check if connected to Ray cluster."""
        try:
            return ray.is_initialized()
        except Exception:
            return False

    @staticmethod
    def _find_nanoroute_binary() -> str:
        """Locate the NanoRoute binary.

        Lookup order:
        1. System PATH (``which nanoroute``)
        2. ``NANOCTRL_NANOROUTE_PATH`` environment variable

        Returns:
            Absolute path to the nanoroute binary.

        Raises:
            RayJobError: If the binary cannot be found.
        """
        # 1. System PATH
        path_bin = shutil.which("nanoroute")
        if path_bin:
            logger.debug(f"Found nanoroute in PATH: {path_bin}")
            return path_bin

        # 2. Environment variable
        env_path = os.getenv("NANOCTRL_NANOROUTE_PATH")
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            logger.debug(f"Found nanoroute via NANOCTRL_NANOROUTE_PATH: {env_path}")
            return env_path

        raise RayJobError(
            "NanoRoute binary not found. Either:\n"
            "  1. Install nanoroute so it is available in PATH, or\n"
            "  2. Set NANOCTRL_NANOROUTE_PATH to the binary path\n"
            "  Build: cd NanoRoute && cargo build --release"
        )
