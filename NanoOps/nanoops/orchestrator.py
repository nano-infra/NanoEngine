"""Session orchestrator for NanoOps."""

import logging
import os
import time
from typing import Dict, Optional

import yaml

from .exceptions import ConfigError, SessionExistsError, SessionNotFoundError
from .health_checker import HealthChecker
from .nanoctrl_client import NanoCtrlClient
from .ray_manager import RayJobManager
from .redis_client import RedisClient
from .utils import allocate_port, validate_session_id

logger = logging.getLogger(__name__)


class SessionOrchestrator:
    """Orchestrates NanoInfra session lifecycle."""

    def __init__(
        self,
        redis_url: str,
        ray_address: str,
        nanoctrl_address: str,
    ):
        """Initialize session orchestrator.

        Args:
            redis_url: Redis connection URL
            ray_address: Ray dashboard HTTP address (for job submission)
            nanoctrl_address: NanoCtrl HTTP address
        """
        self.redis = RedisClient(redis_url)
        self.ray = RayJobManager(ray_address)
        self.nanoctrl = NanoCtrlClient(nanoctrl_address, redis_url)
        self.health = HealthChecker(self.redis, self.nanoctrl, self.ray)

        # Convert dashboard address to GCS address for engines
        # Dashboard: http://host:8265 -> GCS: ray://host:7078
        self.ray_gcs_address = self._dashboard_to_gcs_address(ray_address)

    def _dashboard_to_gcs_address(self, dashboard_address: str) -> str:
        """Convert Ray dashboard address to GCS address for ray.init().

        Args:
            dashboard_address: Dashboard address (http://host:8265)

        Returns:
            GCS address (ray://host:7078)
        """
        from urllib.parse import urlparse

        parsed = urlparse(dashboard_address)
        host = parsed.hostname or "localhost"

        # GCS is typically on port 7078 (or 6379 default)
        # We'll use 7078 as that's what the user has
        gcs_port = 7078

        return f"ray://{host}:{gcs_port}"

    def start_session(
        self,
        session_id: str,
        ensure_nanoctrl: bool = True,
    ) -> Dict[str, any]:
        """Initialize session and setup Redis namespace.

        Args:
            session_id: Unique session identifier
            ensure_nanoctrl: Whether to ensure NanoCtrl is running

        Returns:
            Session info dict

        Raises:
            ValueError: If session_id format is invalid
            SessionExistsError: If session already exists
        """
        # Validate session ID
        if not validate_session_id(session_id):
            raise ValueError(
                f"Invalid session_id '{session_id}': "
                "use alphanumeric characters, hyphens, and underscores only"
            )

        # Check if session already exists
        if self.redis.session_exists(session_id):
            raise SessionExistsError(f"Session '{session_id}' already exists")

        # Ensure NanoCtrl is running
        nanoctrl_info = {}
        if ensure_nanoctrl:
            nanoctrl_info = self.nanoctrl.ensure_running(auto_start=True)

        # Create session metadata
        session_data = {
            "session_id": session_id,
            "status": "initializing",
            "redis_url": self.redis.url,
            "ray_address": self.ray.address,
            "nanoctrl_address": self.nanoctrl.address,
        }

        if nanoctrl_info.get("pid"):
            session_data["nanoctrl_pid"] = nanoctrl_info["pid"]

        self.redis.create_session(session_id, session_data)

        logger.info(f"Session '{session_id}' initialized")
        return session_data

    def set_model_config(
        self,
        session_id: str,
        model_path: str,
        **kwargs,
    ):
        """Store model path in session config.

        Parallelism is no longer stored at session level; it is passed
        per-component at deploy time via ``spawn_component(**overrides)``.

        Args:
            session_id: Session identifier
            model_path: Path to model
            **kwargs: Additional session-level config (e.g. max_model_len)

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if not self.redis.session_exists(session_id):
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        # Validate model path (warning only, may exist on cluster)
        if not os.path.exists(model_path):
            logger.warning(
                f"Model path '{model_path}' not found locally "
                "(may exist on Ray cluster nodes)"
            )

        config_updates = {
            "model_path": model_path,
            **kwargs,
        }

        self.redis.update_session_config(session_id, config_updates)

        logger.info(f"Model config updated: model_path={model_path}")

    def spawn_component(
        self,
        session_id: str,
        component_type: str,
        **overrides,
    ) -> str:
        """Spawn a component via Ray job submission.

        Args:
            session_id: Session identifier
            component_type: Component type (route, prefill, decode)
            **overrides: Config overrides

        Returns:
            Ray job ID

        Raises:
            SessionNotFoundError: If session doesn't exist
            ConfigError: If model not set for engines
        """
        if not self.redis.session_exists(session_id):
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        # Load session config
        session_config = self.redis.get_session_config(session_id)

        # Validate model is set (for engines)
        if component_type in ["prefill", "decode"]:
            if "model_path" not in session_config:
                raise ConfigError(
                    f"Model not set for session '{session_id}'. "
                    "Run 'nanoctrl set --model <path>' first"
                )

        # Create placement group for engines
        pg_id = None
        if component_type in ["prefill", "decode"]:
            num_gpus = self._calculate_num_gpus(overrides)

            pg_id = self.ray.create_placement_group(
                session_id,
                component_type,
                num_gpus,
            )

            self.redis.register_placement_group(
                session_id, pg_id, component_type, num_gpus
            )

        # Generate config file
        config_path = self._generate_component_config(
            session_id,
            component_type,
            session_config,
            overrides,
        )

        # Build environment variables
        env_vars = {
            "NANOCTRL_SCOPE": session_id,
            "NANOCTRL_REDIS_URL": self.redis.url,
            "SESSION_ID": session_id,
            "NANOCTRL_ADDRESS": self.nanoctrl.address,
        }

        if pg_id:
            env_vars["NANOOPS_PLACEMENT_GROUP_ID"] = pg_id

        # For route, don't use working_dir - expect binary to be pre-installed
        working_dir = None
        excludes = []

        # Submit Ray job
        job_id = self.ray.submit_component_job(
            session_id,
            component_type,
            config_path,
            env_vars,
            placement_group_id=pg_id,
            working_dir=working_dir,
            excludes=excludes,
        )

        # Register component in Redis
        component_data = {
            "ray_job_id": job_id,
            "placement_group_id": pg_id,
            "config_path": config_path,
            "host": "0.0.0.0",
            "port": allocate_port(session_id, component_type),
            "status": "spawning",
        }

        self.redis.register_component(
            session_id, component_type, job_id, component_data
        )

        logger.info(f"Component {component_type} spawned: {job_id}")
        return job_id

    def wait_for_ready(
        self,
        session_id: str,
        timeout: int = 300,
    ) -> Dict[str, any]:
        """Wait for session to be ready.

        Args:
            session_id: Session identifier
            timeout: Maximum wait time in seconds

        Returns:
            Ready response with endpoints and component info
        """
        return self.health.wait_for_session_ready(session_id, timeout)

    def stop_session(
        self,
        session_id: str,
        cleanup: bool = True,
    ):
        """Stop session and cleanup resources.

        Args:
            session_id: Session identifier
            cleanup: Whether to cleanup Redis keys
        """
        if not self.redis.session_exists(session_id):
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        logger.info(f"Stopping session '{session_id}'")

        # Stop Ray jobs
        components = self.redis.get_session_components(session_id)
        for comp_type, comps in components.items():
            for comp in comps:
                job_id = comp.get("ray_job_id")
                if job_id:
                    try:
                        self.ray.stop_job(job_id)
                        logger.info(f"Stopped {comp_type} job: {job_id}")
                    except Exception as e:
                        logger.warning(f"Failed to stop job {job_id}: {e}")

        # Remove placement groups
        pgs = self.redis.get_placement_groups(session_id)
        for pg in pgs:
            pg_id = pg.get("pg_id")
            if pg_id:
                try:
                    self.ray.cleanup_placement_group(pg_id)
                except Exception as e:
                    logger.warning(f"Failed to remove PG {pg_id}: {e}")

        # Cleanup Redis keys
        if cleanup:
            num_deleted = self.redis.cleanup_session(session_id)
            logger.info(f"Cleaned up {num_deleted} Redis keys")

        logger.info("Session stopped")

    def _calculate_num_gpus(self, overrides: Dict) -> int:
        """Calculate GPU requirements from parallelism overrides.

        Uses NanoDeploy Config field names directly:
        attention_tp, attention_sp, attention_dp.

        Args:
            overrides: Parallelism config (attention_tp, attention_sp, attention_dp, ...)

        Returns:
            Number of GPUs needed
        """
        tp = overrides.get("attention_tp", 1)
        sp = overrides.get("attention_sp", 1)
        dp = overrides.get("attention_dp", 1)

        return tp * sp * dp

    def _generate_component_config(
        self,
        session_id: str,
        component_type: str,
        session_config: Dict,
        overrides: Dict,
    ) -> str:
        """Generate component-specific config file.

        Args:
            session_id: Session identifier
            component_type: Component type
            session_config: Session configuration
            overrides: Config overrides

        Returns:
            Path to generated config file
        """
        if component_type in ["prefill", "decode"]:
            return self._generate_engine_config(
                session_id, component_type, session_config, overrides
            )
        elif component_type == "route":
            return self._generate_route_config(session_id, session_config, overrides)
        else:
            raise ValueError(f"Unknown component type: {component_type}")

    def _generate_engine_config(
        self,
        session_id: str,
        mode: str,
        session_config: Dict,
        overrides: Dict,
    ) -> str:
        """Generate engine_server.py compatible YAML config.

        Parallelism fields (attention_tp, ffn_ep, ...) come from
        ``overrides`` which map 1:1 to NanoDeploy Config fields.

        Args:
            session_id: Session identifier
            mode: Engine mode (prefill, decode)
            session_config: Session configuration (model_path, etc.)
            overrides: Per-component config (parallelism + any extras)

        Returns:
            Path to generated config file
        """
        # Get master address from Ray GCS address
        from urllib.parse import urlparse

        parsed = urlparse(self.ray_gcs_address)
        master_host = parsed.hostname or "localhost"
        master_address = f"{master_host}:6006"

        config = {
            "model": session_config["model_path"],
            "mode": mode,
            # Parallelism — read from overrides (NanoDeploy Config names)
            "attention_tp": overrides.get("attention_tp", 1),
            "attention_sp": overrides.get("attention_sp", 1),
            "attention_dp": overrides.get("attention_dp", 1),
            "ffn_tp": overrides.get("ffn_tp", 1),
            "ffn_dp": overrides.get("ffn_dp", 1),
            "ffn_ep": overrides.get("ffn_ep", 1),
            # Networking
            "host": "0.0.0.0",
            "port": allocate_port(session_id, mode),
            "nanoctrl_address": self.nanoctrl.address.replace("http://", "").replace(
                "https://", ""
            ),
            "ray_address": "auto",
            "master_address": session_config.get("master_address", master_address),
            # Scheduler / memory defaults (can be overridden via session config)
            "loop_count": session_config.get("loop_count", 1),
            "max_num_batched_tokens": session_config.get(
                "max_num_batched_tokens", 16384
            ),
            "max_model_len": session_config.get("max_model_len", 16384),
            "kvcache_block_size": session_config.get("kvcache_block_size", 256),
            "num_kvcache_blocks": session_config.get("num_kvcache_blocks", 15000),
        }

        config_path = f"/tmp/{session_id}_{mode}_config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        logger.debug(f"Generated engine config: {config_path}")
        return config_path

    def _generate_route_config(
        self,
        session_id: str,
        session_config: Dict,
        overrides: Dict,
    ) -> str:
        """Generate NanoRoute TOML config.

        Args:
            session_id: Session identifier
            session_config: Session configuration
            overrides: Config overrides

        Returns:
            Path to generated config file
        """
        model_path = session_config.get("model_path", "/models/unknown")
        model_name = os.path.basename(model_path)

        # Build config content
        config = f"""[server]
host = "0.0.0.0"
port = {allocate_port(session_id, "route")}
model_name = "{model_name}"

[tokenizer]
path = "{model_path}/tokenizer.json"

[engine]
mode = "Disaggregated"
nanoctrl_address = "{self.nanoctrl.address}"
scope = "{session_id}"

[scheduler]
queue_size = 1000
timeout_ms = 30000
"""

        config_path = f"/tmp/{session_id}_route_config.toml"

        with open(config_path, "w") as f:
            f.write(config)

        logger.debug(f"Generated route config: {config_path}")
        return config_path
