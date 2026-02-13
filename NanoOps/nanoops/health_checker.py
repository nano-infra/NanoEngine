"""Health checker for NanoOps components."""

import logging
import time
from typing import Dict, List

from tqdm import tqdm

from .exceptions import HealthCheckTimeout
from .utils import check_port_listening

logger = logging.getLogger(__name__)


class HealthChecker:
    """Multi-signal health monitoring for NanoInfra components."""

    def __init__(self, redis_client, nanoctrl_client, ray_manager):
        """Initialize health checker.

        Args:
            redis_client: RedisClient instance
            nanoctrl_client: NanoCtrlClient instance
            ray_manager: RayJobManager instance
        """
        self.redis = redis_client
        self.nanoctrl = nanoctrl_client
        self.ray = ray_manager

    def wait_for_session_ready(
        self,
        session_id: str,
        timeout: int = 300,
    ) -> Dict[str, any]:
        """Wait for all session components to be ready.

        Args:
            session_id: Session identifier
            timeout: Maximum wait time in seconds

        Returns:
            Dict with status and component info:
            {
                "status": "ready" | "timeout",
                "endpoints": {"route": "http://..."},
                "components": {...}
            }

        Raises:
            HealthCheckTimeout: If components don't become ready within timeout
        """
        logger.info(f"Waiting for session '{session_id}' components to be ready...")

        start_time = time.time()

        # Get expected components from Redis
        all_components = self.redis.get_session_components(session_id)

        # Flatten component list
        component_list = []
        for comp_type, comps in all_components.items():
            for comp in comps:
                component_list.append(
                    {
                        "type": comp_type,
                        "id": comp.get("component_id"),
                        "ray_job_id": comp.get("ray_job_id"),
                    }
                )

        if not component_list:
            raise HealthCheckTimeout("No components found for session")

        logger.info(f"Found {len(component_list)} components to check")

        # Progress bar
        with tqdm(total=len(component_list), desc="Component health checks") as pbar:
            ready_components = set()

            while time.time() - start_time < timeout:
                all_ready = True

                for comp in component_list:
                    comp_key = f"{comp['type']}:{comp['id']}"

                    # Skip if already marked ready
                    if comp_key in ready_components:
                        continue

                    # Run health checks
                    is_ready = self._check_component_health(session_id, comp)

                    if is_ready:
                        ready_components.add(comp_key)
                        pbar.update(1)
                        logger.debug(f"Component ready: {comp_key}")
                    else:
                        all_ready = False

                if all_ready:
                    logger.info("All components are ready!")
                    return self._build_ready_response(session_id, all_components)

                time.sleep(2)  # Poll every 2 seconds

        # Timeout
        raise HealthCheckTimeout(
            f"Components did not become ready within {timeout} seconds. "
            f"Ready: {len(ready_components)}/{len(component_list)}"
        )

    def _check_component_health(self, session_id: str, component: Dict) -> bool:
        """Check if a single component is healthy.

        Args:
            session_id: Session identifier
            component: Component info dict

        Returns:
            True if healthy, False otherwise
        """
        comp_type = component["type"]
        comp_id = component["id"]
        ray_job_id = component.get("ray_job_id")

        try:
            # Check 1: Ray job status
            if ray_job_id:
                job_status = self.ray.get_job_status(ray_job_id)
                if job_status != "RUNNING":
                    # Skip permanently failed jobs (don't retry them)
                    if job_status in ["FAILED", "STOPPED"]:
                        logger.debug(
                            f"Skipping permanently failed job: {comp_type}:{comp_id} ({job_status})"
                        )
                        # Mark as checked so we don't keep logging it
                        return False

                    logger.debug(f"{comp_type}:{comp_id} Ray job status: {job_status}")
                    return False

            # Check 2: NanoCtrl registration (for engines)
            if comp_type in ["prefill", "decode"]:
                engines = self.nanoctrl.list_engines(session_id, role=comp_type)
                if len(engines) == 0:
                    logger.debug(
                        f"{comp_type}:{comp_id} not registered with NanoCtrl yet"
                    )
                    return False

            # Check 3: TCP port listening (for route)
            if comp_type == "route":
                comp_info = self.redis.get_component_info(
                    session_id, comp_type, comp_id
                )
                port = comp_info.get("port")
                host = comp_info.get("host", "localhost")

                # Handle 0.0.0.0 -> localhost
                if host == "0.0.0.0":
                    host = "localhost"

                if port and not check_port_listening(host, port):
                    logger.debug(
                        f"{comp_type}:{comp_id} port {host}:{port} not listening"
                    )
                    return False

            return True

        except Exception as e:
            logger.warning(f"Health check error for {comp_type}:{comp_id}: {e}")
            return False

    def _build_ready_response(
        self, session_id: str, components: Dict[str, List[Dict]]
    ) -> Dict[str, any]:
        """Build ready response with endpoint URLs.

        Args:
            session_id: Session identifier
            components: Component info from Redis

        Returns:
            Dict with status, endpoints, and component info
        """
        # Get route endpoint
        route_endpoint = None
        if components.get("route"):
            route_comp = components["route"][0]
            host = route_comp.get("host", "localhost")
            port = route_comp.get("port", 3001)

            # Handle 0.0.0.0 -> localhost
            if host == "0.0.0.0":
                host = "localhost"

            route_endpoint = f"http://{host}:{port}/v1/completions"

        # Get engine counts
        prefill_engines = self.nanoctrl.list_engines(session_id, role="prefill")
        decode_engines = self.nanoctrl.list_engines(session_id, role="decode")

        return {
            "status": "ready",
            "endpoints": {"route": route_endpoint} if route_endpoint else {},
            "components": {
                "route": components.get("route", []),
                "prefill": prefill_engines,
                "decode": decode_engines,
            },
            "summary": {
                "route_running": len(components.get("route", [])) > 0,
                "prefill_engines": len(prefill_engines),
                "decode_engines": len(decode_engines),
            },
        }
