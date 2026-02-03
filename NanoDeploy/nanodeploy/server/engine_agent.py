"""Engine Agent Plugin for registering engine info with NanoCtrl control plane."""

import json
from typing import Optional

import httpx
from nanodeploy.config import Config
from nanodeploy.logging import get_logger

logger = get_logger("nanodeploy")


class EngineAgent:
    """Agent that registers engine information with NanoCtrl control plane."""

    def __init__(self, config: Config, engine_component):
        self.config = config
        self.engine = engine_component
        self.nanoctrl_address = config.nanoctrl_address
        self.redis_address: Optional[str] = None

    async def get_redis_address(self) -> Optional[str]:
        """Query NanoCtrl for Redis address."""
        if not self.nanoctrl_address:
            logger.warning(
                "nanoctrl_address not configured, skipping Redis address query"
            )
            return None

        try:
            url = f"http://{self.nanoctrl_address}/get_redis_address"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json={})
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    redis_addr = data.get("redis_address", "")
                    logger.info(f"Got Redis address from NanoCtrl: {redis_addr}")
                    return redis_addr
                else:
                    logger.error(f"Failed to get Redis address: {data}")
                    return None
        except Exception as e:
            logger.error(f"Error querying Redis address from NanoCtrl: {e}")
            return None

    async def register_engine(self) -> bool:
        """Register engine information with NanoCtrl control plane."""
        if not self.nanoctrl_address:
            logger.warning(
                "nanoctrl_address not configured, skipping engine registration"
            )
            return False

        try:
            # Get engine info as JSON
            engine_info_str = self.engine.get_engine_info()
            engine_info = json.loads(engine_info_str)

            # Prepare registration payload
            payload = {
                "engine_id": engine_info.get("id", ""),
                "role": engine_info.get("role", ""),
                "world_size": engine_info.get("world_size", 1),
                "num_blocks": engine_info.get("num_blocks", 0),
                "host": engine_info.get("host", ""),
                "port": engine_info.get("port", 0),
                "peer_addrs": engine_info.get("peer_addrs", []),
            }

            url = f"http://{self.nanoctrl_address}/register_engine"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    logger.info(
                        f"Successfully registered engine {payload['engine_id']} with NanoCtrl"
                    )
                    return True
                else:
                    logger.error(
                        f"Failed to register engine: {data.get('message', 'Unknown error')}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error registering engine with NanoCtrl: {e}")
            return False

    async def startup(self) -> bool:
        """Startup hook: query Redis address and register engine."""
        logger.info(
            "EngineAgent startup: querying Redis address and registering engine"
        )

        # Query Redis address
        self.redis_address = await self.get_redis_address()

        # Register engine
        success = await self.register_engine()

        if success:
            logger.info("EngineAgent startup completed successfully")
        else:
            logger.warning("EngineAgent startup completed with warnings")

        return success
