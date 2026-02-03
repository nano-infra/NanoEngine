"""Engine Agent Plugin for registering engine info with NanoCtrl control plane."""

import asyncio
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
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_stop_event = asyncio.Event()
        self._engine_id: Optional[str] = None
        self._registered = False

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
                    self._engine_id = payload["engine_id"]
                    self._registered = True
                    logger.info(
                        f"Successfully registered engine {payload['engine_id']} with NanoCtrl"
                    )
                    # Start heartbeat task after successful registration
                    self._start_heartbeat()
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

    def _start_heartbeat(self):
        """Start heartbeat task to keep engine registration alive."""
        if not self.nanoctrl_address or not self._registered or not self._engine_id:
            return

        # Stop existing heartbeat task if any
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return

        self._heartbeat_stop_event.clear()

        async def heartbeat_loop():
            """Heartbeat loop: send heartbeat every 15 seconds."""
            try:
                while not self._heartbeat_stop_event.is_set():
                    await asyncio.sleep(15.0)
                    if self._heartbeat_stop_event.is_set():
                        break
                    try:
                        await self._heartbeat_to_nanoctrl()
                    except Exception as e:
                        logger.error(f"Error in heartbeat loop: {e}", exc_info=True)
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
            except Exception as e:
                logger.error(f"Fatal error in heartbeat loop: {e}", exc_info=True)

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        logger.info(
            f"Started heartbeat task for engine {self._engine_id} (interval: 15s)"
        )

    async def _heartbeat_to_nanoctrl(self):
        """Send heartbeat to NanoCtrl to refresh TTL."""
        if not self.nanoctrl_address or not self._engine_id:
            return

        try:
            payload = {"engine_id": self._engine_id}
            url = f"http://{self.nanoctrl_address}/heartbeat_engine"

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

                if data.get("status") == "ok":
                    logger.debug(f"Heartbeat successful for engine {self._engine_id}")
                elif data.get("status") == "not_found":
                    # Engine not found, re-register
                    logger.warning(
                        f"Engine {self._engine_id} not found in NanoCtrl, re-registering..."
                    )
                    self._registered = False
                    await self.register_engine()
                else:
                    logger.warning(
                        f"Heartbeat failed for engine {self._engine_id}: {data.get('message', 'Unknown error')}"
                    )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in heartbeat: {e}")
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}", exc_info=True)

    async def unregister_engine(self) -> bool:
        """Unregister engine from NanoCtrl control plane."""
        if not self.nanoctrl_address or not self._engine_id:
            return False

        # Stop heartbeat task
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            self._heartbeat_stop_event.set()
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        try:
            payload = {"engine_id": self._engine_id}
            url = f"http://{self.nanoctrl_address}/unregister_engine"
            logger.info(f"Unregistering engine {self._engine_id} from NanoCtrl")

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "ok":
                    logger.info(
                        f"Successfully unregistered engine {self._engine_id} from NanoCtrl"
                    )
                    self._registered = False
                    return True
                else:
                    logger.warning(
                        f"Failed to unregister engine: {data.get('message', 'Unknown error')}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error unregistering engine from NanoCtrl: {e}")
            return False

    async def shutdown(self):
        """Shutdown hook: stop heartbeat and unregister engine."""
        logger.info("EngineAgent shutdown: stopping heartbeat and unregistering engine")
        await self.unregister_engine()
