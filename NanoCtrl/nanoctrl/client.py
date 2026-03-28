"""NanoCtrlClient – shared NanoCtrl lifecycle client for all engine types.

Single source of truth for:
- URL normalization (http:// prefix)
- POST /register_engine
- POST /unregister_engine
- POST /heartbeat_engine  (with auto re-register callback on ``not_found``)
- POST /get_engine_info

Usage
-----
Both LLMComponent and EncoderEngine create one instance at startup::

    self._nanoctrl = NanoCtrlClient(config.nanoctrl_address, config.nanoctrl_scope)
    ok = self._nanoctrl.register(engine_id, extra_payload)
    if ok:
        self._nanoctrl.start_heartbeat(on_not_found=self._reregister)

    # on shutdown:
    self._nanoctrl.stop()   # stop heartbeat + unregister
"""

from __future__ import annotations

import logging

import threading
from typing import Callable

import httpx

logger = logging.getLogger("nanoctrl")


class NanoCtrlClient:
    """HTTP client for NanoCtrl engine lifecycle (register / heartbeat / unregister).

    Parameters
    ----------
    address : str
        NanoCtrl server address.  Accepts both ``"host:port"`` and
        ``"http://host:port"`` — the ``http://`` scheme is added when absent.
    scope : str | None
        Optional scope for multi-tenant isolation.  Injected into every
        request payload automatically.
    """

    def __init__(self, address: str, scope: str | None = None) -> None:
        if not address.startswith(("http://", "https://")):
            address = f"http://{address}"
        self._base = address.rstrip("/")
        self._scope = scope
        self._engine_id: str | None = None
        self.registered: bool = False

        self._hb_stop = threading.Event()
        self._hb_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base}/{path}"

    def _body(self, **kwargs) -> dict:
        """Build a request body, injecting scope when configured."""
        if self._scope:
            kwargs["scope"] = self._scope
        return kwargs

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    def check_connection(self) -> None:
        """Verify NanoCtrl is reachable. Raises RuntimeError if not."""
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as c:
                r = c.get(self._base)
                r.raise_for_status()
                logger.info(f"NanoCtrl connection verified: {self._base}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach NanoCtrl at {self._base}: {e}\n"
                f"NanoCtrl must be running before engine startup."
            ) from e

    def register(self, engine_id: str, extra: dict) -> bool:
        """POST /register_engine.

        Parameters
        ----------
        engine_id : str
            Unique engine identifier.
        extra : dict
            Engine-specific fields (role, host, port, peer_addrs, …).
            ``engine_id`` and ``scope`` are injected automatically.

        Returns
        -------
        bool
            ``True`` on success.
        """
        self._engine_id = engine_id
        body = self._body(engine_id=engine_id, **extra)
        try:
            with httpx.Client(timeout=10.0, trust_env=False) as c:
                r = c.post(self._url("register_engine"), json=body)
                r.raise_for_status()
                ok = r.json().get("status") == "ok"
                self.registered = ok
                if ok:
                    logger.info(f"Registered engine {engine_id} with NanoCtrl")
                else:
                    logger.error(
                        f"NanoCtrl registration rejected for {engine_id}: {r.json()}"
                    )
                return ok
        except Exception as e:
            logger.error(f"Failed to register engine {engine_id}: {e}", exc_info=True)
            return False

    def unregister(self) -> bool:
        """POST /unregister_engine for the previously registered engine_id.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on error or if never registered.
        """
        if not self._engine_id:
            return False
        body = self._body(engine_id=self._engine_id)
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as c:
                r = c.post(self._url("unregister_engine"), json=body)
                r.raise_for_status()
                ok = r.json().get("status") == "ok"
                if ok:
                    self.registered = False
                    logger.info(f"Unregistered engine {self._engine_id} from NanoCtrl")
                else:
                    logger.warning(
                        f"NanoCtrl unregister returned non-ok for {self._engine_id}: {r.json()}"
                    )
                return ok
        except Exception as e:
            logger.error(f"Failed to unregister engine {self._engine_id}: {e}")
            return False

    def heartbeat(self) -> str:
        """POST /heartbeat_engine.

        Returns
        -------
        str
            ``"ok"``, ``"not_found"``, or ``"error"``.
        """
        if not self._engine_id:
            return "error"
        body = self._body(engine_id=self._engine_id)
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as c:
                r = c.post(self._url("heartbeat_engine"), json=body)
                r.raise_for_status()
                status = r.json().get("status", "error")
                logger.debug(f"Heartbeat {self._engine_id}: {status}")
                return status
        except Exception as e:
            logger.error(f"Heartbeat error for {self._engine_id}: {e}")
            return "error"

    def get_redis_url(self) -> str | None:
        """POST /get_redis_address → redis_address string, or None on failure."""
        try:
            with httpx.Client(timeout=10.0, trust_env=False) as c:
                r = c.post(self._url("get_redis_address"), json={})
                r.raise_for_status()
                data = r.json()
                addr = data.get("redis_address")
                if addr:
                    # addr is "host:port"; prepend scheme
                    url = addr if addr.startswith("redis://") else f"redis://{addr}"
                    logger.debug(f"Got Redis URL from NanoCtrl: {url}")
                    return url
                logger.error(f"get_redis_address returned unexpected payload: {data}")
        except Exception as e:
            logger.error(f"Failed to get Redis URL from NanoCtrl: {e}")
        return None

    def get_engine_info(self, engine_id: str) -> dict | None:
        """POST /get_engine_info.

        Parameters
        ----------
        engine_id : str
            Target engine to look up.

        Returns
        -------
        dict | None
            Engine info dict on success, ``None`` on failure.
        """
        body = self._body(engine_id=engine_id)
        try:
            with httpx.Client(timeout=5.0, trust_env=False) as c:
                r = c.post(self._url("get_engine_info"), json=body)
                r.raise_for_status()
                data = r.json()
                if data.get("status") == "ok":
                    return data.get("engine_info")
                logger.error(
                    f"get_engine_info for {engine_id} returned: {data.get('status')}"
                )
        except Exception as e:
            logger.error(f"Failed to get engine info for {engine_id}: {e}")
        return None

    # ------------------------------------------------------------------
    # Heartbeat thread
    # ------------------------------------------------------------------

    def start_heartbeat(
        self,
        interval: float = 15.0,
        on_not_found: Callable[[], None] | None = None,
        name: str = "nanoctrl-hb",
    ) -> None:
        """Start a background daemon thread that sends heartbeats every *interval* seconds.

        Parameters
        ----------
        interval : float
            Seconds between heartbeat attempts.
        on_not_found : callable | None
            Called when NanoCtrl responds with ``status=not_found``.
            Typical use: re-register the engine (e.g. after a NanoCtrl restart).
            The callback runs on the heartbeat thread — keep it lightweight.
        name : str
            Thread name (useful for debugging).
        """
        if self._hb_thread is not None and self._hb_thread.is_alive():
            return  # already running
        self._hb_stop.clear()

        def _loop() -> None:
            while not self._hb_stop.wait(interval):
                try:
                    status = self.heartbeat()
                    if status == "not_found" and on_not_found is not None:
                        logger.warning(
                            f"Engine {self._engine_id} not found in NanoCtrl, "
                            "calling on_not_found callback"
                        )
                        on_not_found()
                except Exception as e:
                    logger.error(f"Heartbeat loop error: {e}", exc_info=True)

        self._hb_thread = threading.Thread(target=_loop, name=name, daemon=True)
        self._hb_thread.start()
        logger.info(
            f"Started heartbeat thread '{name}' for engine {self._engine_id} "
            f"(interval={interval}s)"
        )

    def stop_heartbeat(self, timeout: float = 2.0) -> None:
        """Signal the heartbeat thread to stop and wait for it to exit."""
        self._hb_stop.set()
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=timeout)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop heartbeat and unregister from NanoCtrl.

        Call this from the engine's shutdown path.  Safe to call multiple times.
        """
        self.stop_heartbeat(timeout=timeout)
        if self.registered:
            self.unregister()
