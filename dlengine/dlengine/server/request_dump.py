"""Optional on-the-fly request dumper to Redis, for prefix-cache debugging.

When the prefix-cache hit rate looks low, the usual cause is that prompts
diverge earlier than expected (dynamic content injected near the top of the
rendered template, tool list reordering, per-request timestamps/IDs, etc.).
This dumper records the *exact tokenized prompt* of every request so you can
diff consecutive requests and find where the shared prefix breaks.

Enable it with the ``--dump_requests_redis`` serve flag (or the matching
``DLENGINE_DUMP_REQUESTS_REDIS`` env var as a fallback):

- unset / empty / ``0`` / ``false`` -> disabled (zero overhead)
- ``1`` / ``true``                  -> ``redis://127.0.0.1:6379/0``
- any other value                   -> used verbatim as the Redis URL

Other knobs (flag, then env fallback):
- ``--dump_requests_stream`` / ``DLENGINE_DUMP_REQUESTS_STREAM``
  (default ``dlengine:requests``) - stream key
- ``--dump_requests_maxlen`` / ``DLENGINE_DUMP_REQUESTS_MAXLEN``
  (default ``200000``) - approximate stream cap

Each request appends one entry to the Redis stream with fields:
``ts``, ``seq_id``, ``model``, ``affinity_key``, ``prompt_len``,
``token_ids`` (JSON int list) and ``prompt_text`` (decoded, special tokens
kept). Inspect with, e.g.::

    redis-cli XREVRANGE dlengine:requests + - COUNT 5
    redis-cli XLEN dlengine:requests

or in Python::

    import redis, json
    r = redis.from_url("redis://127.0.0.1:6379/0", decode_responses=True)
    rows = r.xrevrange("dlengine:requests", count=50)
    texts = [json.loads(f["token_ids"]) for _id, f in rows]
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Optional

from dlengine.logging import get_logger

logger = get_logger("dlengine")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def _resolve_url(setting: Optional[str]) -> Optional[str]:
    # Explicit config value takes precedence over the env var fallback.
    raw = setting if setting is not None else os.environ.get("DLENGINE_DUMP_REQUESTS_REDIS", "")
    raw = (raw or "").strip()
    if raw.lower() in _FALSE:
        return None
    if raw.lower() in _TRUE:
        return "redis://127.0.0.1:6379/0"
    return raw


class RequestDumper:
    """Fire-and-forget Redis stream writer for inbound request prompts."""

    def __init__(
        self,
        setting: Optional[str] = None,
        stream: Optional[str] = None,
        maxlen: Optional[int] = None,
    ) -> None:
        self._url = _resolve_url(setting)
        self.enabled = self._url is not None
        self._client: Any = None
        self._stream = stream or os.environ.get("DLENGINE_DUMP_REQUESTS_STREAM", "dlengine:requests")
        if maxlen is not None:
            self._maxlen = maxlen
        else:
            try:
                self._maxlen = int(os.environ.get("DLENGINE_DUMP_REQUESTS_MAXLEN", "200000"))
            except ValueError:
                self._maxlen = 200000
        if self.enabled:
            logger.info(
                "Request dump to Redis ENABLED: url=%s stream=%s maxlen~=%d",
                self._url,
                self._stream,
                self._maxlen,
            )

    async def _get_client(self) -> Any:
        if self._client is None:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(self._url, decode_responses=True)
        return self._client

    def dump(self, **fields: Any) -> None:
        """Schedule a non-blocking push of ``fields`` to the Redis stream.

        Safe to call from any coroutine; does nothing when disabled or when
        there is no running event loop.
        """
        if not self.enabled:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._push(fields))

    async def _push(self, fields: dict) -> None:
        try:
            client = await self._get_client()
            entry = {
                k: (v if isinstance(v, (str, int, float)) else json.dumps(v, ensure_ascii=False))
                for k, v in fields.items()
            }
            await client.xadd(self._stream, entry, maxlen=self._maxlen, approximate=True)
        except Exception as e:  # noqa: BLE001 - debug tool must never break serving
            if self.enabled:
                logger.warning("Request dump to Redis disabled after error: %s", e)
                self.enabled = False

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
