"""
Redis-backed job and embed state store.

Connects to NanoCtrl's Redis instance and uses the namespaced key convention:
  {scope}:fold:embed:{embed_id}   Hash  TTL=embed_ttl_s
  {scope}:fold:job:{job_id}       Hash  TTL=job_ttl_s
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis.asyncio as aioredis


def _scoped(scope: str | None, *parts: str) -> str:
    key = ":".join(parts)
    return f"{scope}:{key}" if scope else key


class JobStore:
    def __init__(
        self,
        redis_url: str,
        embed_ttl_s: int = 3600,
        job_ttl_s: int = 86400,
        scope: str | None = None,
    ) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url, decode_responses=True
        )
        self.embed_ttl = embed_ttl_s
        self.job_ttl = job_ttl_s
        self.scope = scope

    # ── Embed ─────────────────────────────────────────────────────────

    def _embed_key(self, embed_id: str) -> str:
        return _scoped(self.scope, "fold", "embed", embed_id)

    async def create_embed(
        self,
        model_name: str,
        shm_path: str | None = None,
    ) -> str:
        embed_id = uuid.uuid4().hex
        key = self._embed_key(embed_id)
        await self._redis.hset(
            key,
            mapping={
                "status": "queued",
                "model_name": model_name,
                "shm_path": shm_path or "",
                "n_token": "",
                "error": "",
                "created_at": str(int(time.time())),
            },
        )
        await self._redis.expire(key, self.embed_ttl)
        return embed_id

    async def get_embed(self, embed_id: str) -> dict[str, Any] | None:
        key = self._embed_key(embed_id)
        data = await self._redis.hgetall(key)
        return data if data else None

    async def set_embed_running(self, embed_id: str) -> None:
        await self._redis.hset(self._embed_key(embed_id), "status", "running")

    async def set_embed_done(self, embed_id: str, shm_path: str, n_token: int) -> None:
        key = self._embed_key(embed_id)
        await self._redis.hset(
            key,
            mapping={"status": "done", "shm_path": shm_path, "n_token": str(n_token)},
        )
        await self._redis.expire(key, self.embed_ttl)

    async def set_embed_error(self, embed_id: str, error: str) -> None:
        await self._redis.hset(
            self._embed_key(embed_id),
            mapping={"status": "error", "error": error[:2000]},
        )

    # ── Job ───────────────────────────────────────────────────────────

    def _job_key(self, job_id: str) -> str:
        return _scoped(self.scope, "fold", "job", job_id)

    async def create_job(self, embed_id: str) -> str:
        job_id = uuid.uuid4().hex
        key = self._job_key(job_id)
        await self._redis.hset(
            key,
            mapping={
                "status": "queued",
                "embed_id": embed_id,
                "structures_json": "",
                "confidence_json": "",
                "error": "",
                "created_at": str(int(time.time())),
            },
        )
        await self._redis.expire(key, self.job_ttl)
        return job_id

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        key = self._job_key(job_id)
        data = await self._redis.hgetall(key)
        return data if data else None

    async def set_job_running(self, job_id: str) -> None:
        await self._redis.hset(self._job_key(job_id), "status", "running")

    async def set_job_done(
        self,
        job_id: str,
        structures: list[dict],
        confidence: list[dict],
    ) -> None:
        key = self._job_key(job_id)
        await self._redis.hset(
            key,
            mapping={
                "status": "done",
                "structures_json": json.dumps(structures),
                "confidence_json": json.dumps(confidence),
            },
        )
        await self._redis.expire(key, self.job_ttl)

    async def set_job_error(self, job_id: str, error: str) -> None:
        await self._redis.hset(
            self._job_key(job_id),
            mapping={"status": "error", "error": error[:2000]},
        )

    async def close(self) -> None:
        await self._redis.aclose()
