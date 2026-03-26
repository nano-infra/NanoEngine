"""
NanoFold ZMQ ROUTER server.

Protocol (packet.fbs ZmqPacket, JSON payload):
  Request:  action=7  payload={"op": "predict"|"embed"|"sample"|"job_status"|"embed_status"|"health", ...}
  Response: action=8  payload={"ok": bool, ...}

ops:
  predict      → {name, sequences, covalent_bonds?, model_name?, use_msa?, use_template?,
                   dtype?, n_cycle?, trimul_kernel?, triatt_kernel?, enable_cache?,
                   seeds, n_sample?, n_step?}
               ← {ok, job_id, embed_id, status}
  embed        → same as predict minus seeds/n_sample/n_step
               ← {ok, embed_id, status}
  sample       → {embed_id, seeds, n_sample?, n_step?}
               ← {ok, job_id, status} | {ok:false, code:404/409, error}
  job_status   → {job_id}
               ← {ok, job_id, status, structures?, confidence?, error?} | {ok:false, code:404}
  embed_status → {embed_id}
               ← {ok, embed_id, status, n_token?} | {ok:false, code:404}
  health       → {}
               ← {ok, gpu, model}
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import zmq
import zmq.asyncio

from .config import NanoFoldConfig
from .job_store import JobStore
from .runner import ProtenixRunner
from .zmq_protocol import ACTION_FOLD_RESPONSE, decode_packet, encode_packet

logger = logging.getLogger(__name__)

# ── Global state (populated in serve()) ───────────────────────────────
_cfg: NanoFoldConfig | None = None
_runner: ProtenixRunner | None = None
_store: JobStore | None = None
_embed_queue: asyncio.Queue | None = None
_sample_queue: asyncio.Queue | None = None
_executor: ThreadPoolExecutor | None = None


# ── NanoCtrl helpers ──────────────────────────────────────────────────


async def _get_redis_url_from_nanoctrl(nanoctrl_url: str) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(f"{nanoctrl_url}/get_redis_address", json={})
            r.raise_for_status()
            addr = r.json().get("redis_address")  # "host:port"
            if addr:
                return f"redis://{addr}" if not addr.startswith("redis://") else addr
            return None
    except Exception as exc:
        logger.warning("Could not fetch Redis URL from NanoCtrl: %s", exc)
        return None


async def _register_with_nanoctrl(cfg: NanoFoldConfig) -> None:
    hostname = socket.gethostname()
    payload = {
        "engine_id": f"fold-{hostname}-{cfg.port}",
        "role": "fold",
        "host": hostname,
        "port": cfg.port,
        "world_size": 1,
        "num_blocks": 0,
        "peer_addrs": [],
        "model_path": cfg.checkpoint_dir,
        "scope": cfg.nanoctrl_scope,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{cfg.nanoctrl_url}/register_engine", json=payload)
            r.raise_for_status()
        logger.info("Registered with NanoCtrl as fold engine.")
    except Exception as exc:
        logger.warning("NanoCtrl registration failed (non-fatal): %s", exc)


async def _unregister_from_nanoctrl(cfg: NanoFoldConfig) -> None:
    hostname = socket.gethostname()
    payload = {
        "engine_id": f"fold-{hostname}-{cfg.port}",
        "scope": cfg.nanoctrl_scope,
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{cfg.nanoctrl_url}/unregister_engine", json=payload)
    except Exception:
        pass


async def _heartbeat_loop(cfg: NanoFoldConfig) -> None:
    hostname = socket.gethostname()
    engine_id = f"fold-{hostname}-{cfg.port}"
    while True:
        await asyncio.sleep(15)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{cfg.nanoctrl_url}/heartbeat_engine",
                    json={"engine_id": engine_id, "scope": cfg.nanoctrl_scope},
                )
        except Exception:
            pass


# ── GPU worker tasks ──────────────────────────────────────────────────


async def _embed_worker() -> None:
    """Serial embed worker — one job at a time on the GPU."""
    while True:
        embed_id, req, fut = await _embed_queue.get()
        await _store.set_embed_running(embed_id)
        try:
            loop = asyncio.get_event_loop()
            _, n_token = await loop.run_in_executor(
                _executor,
                lambda: _runner.run_trunk(
                    name=req.get("name", embed_id),
                    sequences=req.get("sequences", []),
                    covalent_bonds=req.get("covalent_bonds", []),
                    model_name=req.get("model_name", _cfg.model_name),
                    use_msa=req.get("use_msa", False),
                    use_template=req.get("use_template", False),
                    dtype=req.get("dtype", _cfg.dtype),
                    n_cycle=req.get("n_cycle", _cfg.n_cycle),
                    trimul_kernel=req.get("trimul_kernel", _cfg.trimul_kernel),
                    triatt_kernel=req.get("triatt_kernel", _cfg.triatt_kernel),
                    enable_cache=req.get("enable_cache", _cfg.enable_cache),
                ),
            )
            shm_path = str(_runner.shm_dir / embed_id)
            await _store.set_embed_done(embed_id, shm_path, n_token)
            if not fut.done():
                fut.set_result(embed_id)
        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Embed %s failed: %s", embed_id, err)
            await _store.set_embed_error(embed_id, str(exc))
            if not fut.done():
                fut.set_exception(exc)
        finally:
            _embed_queue.task_done()


async def _sample_worker() -> None:
    """Serial diffusion worker — one job at a time on the GPU."""
    while True:
        job_id, embed_id, req, fut = await _sample_queue.get()
        await _store.set_job_running(job_id)
        try:
            loop = asyncio.get_event_loop()
            structures, confidence = await loop.run_in_executor(
                _executor,
                lambda: _runner.run_diffusion(
                    embed_id=embed_id,
                    seeds=req.get("seeds", [101]),
                    n_sample=req.get("n_sample", 1),
                    n_step=req.get("n_step", 200),
                ),
            )
            await _store.set_job_done(job_id, structures, confidence)
            if not fut.done():
                fut.set_result(job_id)
        except Exception as exc:
            err = traceback.format_exc()
            logger.error("Job %s failed: %s", job_id, err)
            await _store.set_job_error(job_id, str(exc))
            if not fut.done():
                fut.set_exception(exc)
        finally:
            _sample_queue.task_done()


# ── Request dispatch ──────────────────────────────────────────────────


async def _dispatch(req: dict) -> dict:
    op = req.get("op", "")

    if op == "health":
        return {"ok": True, "gpu": _cfg.gpu, "model": _cfg.model_name}

    if op == "embed":
        embed_id = await _store.create_embed(req.get("model_name", _cfg.model_name))
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await _embed_queue.put((embed_id, req, fut))
        return {"ok": True, "embed_id": embed_id, "status": "queued"}

    if op == "sample":
        embed_id = req.get("embed_id")
        if not embed_id:
            return {"ok": False, "code": 400, "error": "embed_id required"}
        embed_info = await _store.get_embed(embed_id)
        if not embed_info:
            return {
                "ok": False,
                "code": 404,
                "error": f"embed_id not found: {embed_id}",
            }
        if embed_info["status"] != "done":
            return {
                "ok": False,
                "code": 409,
                "error": f"embed not ready, status={embed_info['status']}",
            }
        job_id = await _store.create_job(embed_id)
        fut = asyncio.get_event_loop().create_future()
        await _sample_queue.put((job_id, embed_id, req, fut))
        return {"ok": True, "job_id": job_id, "status": "queued"}

    if op == "predict":
        embed_id = await _store.create_embed(req.get("model_name", _cfg.model_name))
        job_id = await _store.create_job(embed_id)
        embed_fut: asyncio.Future = asyncio.get_event_loop().create_future()
        await _embed_queue.put((embed_id, req, embed_fut))

        async def _chain() -> None:
            try:
                await embed_fut
                sample_fut: asyncio.Future = asyncio.get_event_loop().create_future()
                await _sample_queue.put((job_id, embed_id, req, sample_fut))
                await sample_fut
            except Exception as exc:
                await _store.set_job_error(job_id, str(exc))

        asyncio.create_task(_chain())
        return {"ok": True, "job_id": job_id, "embed_id": embed_id, "status": "queued"}

    if op == "job_status":
        job_id = req.get("job_id")
        if not job_id:
            return {"ok": False, "code": 400, "error": "job_id required"}
        data = await _store.get_job(job_id)
        if not data:
            return {"ok": False, "code": 404, "error": f"job_id not found: {job_id}"}
        structures = None
        confidence = None
        if data.get("structures_json"):
            structures = json.loads(data["structures_json"])
        if data.get("confidence_json"):
            confidence = json.loads(data["confidence_json"])
        return {
            "ok": True,
            "job_id": job_id,
            "status": data["status"],
            "structures": structures,
            "confidence": confidence,
            "error": data.get("error") or None,
        }

    if op == "embed_status":
        embed_id = req.get("embed_id")
        if not embed_id:
            return {"ok": False, "code": 400, "error": "embed_id required"}
        data = await _store.get_embed(embed_id)
        if not data:
            return {
                "ok": False,
                "code": 404,
                "error": f"embed_id not found: {embed_id}",
            }
        return {
            "ok": True,
            "embed_id": embed_id,
            "status": data["status"],
            "n_token": int(data["n_token"]) if data.get("n_token") else None,
            "error": data.get("error") or None,
        }

    return {"ok": False, "code": 400, "error": f"unknown op: {op!r}"}


# ── ZMQ ROUTER recv loop ──────────────────────────────────────────────


async def _zmq_loop(zmq_socket: Any) -> None:
    """Receive ZmqPacket frames from NanoRoute DEALER(s), dispatch, respond."""
    while True:
        # DEALER → ROUTER: [identity_frame, data_frame]
        parts = await zmq_socket.recv_multipart()
        identity = parts[0]
        data = parts[-1]
        try:
            _action, payload_bytes = decode_packet(data)
            req = json.loads(payload_bytes)
            resp = await _dispatch(req)
        except Exception as exc:
            logger.exception("Dispatch error: %s", exc)
            resp = {"ok": False, "error": str(exc)}
        response_bytes = encode_packet(ACTION_FOLD_RESPONSE, json.dumps(resp).encode())
        await zmq_socket.send_multipart([identity, response_bytes])


# ── Main entry point ──────────────────────────────────────────────────


async def serve(cfg: NanoFoldConfig) -> None:
    global _cfg, _runner, _store, _embed_queue, _sample_queue, _executor

    _cfg = cfg

    # Resolve Redis URL
    redis_url = cfg.redis_url
    if not redis_url:
        redis_url = await _get_redis_url_from_nanoctrl(cfg.nanoctrl_url)
    if not redis_url:
        redis_url = "redis://127.0.0.1:6379"
        logger.warning("Falling back to default Redis URL: %s", redis_url)

    _store = JobStore(
        redis_url=redis_url,
        embed_ttl_s=cfg.embed_ttl_s,
        job_ttl_s=cfg.job_ttl_s,
        scope=cfg.nanoctrl_scope,
    )

    _runner = ProtenixRunner(
        model_name=cfg.model_name,
        checkpoint_dir=cfg.checkpoint_dir,
        dtype=cfg.dtype,
        n_cycle=cfg.n_cycle,
        trimul_kernel=cfg.trimul_kernel,
        triatt_kernel=cfg.triatt_kernel,
        enable_cache=cfg.enable_cache,
        enable_fusion=cfg.enable_fusion,
        enable_tf32=cfg.enable_tf32,
        shm_dir=cfg.shm_dir,
        gpu=cfg.gpu,
    )

    # Single-thread executor: GPU is not re-entrant
    _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nanofold-gpu")
    _embed_queue = asyncio.Queue()
    _sample_queue = asyncio.Queue()

    embed_task = asyncio.create_task(_embed_worker())
    sample_task = asyncio.create_task(_sample_worker())

    await _register_with_nanoctrl(cfg)
    heartbeat_task = asyncio.create_task(_heartbeat_loop(cfg))

    zmq_ctx = zmq.asyncio.Context()
    zmq_socket = zmq_ctx.socket(zmq.ROUTER)
    zmq_socket.bind(f"tcp://*:{cfg.port}")

    logger.info(
        "NanoFold ZMQ server ready on tcp://*:%d (GPU %d, model %s)",
        cfg.port,
        cfg.gpu,
        cfg.model_name,
    )

    try:
        await _zmq_loop(zmq_socket)
    finally:
        heartbeat_task.cancel()
        embed_task.cancel()
        sample_task.cancel()
        await _unregister_from_nanoctrl(cfg)
        await _store.close()
        zmq_socket.close()
        zmq_ctx.term()
        _executor.shutdown(wait=False)
