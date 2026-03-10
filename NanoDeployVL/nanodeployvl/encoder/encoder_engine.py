"""EncoderEngine – standalone vision encoder with EmbeddingPool and NanoCtrl lifecycle.

Single-responsibility: encode images → write to EmbeddingPool → return slot metadata.
RDMA-ready: PeerAgent registers the EmbeddingPool buffer as MR so that
Prefill workers can fetch embeddings via RDMA read.

Lifecycle:
1. Load VisionEncoder (ViT weights)
2. Allocate EmbeddingPool on GPU
3. Start PeerAgent, register MR
4. Register with NanoCtrl as role="encoder"
5. Accept encode requests → write to pool → return VisionSlotMeta
6. Listen for P2P free notifications from Prefill → release slots
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx
import torch

from nanodeploy.context.embedding_pool import EmbeddingPool
from nanodeploy.logging import get_logger

from nanodeployvl.encoder.encoder_config import EncoderConfig
from nanodeployvl.vision.encoder import VisionEncoder

logger = get_logger("encoder_engine")


@dataclass
class VisionSlotMeta:
    """Metadata returned after encoding an image into an EmbeddingPool slot.

    This is the *only* thing the caller needs; the actual tensor stays on
    the encoder GPU and is fetched via RDMA by the Prefill worker.
    """

    encoder_engine_id: str
    slot_idx: int
    num_tokens: int
    hidden_size: int
    max_tokens_per_slot: int


@dataclass
class EncodeRequest:
    """A request to encode one or more images."""

    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    num_images: int


class EncoderEngine:
    """Standalone vision encoder engine.

    Parameters
    ----------
    config : EncoderConfig
        Encoder-specific configuration.
    """

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self.engine_id = str(uuid.uuid4())

        # --- Vision encoder (ViT) ---
        dtype = getattr(torch, config.vision_dtype, torch.bfloat16)
        logger.info("Loading VisionEncoder …")
        self._encoder = VisionEncoder(
            vision_config=config.vision_config,
            model_path=config.model,
            device=config.vision_device,
            dtype=dtype,
        )
        self._spatial_merge_size = config.vision_config.spatial_merge_size

        # --- EmbeddingPool ---
        logger.info("Allocating EmbeddingPool …")
        self.pool = EmbeddingPool(
            num_slots=config.num_slots,
            max_tokens_per_slot=config.max_tokens_per_slot,
            hidden_size=config.hidden_size,
            device=config.vision_device,
            dtype=dtype,
        )

        # --- RDMA PeerAgent ---
        self._peer_agent = None
        self._peer_agent_addr: str | None = None
        self._start_peer_agent()

        # --- P2P free listener (must start before NanoCtrl registration
        #     so that _p2p_port is known when building engine info) ---
        self._p2p_port: int = config.p2p_port
        self._p2p_thread: Optional[threading.Thread] = None
        self._heartbeat_stop_event = threading.Event()
        self._start_p2p_free_listener()

        # --- NanoCtrl registration ---
        self._nanoctrl_registered = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        if config.nanoctrl_address:
            self._register_with_nanoctrl()

        atexit.register(self.shutdown)
        logger.info(
            f"EncoderEngine ready: id={self.engine_id}, "
            f"pool={config.num_slots} slots, device={config.vision_device}"
        )

    # ------------------------------------------------------------------
    # Core encode API
    # ------------------------------------------------------------------

    def encode(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> list[VisionSlotMeta]:
        """Encode images and write embeddings into pool slots.

        Returns a list of ``VisionSlotMeta``, one per image.
        """
        embeddings_list = self._encoder.encode(pixel_values, image_grid_thw)

        results: list[VisionSlotMeta] = []
        for emb in embeddings_list:
            num_tokens = emb.shape[0]
            slot_idx = self.pool.allocate(num_tokens)
            self.pool.write_slot(slot_idx, emb)
            results.append(
                VisionSlotMeta(
                    encoder_engine_id=self.engine_id,
                    slot_idx=slot_idx,
                    num_tokens=num_tokens,
                    hidden_size=self.config.hidden_size,
                    max_tokens_per_slot=self.config.max_tokens_per_slot,
                )
            )

        logger.info(
            f"Encoded {len(results)} images → slots "
            f"{[m.slot_idx for m in results]}, "
            f"pool free={self.pool.available_slots}/{self.config.num_slots}"
        )
        return results

    def free_slots(self, slot_indices: list[int]) -> None:
        """Free embedding slots (called on P2P free from Prefill)."""
        self.pool.free_many(slot_indices)
        logger.info(
            f"Freed slots {slot_indices}, "
            f"pool free={self.pool.available_slots}/{self.config.num_slots}"
        )

    # ------------------------------------------------------------------
    # RDMA / PeerAgent
    # ------------------------------------------------------------------

    def _start_peer_agent(self):
        """Start dlslime PeerAgent and register EmbeddingPool MR."""
        if self.config.nanoctrl_address is None:
            return

        try:
            import dlslime

            start_fn = getattr(dlslime, "start_peer_agent", None)
            if not callable(start_fn):
                logger.warning("dlslime.start_peer_agent not available")
                return

            agent_alias = f"{self.engine_id}:0"
            server_url = self.config.nanoctrl_address
            if not server_url.startswith(("http://", "https://")):
                server_url = f"http://{server_url}"

            available_nics = dlslime.available_nic()
            if not available_nics:
                raise RuntimeError("No available NICs for RDMA")
            nic = available_nics[0]
            scope = self.config.nanoctrl_scope or os.getenv("NANOCTRL_SCOPE")

            self._peer_agent = start_fn(
                alias=agent_alias,
                server_url=server_url,
                device=nic,
                ib_port=1,
                link_type="RoCE",
                qp_num=int(os.environ.get("SLIME_QP_NUM", 1)),
                scope=scope,
            )
            self._peer_agent_addr = agent_alias

            # Register EmbeddingPool buffer as MR
            self.pool.register_mr(self._peer_agent)
            logger.info(f"PeerAgent started: alias={agent_alias}, nic={nic}")
        except Exception as e:
            logger.error(f"Failed to start PeerAgent: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # NanoCtrl lifecycle
    # ------------------------------------------------------------------

    def get_engine_info(self) -> dict:
        if self.config.host in ("0.0.0.0", ""):
            from nanodeploy.context.distributed import get_local_ip

            host = get_local_ip()
        else:
            host = self.config.host

        return {
            "id": self.engine_id,
            "role": "encoder",
            "world_size": 1,
            "num_slots": self.config.num_slots,
            "hidden_size": self.config.hidden_size,
            "host": host,
            "status": "ready",
            "peer_addrs": [self._peer_agent_addr] if self._peer_agent_addr else [],
            "p2p_host": host,
            "p2p_port": self._p2p_port,
        }

    def _register_with_nanoctrl(self):
        if not self.config.nanoctrl_address:
            return
        try:
            info = self.get_engine_info()
            payload = {
                "engine_id": info["id"],
                "role": "encoder",
                "world_size": 1,
                "num_blocks": 0,
                "host": info["host"],
                "port": 0,
                "peer_addrs": info["peer_addrs"],
                "p2p_host": info["p2p_host"],
                "p2p_port": info["p2p_port"],
            }
            if self.config.nanoctrl_scope:
                payload["scope"] = self.config.nanoctrl_scope

            url = f"{self.config.nanoctrl_address}/register_engine"
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "ok":
                    self._nanoctrl_registered = True
                    logger.info(
                        f"Registered encoder engine {self.engine_id} with NanoCtrl"
                    )
                    self._start_heartbeat()
                else:
                    logger.error(f"NanoCtrl registration failed: {data}")
        except Exception as e:
            logger.error(f"Failed to register with NanoCtrl: {e}", exc_info=True)

    def _start_heartbeat(self):
        if not self._nanoctrl_registered:
            return
        self._heartbeat_stop_event.clear()

        def _loop():
            while not self._heartbeat_stop_event.wait(15.0):
                try:
                    payload = {"engine_id": self.engine_id}
                    if self.config.nanoctrl_scope:
                        payload["scope"] = self.config.nanoctrl_scope
                    url = f"{self.config.nanoctrl_address}/heartbeat_engine"
                    with httpx.Client(timeout=5.0, trust_env=False) as client:
                        client.post(url, json=payload)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")

        self._heartbeat_thread = threading.Thread(
            target=_loop, name=f"encoder-hb-{self.engine_id}", daemon=True
        )
        self._heartbeat_thread.start()

    # ------------------------------------------------------------------
    # P2P free listener (ZMQ ROUTER, same pattern as engine_server.py)
    # ------------------------------------------------------------------

    def _start_p2p_free_listener(self):
        """Start a ZMQ ROUTER socket to receive FreeVisionSlots from Prefill engines."""
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        if self._p2p_port:
            sock.bind(f"tcp://{self.config.host}:{self._p2p_port}")
        else:
            self._p2p_port = sock.bind_to_random_port(f"tcp://{self.config.host}")
        logger.info(f"P2P free listener on port {self._p2p_port}")

        def _recv_loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not self._heartbeat_stop_event.is_set():
                events = dict(poller.poll(timeout=500))
                if sock in events:
                    frames = sock.recv_multipart()
                    if len(frames) >= 2:
                        self._handle_p2p_message(frames[-1])

        self._p2p_thread = threading.Thread(
            target=_recv_loop, name=f"encoder-p2p-{self.engine_id}", daemon=True
        )
        self._p2p_thread.start()

    def _handle_p2p_message(self, raw: bytes):
        """Decode a ZmqPacket and handle FreeVisionSlots action."""
        try:
            from nanodeploy.server.zmq_protocol import decode_packet

            action, payload = decode_packet(raw)
            if action == 4:  # FreeVisionSlots
                self._handle_free_vision_slots(payload)
            else:
                logger.warning(f"Unexpected P2P action={action} on encoder")
        except Exception as e:
            logger.error(f"Error handling P2P message: {e}", exc_info=True)

    def _handle_free_vision_slots(self, payload: bytes):
        """Handle FreeVisionSlots FlatBuffer message."""

        from nanodeploy.fbs.FreeVisionSlots import FreeVisionSlots

        buf = bytearray(payload)
        msg = FreeVisionSlots.GetRootAs(buf, 0)
        n = msg.SlotIndicesLength()
        slot_indices = [msg.SlotIndices(i) for i in range(n)]
        source = msg.SourceEngineId()
        if source:
            source = source.decode("utf-8") if isinstance(source, bytes) else source

        logger.info(f"Received FreeVisionSlots from {source}: slots={slot_indices}")
        self.free_slots(slot_indices)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self._heartbeat_stop_event.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        if self._p2p_thread and self._p2p_thread.is_alive():
            self._p2p_thread.join(timeout=2.0)

        if self._nanoctrl_registered and self.config.nanoctrl_address:
            try:
                payload = {"engine_id": self.engine_id}
                if self.config.nanoctrl_scope:
                    payload["scope"] = self.config.nanoctrl_scope
                url = f"{self.config.nanoctrl_address}/unregister_engine"
                with httpx.Client(timeout=5.0, trust_env=False) as client:
                    client.post(url, json=payload)
            except Exception:
                pass

        logger.info(f"EncoderEngine {self.engine_id} shut down.")
