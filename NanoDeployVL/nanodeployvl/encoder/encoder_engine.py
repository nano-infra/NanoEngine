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

import torch

from nanodeploy.context.embedding_pool import EmbeddingPool
from nanodeploy.logging import get_logger
from nanodeploy.server.nanoctrl_client import NanoCtrlClient

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

        # --- ZMQ encode service (NanoRoute connects here) ---
        self._zmq_port: int = config.zmq_port
        self._zmq_thread: Optional[threading.Thread] = None
        self._processor: Optional["ImageProcessor"] = None
        self._start_zmq_encode_service()

        # --- Warmup ImageProcessor (loads HF tokenizer/processor) ---
        if self._processor is None:
            from nanodeployvl.vision.processor import ImageProcessor

            self._processor = ImageProcessor(config.model)
            logger.info("ImageProcessor pre-loaded during init")

        # --- NanoCtrl lifecycle client ---
        self._nanoctrl: NanoCtrlClient | None = None
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
            scope = self.config.nanoctrl_scope

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
            "port": self._zmq_port,
            "status": "ready",
            "peer_addrs": [self._peer_agent_addr] if self._peer_agent_addr else [],
            "p2p_host": host,
            "p2p_port": self._p2p_port,
        }

    def _register_with_nanoctrl(self):
        if not self.config.nanoctrl_address:
            return

        if self._nanoctrl is None:
            self._nanoctrl = NanoCtrlClient(
                self.config.nanoctrl_address, self.config.nanoctrl_scope
            )

        info = self.get_engine_info()
        extra = {
            "role": "encoder",
            "world_size": 1,
            "num_blocks": 0,
            "host": info["host"],
            "port": info["port"],
            "peer_addrs": info["peer_addrs"],
            "p2p_host": info["p2p_host"],
            "p2p_port": info["p2p_port"],
        }
        ok = self._nanoctrl.register(self.engine_id, extra)
        if ok:
            self._nanoctrl.start_heartbeat(name=f"encoder-hb-{self.engine_id}")

    # ------------------------------------------------------------------
    # ZMQ encode service (NanoRoute → EncoderEngine)
    # ------------------------------------------------------------------

    def _start_zmq_encode_service(self):
        """Start a ZMQ ROUTER socket to accept encode requests from NanoRoute.

        Protocol (JSON over ZmqPacket):
        - Request  action=5: {"messages": [...], "image_urls": [...]}
        - Response action=6: {"input_ids": [...], "vision_slots": [...]}
        """
        import zmq

        from nanodeployvl.vision.processor import ImageProcessor

        # Lazy-load processor (shares model path with encoder)
        self._processor = ImageProcessor(self.config.model)

        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        if self._zmq_port:
            sock.bind(f"tcp://{self.config.host}:{self._zmq_port}")
        else:
            self._zmq_port = sock.bind_to_random_port(f"tcp://{self.config.host}")
        logger.info(f"ZMQ encode service on port {self._zmq_port}")

        def _serve_loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not self._heartbeat_stop_event.is_set():
                events = dict(poller.poll(timeout=500))
                if sock in events:
                    frames = sock.recv_multipart()
                    if len(frames) >= 2:
                        identity = frames[0]
                        reply = self._handle_encode_request(frames[-1])
                        sock.send_multipart([identity, reply])

        self._zmq_thread = threading.Thread(
            target=_serve_loop, name=f"encoder-zmq-{self.engine_id}", daemon=True
        )
        self._zmq_thread.start()

    def _handle_encode_request(self, raw: bytes) -> bytes:
        """Process a single encode request and return the response.

        Request (ZmqPacket action=5, JSON payload):
            {"messages": [{"role": ..., "content": ...}, ...]}

        Response (ZmqPacket action=6, JSON payload):
            {"input_ids": [...], "vision_slots": [{...}, ...]}
        """
        import time

        from nanodeploy.server.zmq_protocol import decode_packet, encode_packet

        t_recv = time.perf_counter()
        try:
            action, payload = decode_packet(raw)
            if action != 5:
                return self._encode_error_response(f"Unexpected action={action}")

            req = json.loads(payload)
            messages = req.get("messages", [])
            t_parse = time.perf_counter()
            logger.info(
                f"[ENCODE_TIMING] zmq_recv_to_parse={t_parse-t_recv:.3f}s, payload_bytes={len(raw)}"
            )

            return self._process_encode(messages)

        except Exception as e:
            logger.error(f"Encode request error: {e}", exc_info=True)
            return self._encode_error_response(str(e))

    def _process_encode(self, messages: list[dict]) -> bytes:
        """Full pipeline: messages → chat template → tokenize → encode → response."""
        import io
        import time

        import httpx as _httpx
        from nanodeploy.server.zmq_protocol import encode_packet
        from PIL import Image as _Image

        assert self._processor is not None
        t0 = time.perf_counter()

        # 1. Parse messages into HF format + image URLs
        hf_messages: list[dict] = []
        image_urls: list[str] = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                hf_messages.append({"role": msg["role"], "content": content})
                continue
            hf_parts: list[dict] = []
            for part in content:
                if part.get("type") == "text":
                    hf_parts.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    image_urls.append(part["image_url"]["url"])
                    hf_parts.append({"type": "image"})
            hf_messages.append({"role": msg["role"], "content": hf_parts})
        t1 = time.perf_counter()

        # 2. Download images (synchronous, in encode service thread)
        images: list = []
        if image_urls:
            with _httpx.Client(timeout=30.0, follow_redirects=True) as client:
                for url in image_urls:
                    if url.startswith("data:"):
                        import base64

                        _, encoded = url.split(",", 1)
                        data = base64.b64decode(encoded)
                        images.append(_Image.open(io.BytesIO(data)).convert("RGB"))
                    elif url.startswith(("http://", "https://")):
                        resp = client.get(url)
                        resp.raise_for_status()
                        images.append(
                            _Image.open(io.BytesIO(resp.content)).convert("RGB")
                        )
                    else:
                        images.append(_Image.open(url).convert("RGB"))
        t2 = time.perf_counter()
        logger.info(
            f"[ENCODE_TIMING] parse={t1-t0:.3f}s, download/decode={t2-t1:.3f}s, n_images={len(images)}, n_urls={len(image_urls)}"
        )

        # 3. Apply chat template
        prompt_text = self._processor.apply_chat_template(
            hf_messages, add_generation_prompt=True
        )
        t3 = time.perf_counter()

        # 4. Tokenize text + preprocess images
        if images:
            processed = self._processor.process(text=prompt_text, images=images)
            input_ids = processed["input_ids"].squeeze(0).tolist()
            pixel_values = processed["pixel_values"].to(
                device=self.config.vision_device,
                dtype=getattr(torch, self.config.vision_dtype),
            )
            image_grid_thw = processed["image_grid_thw"].to(
                device=self.config.vision_device
            )
        else:
            input_ids = self._processor.get_token_ids(prompt_text)
            pixel_values = None
            image_grid_thw = None
        t4 = time.perf_counter()
        logger.info(
            f"[ENCODE_TIMING] chat_template={t3-t2:.3f}s, preprocess={t4-t3:.3f}s, n_input_ids={len(input_ids)}, pixel_values={'None' if pixel_values is None else list(pixel_values.shape)}, image_grid_thw={image_grid_thw}"
        )

        # 5. Encode images → EmbeddingPool slots
        vision_slots: list[dict] = []
        if pixel_values is not None:
            metas = self.encode(pixel_values, image_grid_thw)
            vision_slots = [
                {
                    "encoder_engine_id": m.encoder_engine_id,
                    "slot_idx": m.slot_idx,
                    "num_tokens": m.num_tokens,
                    "hidden_size": m.hidden_size,
                    "max_tokens_per_slot": m.max_tokens_per_slot,
                }
                for m in metas
            ]
        t5 = time.perf_counter()

        # 6. Build response
        resp = json.dumps(
            {
                "input_ids": input_ids,
                "vision_slots": vision_slots,
            }
        ).encode()
        t6 = time.perf_counter()
        logger.info(
            f"[ENCODE_TIMING] vit_encode={t5-t4:.3f}s, json_serialize={t6-t5:.3f}s, resp_bytes={len(resp)}, TOTAL={t6-t0:.3f}s"
        )
        return encode_packet(action=6, payload=resp)

    def _encode_error_response(self, error_msg: str) -> bytes:
        from nanodeploy.server.zmq_protocol import encode_packet

        resp = json.dumps({"error": error_msg}).encode()
        return encode_packet(action=6, payload=resp)

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
        # Signal zmq/p2p serve loops to exit (they poll this event)
        self._heartbeat_stop_event.set()
        if self._p2p_thread and self._p2p_thread.is_alive():
            self._p2p_thread.join(timeout=2.0)
        if self._zmq_thread and self._zmq_thread.is_alive():
            self._zmq_thread.join(timeout=2.0)

        # Stop heartbeat thread and unregister from NanoCtrl
        if self._nanoctrl:
            self._nanoctrl.stop()

        logger.info(f"EncoderEngine {self.engine_id} shut down.")
