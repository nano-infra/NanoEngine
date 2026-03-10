import asyncio
import ctypes
import traceback
from collections import defaultdict
from typing import Optional

import flatbuffers
import zmq
import zmq.asyncio
from nanodeploy._cpp import deserialize as deserialize_cpp, serialize
from nanodeploy.config import Config

# FlatBuffers imports
from nanodeploy.fbs.SequenceStatus import SequenceStatus
from nanodeploy.fbs.StepOut import (
    StepOutAddSeqId,
    StepOutAddStatus,
    StepOutAddTokenId,
    StepOutAddTokenIds,
    StepOutEnd,
    StepOutStart,
    StepOutStartTokenIdsVector,
)
from nanodeploy.llm_component import LLMComponent
from nanodeploy.logging import get_logger
from nanodeploy.server.zmq_protocol import decode_packet, encode_packet

logger = get_logger()


class BackendService:
    """Backend service that runs LLM engine and processes requests from queue."""

    def __init__(self, engine_component: LLMComponent, results_queue):
        self.engine = engine_component
        self.results_queue = results_queue
        # Track sequences for early free migration
        self._previous_running_seqs: set[int] = set()
        self._freed_sequences: set[int] = set()

    def _send_response(self, action: int, payload: bytes):
        self.results_queue.put((action, payload))

    def _handle_add_request(self, payload: bytes):
        logger.info(f"Handling ADD request, payload size: {len(payload)}")
        c_buffer = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buffer)
        length = len(payload)

        sequences = deserialize_cpp(ptr, length)
        logger.info(f"Deserialized {len(sequences) if sequences else 0} sequences")
        if not sequences:
            logger.warning("No sequences after deserialization")
            return

        logger.info(
            f"Adding {len(sequences)} sequences to engine. First seq_id: {sequences[0].seq_id if sequences else 'N/A'}"
        )
        self.engine.add_request(sequences)
        logger.info(f"Sequences added to engine successfully")

    def _handle_get_info(self):
        resp_payload = self.engine.get_engine_info().encode("utf-8")
        self._send_response(action=2, payload=resp_payload)

    def _handle_free_sequences(self, payload: bytes):
        """Handle P2P free sequence request."""
        try:
            from nanodeploy.fbs.FreeSequences import FreeSequences

            free_req = FreeSequences.GetRootAs(payload, 0)

            seq_ids = []
            seq_ids_length = free_req.SeqIdsLength()
            if seq_ids_length > 0:
                seq_ids = [free_req.SeqIds(i) for i in range(seq_ids_length)]

            source_engine_id = (
                free_req.SourceEngineId().decode("utf-8")
                if free_req.SourceEngineId()
                else ""
            )

            logger.info(
                f"Received P2P free request from {source_engine_id} for {len(seq_ids)} sequences: {seq_ids}"
            )

            # Free sequences from scheduler
            from nanodeploy.engine.sequence import Sequence

            for seq_id in seq_ids:
                try:
                    # Create minimal sequence object with just seq_id for lookup
                    seq = Sequence([])
                    seq.seq_id = seq_id
                    self.engine.free_to_be_migrated(seq)
                except Exception as e:
                    logger.warning(f"Failed to free sequence {seq_id}: {e}")

        except Exception as e:
            logger.error(f"Error handling free sequences: {e}")
            traceback.print_exc()

    def _send_stepout(self, seq_id, token_ids, status):
        """Send step output with one or more tokens."""
        # Normalize to list
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        builder = flatbuffers.Builder(256)

        # Build token_ids vector
        StepOutStartTokenIdsVector(builder, len(token_ids))
        for token_id in reversed(token_ids):  # FlatBuffers builds vectors in reverse
            builder.PrependUint32(token_id)
        token_ids_vector = builder.EndVector()

        StepOutStart(builder)
        StepOutAddSeqId(builder, seq_id)
        if token_ids:
            StepOutAddTokenId(builder, token_ids[-1])  # Backward compatibility
        StepOutAddTokenIds(builder, token_ids_vector)
        StepOutAddStatus(builder, status)
        step_out = StepOutEnd(builder)
        builder.Finish(step_out)
        payload = builder.Output()
        self._send_response(action=0, payload=payload)

    def _send_migration(self, seq):
        buffer_size = (
            1024 * 1024 * 16
        )  # 16MB buffer to prevent overflow during large history migrations
        buffer = ctypes.create_string_buffer(buffer_size)
        ptr = ctypes.addressof(buffer)

        try:
            payload_size = serialize(ptr, buffer_size, [seq], False)
            payload = buffer.raw[:payload_size]
            self._send_response(action=1, payload=payload)
        except Exception as e:
            logger.error(f"Migration Serialize Error: {e}")

    def _send_p2p_free_if_migrated(self, seq):
        """Send P2P free instruction to source engine if sequence was migrated."""
        # Skip if already freed (prevents duplicate free requests)
        if seq.seq_id in self._freed_sequences:
            return

        try:
            # Check if sequence has MIGRATE slot with BlockContext
            from nanodeploy._cpp import BlockContextSlot

            migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
            if migrate_ctx and migrate_ctx.engine_id:
                source_engine_id = migrate_ctx.engine_id
                logger.info(
                    f"Sequence {seq.seq_id} sending P2P free to source engine {source_engine_id}"
                )
                self.engine.send_free_sequences(source_engine_id, [seq.seq_id])
                # Mark as freed to prevent duplicates
                self._freed_sequences.add(seq.seq_id)
            else:
                logger.debug(
                    f"Sequence {seq.seq_id} has no MIGRATE context (not migrated)"
                )
        except Exception as e:
            logger.error(f"Error sending P2P free for seq {seq.seq_id}: {e}")
            traceback.print_exc()


def run_engine_backend(config: Config, requests_queue, results_queue, p2p_port: int):
    """Entry point for the backend engine process."""
    import time

    from nanodeploy.logging import get_logger

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("Starting Engine Backend Process...")
    logger.info("=" * 80)

    # Initialize Engine
    engine = LLMComponent(config)

    # Set p2p_port and re-register
    engine.p2p_port = p2p_port
    if config.nanoctrl_address:
        engine._register_with_nanoctrl()

    service = BackendService(engine, results_queue)

    import queue

    logger.info("Engine Loop Started in Backend Process")
    while True:
        try:
            # Drain queue of all current requests
            while True:
                try:
                    action, payload = requests_queue.get_nowait()
                    try:
                        if action == 1:
                            service._handle_add_request(payload)
                        elif action == 2:
                            service._handle_get_info()
                        elif action == 3:
                            service._handle_free_sequences(payload)
                        else:
                            logger.warning(f"Unknown action: {action}")
                    except Exception as e:
                        logger.error(f"Error handling request action {action}: {e}")
                        traceback.print_exc()
                except queue.Empty:
                    break

            if engine.scheduler.is_finished():
                time.sleep(0.001)
                continue

            dp_seqs, outputs, num_tokens, total_running, sch_lat, post_lat = (
                engine.step()
            )

            # Note: num_tokens is negative for decode (represents capacity usage)
            logger.debug(f"Engine step completed: {total_running} running sequences")

            # Single-pass optimization: merge all sequence processing into one loop
            current_running_seqs = set()
            newly_appeared_seqs = []  # Store newly appeared sequences for early free

            for seqs in dp_seqs:
                for seq in seqs:
                    current_running_seqs.add(seq.seq_id)

                    # Skip system sequences
                    if seq.seq_id < 8:
                        continue

                    # Track newly appeared sequences for early free (decode only)
                    if (
                        engine.config.mode == "decode"
                        and seq.seq_id not in service._previous_running_seqs
                    ):
                        newly_appeared_seqs.append(seq)

                    # Send stepout/migration based on sequence state
                    if seq.is_finished:
                        service._send_stepout(
                            seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED
                        )
                        # Clean up tracking to prevent memory leak
                        service._freed_sequences.discard(seq.seq_id)
                    elif seq.is_to_be_migrated:
                        service._send_migration(seq)
                    elif len(seq.token_ids) > 0:
                        # Send last token for all running sequences (1 token per step in decode)
                        service._send_stepout(
                            seq.seq_id,
                            seq.token_ids[-1],
                            SequenceStatus.RUNNING,
                        )

            # Early free: Process only newly appeared sequences (much faster than full iteration)
            if newly_appeared_seqs:
                from nanodeploy._cpp import BlockContextSlot

                for seq in newly_appeared_seqs:
                    migrate_ctx = seq.block_ctx(BlockContextSlot.MIGRATE)
                    if migrate_ctx and migrate_ctx.engine_id:
                        logger.info(
                            f"Early free: seq {seq.seq_id} migrated from {migrate_ctx.engine_id}"
                        )
                        service._send_p2p_free_if_migrated(seq)

            # Free vision embedding slots on encoder after prefill consumes them
            # (EP-separated mode: notify encoder to reclaim EmbeddingPool slots)
            vision_free_by_encoder: dict[str, list[int]] = defaultdict(list)
            for seqs in dp_seqs:
                for seq in seqs:
                    vs_list = seq.vision_slots
                    if not vs_list:
                        continue
                    for vs in vs_list:
                        vision_free_by_encoder[vs["encoder_engine_id"]].append(
                            vs["slot_idx"]
                        )
                    seq.clear_vision_slots()

            for encoder_id, slot_indices in vision_free_by_encoder.items():
                try:
                    engine.send_free_vision_slots(encoder_id, slot_indices)
                except Exception as e:
                    logger.error(
                        f"Failed to send vision slot free to {encoder_id}: {e}"
                    )

            # Update tracking for next step
            service._previous_running_seqs = current_running_seqs

        except Exception as e:
            logger.error(f"Engine Backend Loop Error: {e}")
            traceback.print_exc()
            time.sleep(1)


class EngineServer:
    def __init__(self, config: Config):
        self.config = config
        import multiprocessing

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()
        self.backend_process = None

    async def serve(self):
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.DEALER)
        listen_addr = f"tcp://*:{self.config.port}"
        socket.bind(listen_addr)

        # Create P2P socket for receiving free instructions (dynamic port)
        p2p_socket = ctx.socket(zmq.DEALER)
        p2p_socket.bind("tcp://*:0")  # Bind to OS-assigned port
        p2p_endpoint = p2p_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        p2p_port = int(p2p_endpoint.split(":")[-1])
        logger.info(f"P2P socket bound to port {p2p_port}")

        # Start Backend Process
        import multiprocessing

        self.backend_process = multiprocessing.Process(
            target=run_engine_backend,
            args=(self.config, self.requests_queue, self.results_queue, p2p_port),
            daemon=True,
        )
        self.backend_process.start()

        # Determine ZMQ connection host for registration logs
        zmq_host = "127.0.0.1" if self.config.host == "0.0.0.0" else self.config.host

        logger.info("=" * 80)
        logger.info("Engine Server (Frontend) Started - Configuration Summary")
        logger.info("=" * 80)
        logger.info(f"Mode:            {self.config.mode}")
        logger.info(f"Model:           {self.config.model}")
        logger.info(f"Bind Address:    {listen_addr} (listening on all interfaces)")
        logger.info(f"ZMQ Connect:     tcp://{zmq_host}:{self.config.port}")
        logger.info(f"P2P Connect:     tcp://{zmq_host}:{p2p_port}")
        logger.info(f"World Size:      {self.config.attn_world_size}")
        logger.info(
            f"Attention:       DP={self.config.attention_dp}, SP={self.config.attention_sp}, TP={self.config.attention_tp}"
        )
        logger.info(
            f"FFN:             DP={self.config.ffn_dp}, EP={self.config.ffn_ep}, TP={self.config.ffn_tp}"
        )
        logger.info(
            f"KV Cache:        {self.config.num_kvcache_blocks} blocks x {self.config.kvcache_block_size} tokens"
        )
        logger.info(
            f"Max Tokens:      {self.config.max_num_batched_tokens} batched, {self.config.max_model_len} model length"
        )
        logger.info(
            f"NanoCtrl:        {self.config.nanoctrl_address or 'Not configured'}"
        )
        logger.info(f"Ray Address:     {self.config.ray_address}")
        logger.info(
            f"Redis Scope:      {self.config.nanoctrl_scope or 'Not set (using NanoCtrl default)'}"
        )
        logger.info("=" * 80)

        async def recv_loop():
            logger.info("Recv loop started, waiting for packets...")
            while True:
                try:
                    data = await socket.recv()
                    action, payload = decode_packet(bytes(data))
                    self.requests_queue.put_nowait((action, payload))
                except zmq.ZMQError as e:
                    if e.errno != zmq.ETERM:
                        logger.error(f"ZMQ recv error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Recv loop error: {e}")
                    traceback.print_exc()

        async def results_loop():
            logger.info("Results loop started, forwarding backend events...")
            loop = asyncio.get_event_loop()
            while True:
                try:
                    action, payload = await loop.run_in_executor(
                        None, self.results_queue.get
                    )
                    data = encode_packet(action, payload)
                    await socket.send(data)
                except Exception as e:
                    logger.error(f"Results loop error: {e}")
                    traceback.print_exc()
                    break

        async def p2p_recv_loop():
            """P2P recv loop for free instructions."""
            logger.info("P2P recv loop started, waiting for free instructions...")
            while True:
                try:
                    data = await p2p_socket.recv()
                    logger.debug(f"Received P2P packet: {len(data)} bytes")
                    action, payload = decode_packet(bytes(data))
                    self.requests_queue.put_nowait((action, payload))
                except zmq.ZMQError as e:
                    if e.errno != zmq.ETERM:
                        logger.error(f"ZMQ P2P recv error: {e}")
                    break
                except Exception as e:
                    logger.error(f"P2P recv loop error: {e}")
                    traceback.print_exc()

        await asyncio.gather(
            recv_loop(),
            results_loop(),
            p2p_recv_loop(),
        )


def main():
    logger.info("=" * 80)
    logger.info("NanoDeploy Engine Server")
    logger.info("=" * 80)

    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(description="NanoDeploy Engine Server")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)
    args = parser.parse_args()
    init_args = {k: v for k, v in vars(args).items() if k != "config"}

    logger.info("Initializing configuration...")
    try:
        config = Config(**init_args)
    except Exception as e:
        logger.error(f"Config Init Error: {e}")
        return

    server = EngineServer(config)

    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()
