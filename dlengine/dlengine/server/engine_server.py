import asyncio
import os
import traceback
from collections import defaultdict
from typing import Optional

import flatbuffers
import zmq
import zmq.asyncio

from dlengine._cpp import Sequence
from dlengine.config import Config

# FlatBuffers imports
from dlengine.fbs.SequenceStatus import SequenceStatus
from dlengine.fbs.StepOut import (
    StepOutAddSeqId,
    StepOutAddStatus,
    StepOutAddTokenId,
    StepOutAddTokenIds,
    StepOutEnd,
    StepOutStart,
    StepOutStartTokenIdsVector,
)
from dlengine.llm_component import LLMComponent
from dlengine.logging import get_logger
from dlengine.server.zmq_protocol import decode_packet, encode_packet

logger = get_logger()

# Internal (backend -> frontend results_queue) action code for one step's
# batched stepouts. The frontend unpacks the batch and forwards each entry
# over zmq as a regular action-0 StepOut, so the wire protocol seen by
# clients (ZmqEngineWorker, DLRouter) is unchanged.
_ACTION_STEPOUT_BATCH = 4

# client -> engine: ABORT (stop generating for the given seq_ids and free their
# KV blocks). Reuses the FreeSequences flatbuffer payload (seq_ids list).
_ACTION_ABORT = 5

# client -> engine: fetch Prometheus-format metrics from the backend process.
_ACTION_GET_METRICS = 6


def build_stepout_payload(seq_id, token_ids, status) -> bytes:
    """Build a StepOut flatbuffer payload for one sequence."""
    if isinstance(token_ids, int):
        token_ids = [token_ids]

    builder = flatbuffers.Builder(256)

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
    return builder.Output()


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
        from dlengine.server import pd

        try:
            sequences = pd.decode_add_requests(payload)
        except Exception:
            sequences = [pd.decode_migration_bytes(payload)]
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

    def _handle_get_metrics(self):
        """Fetch Prometheus-format metrics from the engine metrics_manager."""
        try:
            metrics_manager = getattr(self.engine, "metrics_manager", None)
            if metrics_manager is None:
                resp_payload = b""
            else:
                resp_payload = metrics_manager.to_prometheus().encode("utf-8")
            self._send_response(action=_ACTION_GET_METRICS, payload=resp_payload)
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            self._send_response(action=_ACTION_GET_METRICS, payload=b"")

    def _handle_free_sequences(self, payload: bytes):
        """Handle P2P free sequence request."""
        try:
            from dlengine.fbs.FreeSequences import FreeSequences

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

    def _handle_abort(self, payload: bytes):
        """Abort sequences and notify clients with a FINISHED stepout.

        Called between steps (the backend loop defers aborts while a forward is
        in flight, since aborting frees KV blocks the forward may still touch).
        """
        try:
            from dlengine.fbs.FreeSequences import FreeSequences

            abort_req = FreeSequences.GetRootAs(payload, 0)
            n = abort_req.SeqIdsLength()
            seq_ids = [abort_req.SeqIds(i) for i in range(n)] if n > 0 else []
            if not seq_ids:
                return

            aborted = self.engine.abort(seq_ids)
            logger.info(f"Aborted {len(aborted)}/{len(seq_ids)} sequences: {aborted}")

            # Emit a FINISHED stepout for each aborted sequence so any client
            # (OpenAI server, DLRouter) tears down the request cleanly. The
            # aborted seq has been removed from the running set, so the normal
            # step emit loop would not otherwise report it.
            for seq_id in aborted:
                self._send_stepout(seq_id, [], SequenceStatus.FINISHED)
                self._freed_sequences.discard(seq_id)
        except Exception as e:
            logger.error(f"Error handling abort: {e}")
            traceback.print_exc()

    def _send_stepout(self, seq_id, token_ids, status):
        """Send step output with one or more tokens."""
        payload = build_stepout_payload(seq_id, token_ids, status)
        self._send_response(action=0, payload=payload)

    def _send_stepout_batch(self, entries):
        """Send one step's worth of stepouts as a single queue put.

        ``entries`` is a list of (seq_id, token_ids, status) tuples. Batching
        moves the per-seq flatbuffer building and zmq sends to the frontend
        process, off the engine step loop's critical path (one mp.Queue put
        per step instead of one per running sequence).
        """
        if entries:
            self._send_response(action=_ACTION_STEPOUT_BATCH, payload=entries)

    def _send_migration(self, seq):
        try:
            from dlengine._cpp import encode_migration_request

            payload = bytes(encode_migration_request(seq))
            self._send_response(action=1, payload=payload)
        except Exception as e:
            logger.error(f"Migration Serialize Error: {e}")

    def _send_p2p_free_if_migrated(self, seq):
        """Send P2P free instruction to source engine if sequence was migrated."""
        # Skip if already freed (prevents duplicate free requests)
        if seq.seq_id in self._freed_sequences:
            return

        try:
            source_engine_id = seq.migrate_engine_id()
            if source_engine_id:
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

    from dlengine.logging import get_logger

    logger = get_logger()
    logger.info("=" * 80)
    logger.info("Starting Engine Backend Process...")
    logger.info("=" * 80)

    # Initialize Engine
    engine = LLMComponent(config)

    # Set p2p_port and re-register
    engine.p2p_port = p2p_port
    if config.ctrl_address:
        engine._register_with_nanoctrl()

    service = BackendService(engine, results_queue)

    import queue

    logger.info("Engine Loop Started in Backend Process")

    # Loop-phase timing (outside engine.step(), which has its own breakdown in
    # the heartbeat): drain = inbound request handling (add/deserialize),
    # emit = step_complete + stepout/migration/vision-free emission. Logged
    # every ~5s. With the pipelined loop below, drain and emit run while the
    # next forward is already executing on the workers, so they no longer
    # show up as GPU idle gap.
    _lp_drain_ms = 0.0
    _lp_emit_ms = 0.0
    _lp_steps = 0
    _lp_last_log = time.time()

    # Pipelined step loop: after waiting on step N's replies and running the
    # (cheap) postprocess, immediately schedule and submit step N+1 so the
    # GPU starts working again; then do step N's bookkeeping (token counting,
    # heartbeat, stepout emission) and the request-queue drain in the shadow
    # of step N+1's forward. The serial critical path between two forwards
    # shrinks to wait + postprocess + schedule + serialize + submit.
    pending = None  # in-flight PendingStep (forward submitted, not yet waited)
    deferred_frees = []  # free_sequences payloads parked while a forward is in flight
    deferred_aborts = []  # abort payloads parked while a forward is in flight

    while True:
        try:
            # Drain queue of all current requests (overlapped with the
            # in-flight forward when pending is set)
            _t_drain = time.perf_counter()
            while True:
                try:
                    action, payload = requests_queue.get_nowait()
                    try:
                        if action == 1:
                            service._handle_add_request(payload)
                        elif action == 2:
                            service._handle_get_info()
                        elif action == _ACTION_GET_METRICS:
                            service._handle_get_metrics()
                        elif action == 3:
                            # Freeing sequences mutates scheduler/block state;
                            # unsafe while those seqs may be in the in-flight
                            # batch. Park until the forward completes.
                            if pending is not None:
                                deferred_frees.append(payload)
                            else:
                                service._handle_free_sequences(payload)
                        elif action == _ACTION_ABORT:
                            # Aborting frees KV blocks the in-flight forward may
                            # still touch; park until the forward completes
                            # (same constraint as free_sequences above).
                            if pending is not None:
                                deferred_aborts.append(payload)
                            else:
                                service._handle_abort(payload)
                        else:
                            logger.warning(f"Unknown action: {action}")
                    except Exception as e:
                        logger.error(f"Error handling request action {action}: {e}")
                        traceback.print_exc()
                except queue.Empty:
                    break
            _lp_drain_ms += (time.perf_counter() - _t_drain) * 1000

            if pending is None:
                if engine.scheduler.is_finished():
                    time.sleep(0.001)
                    continue
                pending = engine.step_begin()

            # Wait for the in-flight forward and apply its tokens.
            result = engine.step_finish(pending)
            done = pending
            pending = None

            # No forward in flight: safe to apply parked frees/aborts before the
            # next schedule sees (and could re-batch) those sequences.
            if deferred_frees:
                for payload in deferred_frees:
                    service._handle_free_sequences(payload)
                deferred_frees.clear()
            if deferred_aborts:
                for payload in deferred_aborts:
                    service._handle_abort(payload)
                deferred_aborts.clear()

            # Kick off the next step's forward before doing step N's
            # bookkeeping, so the GPU is busy while we count/emit below.
            if not engine.scheduler.is_finished():
                pending = engine.step_begin()

            _t_emit = time.perf_counter()
            engine.step_complete(done, result)

            logger.debug(f"Engine step completed: {result.real_bs} running sequences")
            # Single-pass optimization: merge all sequence processing into one loop.
            # NOTE: avoid seq.token_ids here -- each access copies the full C++
            # token vector into a Python list, which costs tens of ms/step at
            # full batch. Use scalar properties (seq_id/last_token/num_tokens).
            track_running = engine.config.mode == "decode"
            current_running_seqs = set()
            newly_appeared_seqs = []  # Store newly appeared sequences for early free
            stepout_batch = []  # (seq_id, token_id, status) for this step
            vision_free_by_encoder: dict[str, list[int]] = defaultdict(list)

            for seqs in result.dp_seqs:
                for seq in seqs:
                    seq_id = seq.seq_id

                    # Free vision embedding slots on encoder after prefill consumes
                    # them (EP-separated mode: encoder reclaims EmbeddingPool slots)
                    vs_list = seq.vision_slots
                    if vs_list:
                        for vs in vs_list:
                            vision_free_by_encoder[vs["encoder_engine_id"]].append(
                                vs["slot_idx"]
                            )
                        seq.clear_vision_slots()

                    if track_running:
                        current_running_seqs.add(seq_id)

                    # Skip system sequences
                    if seq_id < 8:
                        continue

                    # Track newly appeared sequences for early free (decode only)
                    if track_running and seq_id not in service._previous_running_seqs:
                        newly_appeared_seqs.append(seq)

                    # Send stepout/migration based on sequence state
                    if seq.is_finished:
                        stepout_batch.append(
                            (seq_id, seq.last_token, SequenceStatus.FINISHED)
                        )
                        # Clean up tracking to prevent memory leak
                        service._freed_sequences.discard(seq_id)
                    elif seq.is_to_be_migrated:
                        service._send_migration(seq)
                    elif seq.num_tokens > 0:
                        # Send last token for all running sequences (1 token per step in decode)
                        stepout_batch.append(
                            (seq_id, seq.last_token, SequenceStatus.RUNNING)
                        )

            service._send_stepout_batch(stepout_batch)

            # Early free: Process only newly appeared sequences (much faster than full iteration)
            if newly_appeared_seqs:
                for seq in newly_appeared_seqs:
                    source_engine_id = seq.migrate_engine_id()
                    if source_engine_id:
                        logger.info(
                            f"Early free: seq {seq.seq_id} migrated from {source_engine_id}"
                        )
                        service._send_p2p_free_if_migrated(seq)

            for encoder_id, slot_indices in vision_free_by_encoder.items():
                try:
                    engine.send_free_vision_slots(encoder_id, slot_indices)
                except Exception as e:
                    logger.error(
                        f"Failed to send vision slot free to {encoder_id}: {e}"
                    )

            # Update tracking for next step
            service._previous_running_seqs = current_running_seqs

            _lp_emit_ms += (time.perf_counter() - _t_emit) * 1000
            _lp_steps += 1
            now = time.time()
            if now - _lp_last_log >= 5.0 and _lp_steps > 0:
                logger.info(
                    f"[backend_loop] drain={_lp_drain_ms / _lp_steps:.2f} "
                    f"emit={_lp_emit_ms / _lp_steps:.2f} ms/step (n={_lp_steps})"
                )
                _lp_drain_ms = 0.0
                _lp_emit_ms = 0.0
                _lp_steps = 0
                _lp_last_log = now

        except Exception as e:
            logger.error(f"Engine Backend Loop Error: {e}")
            traceback.print_exc()
            # Drop any in-flight step: its executor handle can no longer be
            # safely waited on after an arbitrary failure.
            pending = None
            time.sleep(1)


class EngineServer:
    def __init__(self, config: Config):
        self.config = config
        import multiprocessing

        self.requests_queue = multiprocessing.Queue()
        self.results_queue = multiprocessing.Queue()
        self.backend_process = None

    async def serve(self, bind_endpoint: Optional[str] = None):
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.DEALER)
        # Default binds a TCP port (disaggregated stack, Rust DLRouter client).
        # ``dlengine serve`` passes an ipc:// endpoint so the co-located OpenAI
        # HTTP server can connect without a port conflict.
        listen_addr = bind_endpoint or f"tcp://*:{self.config.port}"
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
        logger.info(f"NanoCtrl:        {self.config.ctrl_address or 'Not configured'}")
        logger.info(f"Ray Address:     {self.config.ray_address}")
        logger.info(
            f"Redis Scope:      {self.config.ctrl_scope or 'Not set (using NanoCtrl default)'}"
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
                    if action == _ACTION_STEPOUT_BATCH:
                        # One step's stepouts, batched by the backend to keep
                        # its loop fast. Fan out as regular per-seq StepOut
                        # packets so the zmq wire protocol is unchanged.
                        for seq_id, token_id, status in payload:
                            data = encode_packet(
                                0, build_stepout_payload(seq_id, token_id, status)
                            )
                            await socket.send(data)
                        continue
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

        async def backend_watchdog():
            """Exit the engine server if its backend process dies.

            The backend (run_engine_backend) does the heavy CUDA/Ray init and
            can fail at startup (e.g. no free GPUs / Ray placement group
            unavailable). Without this, the recv/results/p2p loops keep running
            on an empty shell, the GET_INFO request is never answered, and the
            HTTP server's readiness handshake hangs forever. Surfacing the death
            lets ``run_engine_server`` unwind, reap the backend, and close the
            zmq socket so the parent fails fast instead of wedging.
            """
            while True:
                bp = self.backend_process
                if bp is not None and not bp.is_alive():
                    logger.error(
                        f"Engine backend process exited (exitcode={bp.exitcode}); "
                        "shutting down engine server"
                    )
                    return
                await asyncio.sleep(0.5)

        tasks = [
            asyncio.create_task(coro)
            for coro in (
                recv_loop(),
                results_loop(),
                p2p_recv_loop(),
                backend_watchdog(),
            )
        ]
        try:
            done, still_pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        # Propagate any real failure (not a clean watchdog-triggered exit).
        for t in done:
            exc = t.exception()
            if exc is not None:
                raise exc


def run_engine_server(config: Config, bind_endpoint: Optional[str] = None):
    """Top-level entry point for spawning EngineServer in a child process.

    Used by ``dlengine serve``: the OpenAI HTTP server starts this
    via ``multiprocessing.Process`` so the engine runs in its own process and
    exposes a zmq DEALER over ``bind_endpoint`` (an ipc:// socket). Must be a
    module-level function so it is picklable across the process boundary.
    """
    try:
        import uvloop

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:  # noqa: BLE001
        pass

    server = EngineServer(config)

    # The HTTP parent terminates us with SIGTERM on shutdown. Convert it to a
    # KeyboardInterrupt so asyncio unwinds and the finally below tears down the
    # daemon backend process (and, with it, the Ray ModelRunner actors) instead
    # of leaking them. Default SIGTERM would kill us without any cleanup.
    import signal as _signal

    def _on_term(_signum, _frame):
        raise KeyboardInterrupt

    try:
        _signal.signal(_signal.SIGTERM, _on_term)
    except Exception:  # noqa: BLE001
        pass

    exit_code = 0
    try:
        # serve() returns (rather than blocking forever) if the backend process
        # dies — see backend_watchdog. Treat that as a failure exit so the HTTP
        # parent's liveness check sees a dead engine and aborts startup.
        asyncio.run(server.serve(bind_endpoint=bind_endpoint))
        logger.error("Engine server stopped (backend process is no longer alive)")
        exit_code = 1
    except KeyboardInterrupt:
        logger.info("Engine server process shutting down...")
    finally:
        backend = getattr(server, "backend_process", None)
        if backend is not None:
            try:
                backend.terminate()
                backend.join(timeout=5)
                if backend.is_alive():
                    backend.kill()
                    backend.join(timeout=3)
            except Exception:  # noqa: BLE001
                pass
    # Hard-exit so lingering non-daemon executor threads (e.g. a results_loop
    # blocked in Queue.get) can't keep this process alive — otherwise the parent
    # would never observe the engine as dead.
    os._exit(exit_code)


def main():
    logger.info("=" * 80)
    logger.info("DLEngine Engine Server")
    logger.info("=" * 80)

    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(description="DLEngine Engine Server")
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
