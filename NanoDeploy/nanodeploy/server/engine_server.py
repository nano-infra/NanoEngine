import argparse
import asyncio
import ctypes
import logging
import os
import struct
import sys
import traceback
from typing import Dict, List, Optional

# Ensure we can import the generated protobuf files
sys.path.append(os.getcwd())

import grpc
import engine_rpc_pb2
import engine_rpc_pb2_grpc

import flatbuffers
from nanodeploy._cpp import deserialize as deserialize_cpp
from nanodeploy._cpp import serialize
from nanodeploy.config import Config
from nanodeploy.server.llm_component import LLMComponent
from nanodeploy.logging import get_logger

# FlatBuffers imports
from nanodeploy.fbs.SequenceStatus import SequenceStatus
from nanodeploy.fbs.StepOut import (
    StepOutEnd,
    StepOutStart,
    StepOutAddSeqId,
    StepOutAddTokenId,
    StepOutAddStatus,
)
from nanodeploy.fbs.EngineInfo import EngineInfo
from nanodeploy.fbs.P2PInit import P2PInit as FbsP2PInit, P2PInitT
from nanodeploy.fbs.P2PInitResponse import P2PInitResponseT, P2PInitResponse
from nanodeploy.fbs.Peer import Peer, PeerT
from nanodeploy.fbs.P2PConnect import P2PConnect, P2PConnectT

logger = get_logger()

# Protocol Constants
MAGIC = 0x504F4B45
RESP_META_SIZE = 12

class EngineService(engine_rpc_pb2_grpc.EngineServiceServicer):
    def __init__(self, engine_component: LLMComponent):
        self.engine = engine_component
        # We assume a single primary controller connection for simplicity in this migration.
        # If multiple connect, they will all receive stream events if we broadcast, 
        # or we just support one active controller.
        # For now, let's use a broadcast queue or just single active connection logic.
        # To match legacy behavior (one connection), we'll store the latest active queue.
        self.active_queue: Optional[asyncio.Queue] = None

    async def Interact(self, request_iterator, context):
        peer = context.peer()
        logger.info(f"New gRPC Connection from {peer}")
        
        response_queue = asyncio.Queue()
        self.active_queue = response_queue

        reader_task = asyncio.create_task(self._reader_loop(request_iterator))
        
        try:
            while True:
                # Writer Loop
                packet = await response_queue.get()
                yield packet
        except asyncio.CancelledError:
            logger.info("Interact cancelled")
        except grpc.RpcError as e:
            logger.warning(f"gRPC Error: {e}")
        finally:
            logger.info(f"Connection from {peer} closed")
            reader_task.cancel()
            if self.active_queue == response_queue:
                self.active_queue = None

    async def _reader_loop(self, request_iterator):
        try:
            async for packet in request_iterator:
                await self._handle_packet(packet)
        except grpc.RpcError:
            pass
        except Exception as e:
            logger.error(f"Error in reader loop: {e}")
            traceback.print_exc()

    async def _handle_packet(self, packet: engine_rpc_pb2.StreamPacket):
        seq_id = packet.seq_id
        action = packet.action
        payload = packet.payload
        
        try:
            if action == 1:  # Add Request
                self._handle_add_request(payload)
            elif action == 2:  # Get Engine Info
                self._handle_get_info(seq_id)
            elif action == 3:  # P2P Init
                await self._handle_p2p_init(payload, seq_id)
            elif action == 4:  # P2P Connect
                await self._handle_p2p_connect(payload, seq_id)
            else:
                logger.warning(f"Unknown Action: {action}")
        except Exception as e:
            logger.error(f"Error handling packet action {action}: {e}")
            traceback.print_exc()

    # --- Handlers ---

    def _handle_add_request(self, payload: bytes):
        # Convert bytes to ctypes buffer for C++ interop
        c_buffer = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buffer)
        length = len(payload)

        sequences = deserialize_cpp(ptr, length)
        if not sequences:
            return
            
        logger.debug(f"Adding {len(sequences)} sequences.")
        self.engine.add_request(sequences)

    def _handle_get_info(self, req_seq_id: int):
        resp_payload = self.engine.get_engine_info()
        self._send_response(action=2, payload=resp_payload, seq_id=req_seq_id)

    async def _handle_p2p_init(self, payload: bytes, req_seq_id: int):
        try:
            p2p_init_view = FbsP2PInit.GetRootAs(payload, 0)
            p2p_init_obj = P2PInitT.InitFromObj(p2p_init_view)
        except Exception as e:
            logger.error(f"Failed to parse P2PInit: {e}")
            return

        init_args_list = []
        if p2p_init_obj.nodes:
            for node_t in p2p_init_obj.nodes:
                remote_id = node_t.id.decode("utf-8") if isinstance(node_t.id, bytes) else node_t.id
                remote_role = node_t.role.decode("utf-8") if isinstance(node_t.role, bytes) else node_t.role
                
                if remote_id == self.engine.config.engine_id:
                    continue

                node_builder = flatbuffers.Builder(1024)
                node_off = node_t.Pack(node_builder)
                node_builder.Finish(node_off)
                node_bytes = bytes(node_builder.Output())
                
                init_args_list.append((remote_id, remote_role, node_bytes))

        # Run parallel P2P init
        loop = asyncio.get_running_loop()
        results = []
        
        # We use run_in_executor to avoid blocking the async loop with network calls inside engine
        def run_p2p(node_bytes):
             return self.engine.p2p_init(node_bytes)

        futures = []
        for _, _, node_bytes in init_args_list:
            futures.append(loop.run_in_executor(None, run_p2p, node_bytes))
        
        if futures:
            done_results = await asyncio.gather(*futures, return_exceptions=True)
            for res in done_results:
                if isinstance(res, Exception):
                    logger.error(f"P2P Init Error: {res}")
                else:
                    results.append(res)
        
        # Build Response
        response_t = P2PInitResponseT()
        response_t.responses = []
        for peer_bytes in results:
            try:
                peer_view = Peer.GetRootAs(peer_bytes, 0)
                peer_t = PeerT.InitFromObj(peer_view)
                response_t.responses.append(peer_t)
            except Exception as e:
                logger.error(f"Failed to unpack Peer bytes: {e}")

        builder = flatbuffers.Builder(4096)
        resp_off = response_t.Pack(builder)
        builder.Finish(resp_off)
        resp_payload = bytes(builder.Output())

        self._send_response(action=3, payload=resp_payload, seq_id=req_seq_id)

    async def _handle_p2p_connect(self, payload: bytes, req_seq_id: int):
        try:
            p2p_connect_view = P2PConnect.GetRootAs(payload, 0)
            p2p_connect_obj = P2PConnectT.InitFromObj(p2p_connect_view)
        except Exception as e:
            logger.error(f"Failed to parse P2PConnect: {e}")
            return

        connect_args = []
        if p2p_connect_obj.peers:
            for peer_t in p2p_connect_obj.peers:
                target_id = peer_t.remoteId.decode("utf-8") if isinstance(peer_t.remoteId, bytes) else peer_t.remoteId
                builder = flatbuffers.Builder(1024)
                off = peer_t.Pack(builder)
                builder.Finish(off)
                peer_bytes = bytes(builder.Output())
                connect_args.append(peer_bytes)

        loop = asyncio.get_running_loop()
        def run_connect(peer_bytes):
            self.engine.p2p_connect(peer_bytes)

        futures = [loop.run_in_executor(None, run_connect, pb) for pb in connect_args]
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)
        
        # Response (Success) - Empty payload
        self._send_response(action=4, payload=b"", seq_id=req_seq_id)

    def _send_response(self, action: int, payload: bytes, seq_id: int = 0):
        if self.active_queue:
            packet = engine_rpc_pb2.StreamPacket(
                seq_id=seq_id,
                action=action,
                payload=payload
            )
            self.active_queue.put_nowait(packet)

    # --- Engine Loop Integration ---
    async def engine_loop(self):
        logger.info("Engine Loop Started")
        while True:
            try:
                # 1. Yield to allow network IO
                await asyncio.sleep(0.001)

                # 2. Check scheduler
                if self.engine.scheduler.is_finished():
                    continue

                # 3. Step (Blocking call, assume fast enough or TODO move to executor)
                # Moving to executor might be complex if it modifies shared state not thread-safe.
                # Assuming existing code was single threaded, calling it here is safe but blocks async loop.
                # We'll call it directly for now.
                dp_seqs, outputs, num_tokens, total_running, sch_lat, post_lat = self.engine.step()

                # 4. Process Outputs
                for seqs in dp_seqs:
                    for seq in seqs:
                        if seq.seq_id < 8: continue # data parallel dummy

                        if seq.is_finished:
                            self._send_stepout(seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED)
                        elif seq.is_to_be_migrated:
                            self._send_migration(seq)
                        elif len(seq.token_ids) > 0:
                            start_node = max(0, len(seq.token_ids) - num_tokens)
                            self._send_stepout(seq.seq_id, seq.token_ids[-1], SequenceStatus.RUNNING_DECODE)

            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                await asyncio.sleep(1)

    def _send_stepout(self, seq_id, token_id, status):
        builder = flatbuffers.Builder(128)
        StepOutStart(builder)
        StepOutAddSeqId(builder, seq_id)
        StepOutAddTokenId(builder, token_id)
        StepOutAddStatus(builder, status)
        step_out = StepOutEnd(builder)
        builder.Finish(step_out)
        payload = builder.Output()

        # Action ?? Legacy code didn't specify Action for StepOut in the struct header for *responses*?
        # Actually it did. The legacy code for StepOut used `header + meta + payload`.
        # meta was dummy bytes `b"\x00" * RESP_META_SIZE`.
        # Reader in Rust: `if let Ok(step_out) = flatbuffers::root::<StepOut>(&body)`
        # It checked `meta.action == 1` (Migration) or `2/3`.
        # StepOut seemed to be the "Default" fallthrough if it's not 1, 2, or 3.
        # Implies Action 0?
        # In `_send_stepout`: `meta = b"\x00" * RESP_META_SIZE`. Packed as `<III` (seq=0, status=0, action=0).
        # So Action 0 is StepOut.
        self._send_response(action=0, payload=payload, seq_id=seq_id)

    def _send_migration(self, seq):
        buffer_size = 4096 * 16
        buffer = ctypes.create_string_buffer(buffer_size)
        ptr = ctypes.addressof(buffer)

        try:
            payload_size = serialize(ptr, buffer_size, [seq], False)
            payload = buffer.raw[:payload_size]
            self._send_response(action=1, payload=payload, seq_id=seq.seq_id)
        except Exception as e:
            logger.error(f"Migration Serialize Error: {e}")


class EngineServer:
    def __init__(self, config: Config):
        self.config = config
        self.engine = LLMComponent(config)
        self.service = EngineService(self.engine)

    async def serve(self):
        # Configure KeepAlive to prevent disconnections
        options = [
            ('grpc.keepalive_time_ms', 10000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_recv_ping_interval_without_data_ms', 5000),
            ('grpc.http2.max_ping_strikes', 0),
            ('grpc.max_connection_age_ms', 100000),
        ]
        server = grpc.aio.server(options=options)
        engine_rpc_pb2_grpc.add_EngineServiceServicer_to_server(self.service, server)
        listen_addr = f'[::]:{self.config.port}'
        server.add_insecure_port(listen_addr)
        
        logger.info(f"Starting gRPC Engine Server on {listen_addr}...")
        await server.start()
        
        # Run engine loop and server wait
        await asyncio.gather(
            server.wait_for_termination(),
            self.service.engine_loop()
        )

def main():
    logger.info("PYTHON gRPC SERVER STARTING")
    
    # Simple Config Init (simplified from original for brevity, keeping core logic)
    from jsonargparse import ActionConfigFile, ArgumentParser
    parser = ArgumentParser(description="NanoDeploy Engine Server")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)
    args = parser.parse_args()
    init_args = {k: v for k, v in vars(args).items() if k != "config"}
    
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
