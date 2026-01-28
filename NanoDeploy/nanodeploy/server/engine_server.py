import argparse
import ctypes
import json
import logging
import select
import socket
import struct
import threading
import time
import uuid
from typing import Dict, List, Optional, Set

import etcd3
import flatbuffers
from nanodeploy._cpp import deserialize as deserialize_cpp
from nanodeploy.config import Config

# Core NanoDeploy imports
from nanodeploy.engine.sequence import Sequence
from nanodeploy.fbs.EngineInfo import EngineInfo
from nanodeploy.fbs.P2PInit import P2PInit
from nanodeploy.fbs.Peer import Peer

# Import SequenceStatus class directly from the generated file to avoid import errors
# if __init__.py is empty or overwritten.
from nanodeploy.fbs.SequenceStatus import SequenceStatus

# Import free functions from StepOut module
from nanodeploy.fbs.StepOut import (
    StepOutAddSeqId,
    StepOutAddStatus,
    StepOutAddTokenId,
    StepOutEnd,
    StepOutStart,
)
from nanodeploy.logging import get_logger
from nanodeploy.server.llm_component import LLMComponent

logger = get_logger()

# Protocol Constants
MAGIC = 0x504F4B45
HEADER_FMT = "<III"  # Magic(4), MetaSize(4), DataSize(4) -> 12 bytes
HEADER_SIZE = 12
META_SIZE = 72
RESP_META_SIZE = 12


def get_buffer_ptr_len(b: bytes):
    """Returns (ptr, len) for a bytes object."""
    if not isinstance(b, (bytes, bytearray)):
        raise ValueError("Inputs must be bytes or bytearray")

    # Create a ctypes buffer from the bytes
    # We must ensure 'b' stays alive while C++ uses it.
    # Since deserialize_cpp is synchronous and copies if needed (FlatBuffers verifier),
    # passing the pointer of the current bytes object is safe for the duration of the call.

    # For bytes, we can use from_buffer_copy if we want a mutable buffer, but here we just need read access.
    # ctypes.c_char_p(b) creates a null-terminated string, which might be wrong for binary data with nulls.

    # Use array interface or ctypes casting
    ptr = ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p).value
    # Note: c_char_p(b) might create a TEMP copy if b is immutable bytes?
    # Python bytes are immutable.
    # Safer way:
    buffer = (ctypes.c_char * len(b)).from_buffer_copy(b)
    ptr = ctypes.addressof(buffer)
    return ptr, len(b)


class EngineServer:
    def __init__(self, config: Config):
        self.config = config
        self.host = config.host
        self.port = config.port

        # Initialize Engine
        logger.info(f"Initializing LLMEngine with config: {config}")
        self.engine = LLMComponent(config)

        # Connection state
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.server_socket.setblocking(False)

        self.conn: Optional[socket.socket] = None
        self.read_buffer = b""
        self.current_header = None  # (magic, meta_size, data_size)

    # ... (rest of class methods are unchanged, but I need to be careful not to delete them if I'm replacing the class init only)

    # I will use a targeted replace for __init__ first, then the main block.
    # Actually, I can just replace the definition of EngineServer.__init__ and the bottom block.

    def run(self):
        logger.info(f"Engine Server listening on {self.host}:{self.port}")

        try:
            while True:
                self.loop_step()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            if self.conn:
                self.conn.close()
            self.server_socket.close()

    def loop_step(self):
        """Single iteration of the event loop."""

        # 1. Network I/O
        self.handle_network()

        # 2. Engine Step
        self.engine_step()

    def handle_network(self):
        # Accept connection if none
        if self.conn is None:
            try:
                conn, addr = self.server_socket.accept()
                logger.info(f"Accepted connection from {addr}")
                # Use blocking mode to ensure sendall() completes without BlockingIOError.
                # Since we use select() for reading, this is acceptable for this simple server.
                conn.setblocking(True)
                self.conn = conn
            except BlockingIOError:
                pass
            return

        # Check for data
        try:
            # Non-blocking recv
            # We assume small socket buffer reads are fine, we aggregate in self.read_buffer

            # Use select to poll valid for reading
            rlist, _, _ = select.select([self.conn], [], [], 0)
            if rlist:
                chunk = self.conn.recv(4096)
                if not chunk:
                    logger.info("Connection closed by peer")
                    self.conn.close()
                    self.conn = None
                    self.read_buffer = b""
                    return
                self.read_buffer += chunk
                self.process_buffer()

        except (BlockingIOError, ConnectionResetError, BrokenPipeError) as e:
            if isinstance(e, (ConnectionResetError, BrokenPipeError)):
                logger.info(f"Connection lost: {e}")
                self.conn.close()
                self.conn = None
                self.read_buffer = b""

    def process_buffer(self):
        """Process accumulated data in read_buffer."""
        while True:
            # 1. Parse Header
            if self.current_header is None:
                if len(self.read_buffer) >= HEADER_SIZE:
                    header_data = self.read_buffer[:HEADER_SIZE]
                    self.read_buffer = self.read_buffer[HEADER_SIZE:]
                    self.current_header = struct.unpack(HEADER_FMT, header_data)

                    magic, meta_size, data_size = self.current_header
                    logger.debug(
                        f"Recv Header: Magic={hex(magic)}, Meta={meta_size}, Payload={data_size}",
                    )

                    if magic != MAGIC:
                        logger.info(f"Invalid Magic: {hex(magic)}. Closing connection.")
                        self.conn.close()
                        self.conn = None
                        return
                else:
                    # Not enough data for header
                    break

            if self.current_header is not None:
                magic, meta_size, data_size = self.current_header
                total_needed = meta_size + data_size

                if len(self.read_buffer) >= total_needed:
                    # Extract full message
                    meta_data = self.read_buffer[:meta_size]
                    payload_data = self.read_buffer[meta_size : meta_size + data_size]
                    self.read_buffer = self.read_buffer[meta_size + data_size :]

                    # Reset header for next message
                    self.current_header = None

                    # Handle Message
                    self.handle_message(meta_data, payload_data)
                else:
                    # Not enough data for body
                    break

    def handle_message(self, meta: bytes, payload: bytes):
        # Parse Meta (72 bytes)
        # NetMetaRaw: action(4), seq(4), actor_id(32), actor_type(32)
        try:
            action, seq_id, actor_id, actor_type = struct.unpack("<II32s32s", meta)
        except struct.error:
            logger.error("Failed to unpack metadata")
            return

        try:
            if action == 1:  # Add Request (Legacy/Standard)
                self.handle_add_request(payload)
            elif action == 2:  # GET_ENGINE_INFO
                # Return EngineInfo as FlatBuffers
                resp_payload = self.engine.get_engine_info()

                seq_id_u32 = 0
                status = 0
                meta = struct.pack("<III", seq_id_u32, status, action)
                header = struct.pack(
                    HEADER_FMT, MAGIC, RESP_META_SIZE, len(resp_payload)
                )

                try:
                    self.conn.sendall(header + meta + resp_payload)
                except (BlockingIOError, BrokenPipeError):
                    pass

            elif action == 3:  # P2P_INIT (Binary FlatBuffers)
                import flatbuffers
                from nanodeploy.fbs.nanodeploy.fbs.EngineInfo import (
                    EngineInfo,
                    EngineInfoT,
                )
                from nanodeploy.fbs.nanodeploy.fbs.P2PInit import (
                    P2PInit as FbsP2PInit,
                    P2PInitT,
                )
                from nanodeploy.fbs.nanodeploy.fbs.P2PInitResponse import (
                    P2PInitResponse,
                    P2PInitResponseT,
                )
                from nanodeploy.fbs.nanodeploy.fbs.Peer import Peer, PeerT

                try:
                    # Decode P2PInit using Object API
                    p2p_init_view = FbsP2PInit.GetRootAs(payload, 0)
                    p2p_init_obj = P2PInitT.InitFromObj(p2p_init_view)
                except Exception as e:
                    logger.error(f"Failed to parse P2PInit FlatBuffers: {e}")
                    return

                # Prepare args for parallel execution
                init_args_list = []
                if p2p_init_obj.nodes:
                    for node_t in p2p_init_obj.nodes:
                        remote_id = (
                            node_t.id.decode("utf-8")
                            if isinstance(node_t.id, bytes)
                            else node_t.id
                        )
                        remote_role = (
                            node_t.role.decode("utf-8")
                            if isinstance(node_t.role, bytes)
                            else node_t.role
                        )

                        if remote_id == self.config.engine_id:
                            continue

                        # Serialize NodeT to bytes to pass to p2p_init
                        # This satisfies "input args just input bytes"
                        node_builder = flatbuffers.Builder(1024)
                        node_off = node_t.Pack(node_builder)
                        node_builder.Finish(node_off)
                        node_bytes = bytes(node_builder.Output())

                        init_args_list.append((remote_id, remote_role, node_bytes))

                # Parallel Init
                import concurrent.futures

                results = []
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_remote = {
                        executor.submit(self.engine.p2p_init, node_bytes): (
                            remote_id,
                            remote_role,
                        )
                        for remote_id, remote_role, node_bytes in init_args_list
                    }

                    for future in concurrent.futures.as_completed(future_to_remote):
                        remote_id, remote_role = future_to_remote[future]
                        try:
                            # Returns bytes (Serialized Peer table)
                            peer_bytes = future.result()
                            logger.info(
                                f"P2P Init Success for {remote_id} ({remote_role})"
                            )
                            results.append(peer_bytes)
                        except Exception as e:
                            logger.error(f"P2P Init failed for {remote_id}: {e}")

                # Build Response from results
                # results is list[bytes] where each bytes is a serialized Peer table

                response_t = P2PInitResponseT()
                response_t.responses = []

                for peer_bytes in results:
                    try:
                        # 1. Deserialize View
                        peer_view = Peer.GetRootAs(peer_bytes, 0)
                        # 2. Unpack to T
                        peer_t = PeerT.InitFromObj(peer_view)
                        response_t.responses.append(peer_t)
                    except Exception as e:
                        logger.error(f"Failed to unpack Peer bytes: {e}")

                builder = flatbuffers.Builder(4096)
                resp_off = response_t.Pack(builder)
                builder.Finish(resp_off)
                resp_payload = bytes(builder.Output())
                # Send response
                seq_id_u32 = 0
                status = 0
                meta = struct.pack("<III", seq_id_u32, status, action)
                header = struct.pack(
                    HEADER_FMT, MAGIC, RESP_META_SIZE, len(resp_payload)
                )

                try:
                    self.conn.sendall(header + meta + resp_payload)
                except (BlockingIOError, BrokenPipeError):
                    logger.error("Failed to send P2PInit Response")
                    self.conn.close()
                    self.conn = None

            elif action == 4:  # P2P_CONNECT (Binary FlatBuffers)
                from nanodeploy.fbs.nanodeploy.fbs.P2PConnect import (
                    P2PConnect,
                    P2PConnectT,
                )

                try:
                    p2p_connect_view = P2PConnect.GetRootAs(payload, 0)
                    p2p_connect_obj = P2PConnectT.InitFromObj(p2p_connect_view)
                except Exception as e:
                    logger.error(f"Failed to parse P2PConnect FlatBuffers: {e}")
                    return

                # Prepare args for parallel execution
                connect_args = []
                if p2p_connect_obj.peers:
                    for peer_t in p2p_connect_obj.peers:
                        target_id = (
                            peer_t.remoteId.decode("utf-8")
                            if isinstance(peer_t.remoteId, bytes)
                            else peer_t.remoteId
                        )

                        # Serialize PeerT to bytes to pass to p2p_connect
                        builder = flatbuffers.Builder(1024)
                        off = peer_t.Pack(builder)
                        builder.Finish(off)
                        peer_bytes = bytes(builder.Output())

                        connect_args.append((target_id, peer_bytes))

                # Parallel Connect
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_target = {
                        executor.submit(self.engine.p2p_connect, peer_bytes): target_id
                        for target_id, peer_bytes in connect_args
                    }

                    for future in concurrent.futures.as_completed(future_to_target):
                        target_id = future_to_target[future]
                        try:
                            future.result()
                            logger.info(f"P2P Connect Success for {target_id}")
                        except Exception as e:
                            logger.error(f"Failed to connect P2P to {target_id}: {e}")

                # Send Response (Success)
                seq_id_u32 = 0
                status = 0
                meta = struct.pack("<III", seq_id_u32, status, action)
                resp_payload = b""
                header = struct.pack(
                    HEADER_FMT, MAGIC, RESP_META_SIZE, len(resp_payload)
                )
                try:
                    self.conn.sendall(header + meta + resp_payload)
                except (BlockingIOError, BrokenPipeError):
                    pass

                logger.info("P2P Connect sequence completed.")

            else:
                logger.error(f"Unknown Action ID: {action}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            import traceback

            traceback.print_exc()

    def handle_add_request(self, payload: bytes):
        logger.debug("TRACE: handle_add_request start")
        # Safer way to get pointer: create ctypes buffer copy/reference
        import ctypes

        # Ensure payload is immutable bytes, copy to a mutable buffer to be safe and aligned?
        # Actually, just getting address of bytes object data is tricky in pure python without C API.
        # Best way: create a ctypes string buffer from the bytes.
        # This involves a copy, but is safe.
        c_buffer = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buffer)
        length = len(payload)

        # deserialize_cpp returns std::vector<std::shared_ptr<Sequence>>
        # bound to Python as List[nanodeploy._cpp.Sequence]
        logger.debug("TRACE: calling deserialize_cpp")
        sequences = deserialize_cpp(ptr, length)

        # DUMMY SEQUENCE GENERATION REMOVED

        if not sequences:
            logger.debug("Deserialized empty sequence list.")
            return

        logger.debug(f"Adding {len(sequences)} sequences to engine.")
        for s in sequences:
            logger.debug(
                f"  [Recv Action 1] Seq {s.seq_id}, Tokens: {len(s.token_ids)}, MaxTokens: {s.sampling_params.max_tokens}, Ids: {s.token_ids if len(s.token_ids) < 20 else str(s.token_ids[:10])+'...'}"
            )

        logger.debug("TRACE: calling engine.add_request")
        self.engine.add_request(sequences)
        logger.debug("TRACE: handle_add_request done")

    def engine_step(self):
        try:
            if not self.engine.scheduler.is_finished():
                dp_seqs, outputs, num_tokens, total_running, sch_lat, post_lat = (
                    self.engine.step()
                )

                for seqs in dp_seqs:
                    for seq in seqs:
                        if seq.seq_id < 8:
                            # dummy seq
                            continue

                        if seq.is_finished:
                            logger.info(
                                f"Seq {seq.seq_id} FINISHED. Reason: {seq.status}"
                            )
                            self.send_stepout(
                                seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED
                            )
                        elif seq.is_to_be_migrated:
                            self.send_migration(seq)
                        elif len(seq.token_ids) > 0:
                            status_enum = SequenceStatus.RUNNING_DECODE
                            start_node = max(0, len(seq.token_ids) - num_tokens)
                            logger.debug(
                                f"Seq {seq.seq_id} RUNNING. Len: {len(seq.token_ids)} Max: {seq.sampling_params.max_tokens}"
                            )
                            self.send_stepout(
                                seq.seq_id, seq.token_ids[-1], status_enum
                            )

        except Exception as e:
            logger.error(f"Error during engine step: {e}")
            import traceback

            traceback.print_exc()

    def send_migration(self, seq):
        if self.conn is None:
            return

        logger.debug(
            f"Migrating Seq {seq.seq_id}, Tokens: {len(seq.token_ids)}, Ids: {seq.token_ids if len(seq.token_ids) < 20 else str(seq.token_ids[:10])+'...'}"
        )

        import ctypes
        import sys

        # Use C++ serialization
        from nanodeploy._cpp import serialize

        buffer_size = 4096 * 16  # 64KB
        buffer = ctypes.create_string_buffer(buffer_size)
        ptr = ctypes.addressof(buffer)

        try:
            logger.debug(f"Serializing Seq {seq.seq_id}...")
            # serialize(data_ptr, buffer_size, seqs_list, is_prefill)
            payload_size = serialize(ptr, buffer_size, [seq], False)
            logger.debug(f"Serialized size: {payload_size}")
            payload = buffer.raw[:payload_size]
        except Exception as e:
            logger.error(f"Serialization failed: {e}")
            import traceback

            traceback.print_exc()
            return

        # Header + Meta
        action = 1
        seq_id_u32 = seq.seq_id & 0xFFFFFFFF
        status = 0

        meta = struct.pack("<III", seq_id_u32, status, action)
        header = struct.pack(HEADER_FMT, MAGIC, RESP_META_SIZE, payload_size)

        try:
            self.conn.sendall(header + meta + payload)
        except (BlockingIOError, BrokenPipeError):
            logger.error("Failed to send Migration")
            self.conn.close()
            self.conn = None

    def send_stepout(self, seq_id, token_id, status):
        if self.conn is None:
            return

        builder = flatbuffers.Builder(128)
        StepOutStart(builder)
        StepOutAddSeqId(builder, seq_id)
        StepOutAddTokenId(builder, token_id)
        StepOutAddStatus(builder, status)
        step_out = StepOutEnd(builder)
        builder.Finish(step_out)

        payload = builder.Output()

        # Header
        header = struct.pack(HEADER_FMT, MAGIC, RESP_META_SIZE, len(payload))
        meta = b"\x00" * RESP_META_SIZE  # Dummy meta

        try:
            self.conn.sendall(header + meta + payload)
        except (BlockingIOError, BrokenPipeError):
            logger.error("Failed to send StepOut")
            self.conn.close()
            self.conn = None


from jsonargparse import ActionConfigFile, ArgumentParser


def main():
    logger.info("PYTHON SERVER: Script started (main function entered)")
    parser = ArgumentParser(description="NanoDeploy Engine Server")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)

    args = parser.parse_args()

    # Filter out 'config' and any other internal keys argument if present
    init_args = {k: v for k, v in vars(args).items() if k != "config"}

    try:
        config = Config(**init_args)
    except Exception as e:
        logger.error(f"Error initializing Config: {e}")
        exit(1)

    server = EngineServer(config)
    server.run()


if __name__ == "__main__":
    main()
