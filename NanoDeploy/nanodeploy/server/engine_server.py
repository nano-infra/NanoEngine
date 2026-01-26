import argparse
import ctypes
import select
import socket
import struct
import time
import uuid
from typing import List, Optional

import flatbuffers
from nanodeploy._cpp import deserialize as deserialize_cpp
from nanodeploy.config import Config

# Core NanoDeploy imports
from nanodeploy.engine.sequence import Sequence

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
from nanodeploy.llm import LLM

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
        print(f"Initializing LLMEngine with config: {config}")
        self.engine = LLM(config)

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
        print(f"Engine Server listening on {self.host}:{self.port}")

        try:
            while True:
                self.loop_step()
        except KeyboardInterrupt:
            print("Shutting down...")
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
                print(f"Accepted connection from {addr}")
                conn.setblocking(False)
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
                    print("Connection closed by peer")
                    self.conn.close()
                    self.conn = None
                    self.read_buffer = b""
                    return
                self.read_buffer += chunk
                self.process_buffer()

        except (BlockingIOError, ConnectionResetError, BrokenPipeError) as e:
            if isinstance(e, (ConnectionResetError, BrokenPipeError)):
                print(f"Connection lost: {e}")
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
                    if magic != MAGIC:
                        print(f"Invalid Magic: {hex(magic)}. Closing connection.")
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
        # Deserialize SequenceList
        try:
            # Safer way to get pointer: create ctypes buffer copy/reference
            import ctypes

            # Ensure payload is immutable bytes, copy to a mutable buffer to be safe and aligned?
            # Actually, just getting address of bytes object data is tricky in pure python without C API.
            # Best way: create a ctypes string buffer from the bytes.
            # This involves a copy, but is safe.
            c_buffer = ctypes.create_string_buffer(payload, len(payload))
            ptr = ctypes.addressof(c_buffer)
            length = len(payload)

            # ptr, length = get_buffer_ptr_len(payload)
            # deserialize_cpp returns std::vector<std::shared_ptr<Sequence>>
            # bound to Python as List[nanodeploy._cpp.Sequence]
            sequences = deserialize_cpp(ptr, length)

            if not sequences:
                print("Deserialized empty sequence list.")
                return

            print(f"Adding {len(sequences)} sequences to engine.")
            self.engine.add_request(sequences)

        except Exception as e:
            print(f"Error handling message: {e}")
            import traceback

            traceback.print_exc()

    def engine_step(self):
        # Run one step of LLMEngine
        # LLMEngine.step() returns (outputs, num_tokens, ...)
        # outputs contains FINISHED or MIGRATED sequences.

        # We need to detect NEWLY generated tokens for ALL active sequences to stream StepOut.
        # However, LLMEngine.step currently only returns finished sequences.

        # We need to manually inspect active sequences or modify LLMEngine.
        # Since I cannot easily modify LLMEngine interface right now without breaking things,
        # let's look at how to get active sequences.

        # self.engine.scheduler.running contains list of sequences.
        # We can track their 'num_completed_tokens' (or check last added token).

        # For efficiency, we can assume that every active sequence in 'running' that is in RUNNING state
        # generated one token if step() was successful and is_prefill=False.

        # Let's call step()

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
                            self.send_stepout(
                                seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED
                            )
                        elif seq.is_to_be_migrated:
                            self.send_stepout(
                                seq.seq_id, 0, SequenceStatus.TO_BE_MIGRATED
                            )
                        elif len(seq.token_ids) > 0:
                            status_enum = SequenceStatus.RUNNING_DECODE
                            self.send_stepout(
                                seq.seq_id, seq.token_ids[-1], status_enum
                            )

        except Exception as e:
            print(f"Error during engine step: {e}")
            import traceback

            traceback.print_exc()

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
            print("Failed to send StepOut")
            self.conn.close()
            self.conn = None


from jsonargparse import ActionConfigFile, ArgumentParser


def main():
    parser = ArgumentParser(description="NanoDeploy Engine Server")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_class_arguments(Config, fail_untyped=False)

    args = parser.parse_args()

    # Filter out 'config' and any other internal keys argument if present
    init_args = {k: v for k, v in vars(args).items() if k != "config"}

    try:
        config = Config(**init_args)
    except Exception as e:
        print(f"Error initializing Config: {e}")
        exit(1)

    server = EngineServer(config)
    server.run()


if __name__ == "__main__":
    main()
