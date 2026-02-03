import asyncio
import ctypes
import struct
import traceback
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
    StepOutEnd,
    StepOutStart,
)
from nanodeploy.logging import get_logger
from nanodeploy.server.llm_component import LLMComponent
from nanodeploy.server.zmq_protocol import decode_packet, encode_packet

logger = get_logger()


class EngineService:
    """ZMQ-based engine service. Handles AddRequest, GetEngineInfo, StepOut, Migration."""

    def __init__(self, engine_component: LLMComponent):
        self.engine = engine_component
        self._send_queue: Optional[asyncio.Queue] = None

    def _send_response(self, action: int, payload: bytes, seq_id: int = 0):
        if self._send_queue:
            data = encode_packet(seq_id, action, payload)
            self._send_queue.put_nowait(data)

    def _handle_add_request(self, payload: bytes):
        c_buffer = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buffer)
        length = len(payload)

        sequences = deserialize_cpp(ptr, length)
        if not sequences:
            return

        logger.debug(f"Adding {len(sequences)} sequences.")
        self.engine.add_request(sequences)

    def _handle_get_info(self, req_seq_id: int):
        resp_payload = self.engine.get_engine_info().encode("utf-8")
        self._send_response(action=2, payload=resp_payload, seq_id=req_seq_id)

    def _handle_packet(self, seq_id: int, action: int, payload: bytes):
        try:
            if action == 1:  # Add Request
                self._handle_add_request(payload)
            elif action == 2:  # Get Engine Info
                self._handle_get_info(seq_id)
            else:
                logger.warning(f"Unknown Action: {action}")
        except Exception as e:
            logger.error(f"Error handling packet action {action}: {e}")
            traceback.print_exc()

    def _send_stepout(self, seq_id, token_id, status):
        builder = flatbuffers.Builder(128)
        StepOutStart(builder)
        StepOutAddSeqId(builder, seq_id)
        StepOutAddTokenId(builder, token_id)
        StepOutAddStatus(builder, status)
        step_out = StepOutEnd(builder)
        builder.Finish(step_out)
        payload = builder.Output()
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

    async def engine_loop(self):
        logger.info("Engine Loop Started")
        while True:
            try:
                await asyncio.sleep(0.001)

                if self.engine.scheduler.is_finished():
                    continue

                dp_seqs, outputs, num_tokens, total_running, sch_lat, post_lat = (
                    self.engine.step()
                )

                for seqs in dp_seqs:
                    for seq in seqs:
                        if seq.seq_id < 8:
                            continue

                        if seq.is_finished:
                            self._send_stepout(
                                seq.seq_id, seq.token_ids[-1], SequenceStatus.FINISHED
                            )
                        elif seq.is_to_be_migrated:
                            self._send_migration(seq)
                        elif len(seq.token_ids) > 0:
                            start_node = max(0, len(seq.token_ids) - num_tokens)
                            self._send_stepout(
                                seq.seq_id,
                                seq.token_ids[-1],
                                SequenceStatus.RUNNING_DECODE,
                            )

            except Exception as e:
                logger.error(f"Engine Loop Error: {e}")
                await asyncio.sleep(1)


class EngineServer:
    def __init__(self, config: Config):
        self.config = config
        self.engine = LLMComponent(config)
        self.service = EngineService(self.engine)

    async def serve(self):
        ctx = zmq.asyncio.Context()
        socket = ctx.socket(zmq.DEALER)
        listen_addr = f"tcp://*:{self.config.port}"
        socket.bind(listen_addr)

        logger.info(f"Starting ZMQ Engine Server on {listen_addr}...")

        send_queue: asyncio.Queue = asyncio.Queue()
        self.service._send_queue = send_queue

        async def recv_loop():
            while True:
                try:
                    data = await socket.recv()
                    seq_id, action, payload = decode_packet(bytes(data))
                    self.service._handle_packet(seq_id, action, payload)
                except zmq.ZMQError as e:
                    if e.errno != zmq.ETERM:
                        logger.error(f"ZMQ recv error: {e}")
                    break
                except Exception as e:
                    logger.error(f"Recv loop error: {e}")
                    traceback.print_exc()

        async def send_loop():
            while True:
                try:
                    data = await send_queue.get()
                    await socket.send(data)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Send loop error: {e}")
                    break

        await asyncio.gather(
            recv_loop(),
            send_loop(),
            self.service.engine_loop(),
        )


def main():
    logger.info("PYTHON ZMQ ENGINE SERVER STARTING")

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
