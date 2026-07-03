import ctypes
import struct
import time as _time

from dlslime.rpc import method

from dlengine._rust.proto import RunnerOut

# Each run_batch reply is prefixed with an 8-byte little-endian uint64 holding
# the server-side handler duration in nanoseconds (decode + forward). The
# client subtracts this from the measured round trip to derive a "pure" network
# latency (wire + queueing), excluding remote GPU compute.
_REPLY_HEADER = struct.Struct("<Q")
REPLY_HEADER_SIZE = _REPLY_HEADER.size


def pack_reply_header(server_handler_ns: int) -> bytes:
    return _REPLY_HEADER.pack(int(server_handler_ns))


def unpack_reply_header(data) -> int:
    """Read the server handler nanoseconds from the start of a reply buffer."""
    return _REPLY_HEADER.unpack_from(data, 0)[0]


def encode_run_request(data: bytes, is_prefill: bool) -> bytes:
    return bytes((1 if is_prefill else 0,)) + data


def decode_run_request(ptr: int, nbytes: int) -> tuple[bytes, bool]:
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    payload = bytes(buf)
    if not payload:
        raise ValueError("Empty run request payload")
    return payload[1:], bool(payload[0])


def encode_run_result(result, server_handler_ns: int = 0) -> bytes:
    """Encode the per-step worker result as Rust protocol bytes.

    Accepts either the legacy ``list[list[int]]`` (token_ids only) or the
    tuple ``(token_ids: list[list[int]], logprobs: list[list[float]] | None)``
    shipped when SamplingParams.return_completion_logprobs is on.

    Encoding runs in Rust to keep the
    per-seq loop out of the interpreter on the per-step hot path.
    """
    if isinstance(result, tuple):
        token_ids, logprobs = result
    else:
        token_ids, logprobs = result, None

    return bytes(RunnerOut(token_ids, logprobs, server_handler_ns).to_bytes())


def decode_run_result(data: bytes):
    """Decode a worker result.

    Returns ``(list[list[int]], list[list[float]] | None)``. The 8-byte
    server-timing header prepended by encode_run_result is skipped inside the
    Rust decoder, which keeps the per-seq/per-token decode loop out of Python
    on the per-step hot path.
    """
    return RunnerOut.from_bytes(data).result


class ModelRunnerRpcService:
    def __init__(self, runner=None):
        self._runner = runner

    @method(raw=True)
    def run_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        # Always measure the handler duration (decode + forward); it is shipped
        # back in the reply header so the client can isolate network latency.
        t0 = _time.perf_counter()
        data, is_prefill = decode_run_request(ptr, nbytes)
        t1 = _time.perf_counter()
        result = self._runner.run_from_bytes(data, is_prefill)
        t2 = _time.perf_counter()
        handler_ns = int((t2 - t0) * 1e9)
        encoded = encode_run_result(result, handler_ns)
        t3 = _time.perf_counter()
        # Gated by Config.dlslime_timing (threaded into RunnerConfig on the
        # worker, same as step_timing). INFO so it is visible without flooding
        # the logs with unrelated DEBUG output.
        if not is_prefill:
            from dlengine.worker.runner_config import get_runner_config

            if get_runner_config().dlslime_timing:
                from dlengine.logging import get_logger

                get_logger().info(
                    f"[dlslime worker] decode_req={(t1-t0)*1000:.2f}ms "
                    f"forward={(t2-t1)*1000:.2f}ms "
                    f"encode={(t3-t2)*1000:.2f}ms "
                    f"total={(t3-t0)*1000:.2f}ms "
                    f"resp_bytes={len(encoded)}"
                )
        return encoded

    @method(raw=True)
    def migrate_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        self._runner.migrate_from_bytes(bytes(buf))
        return b""
