import ctypes
import itertools
import struct
import time as _time

from dlslime.rpc import method

from dlengine._rust.proto import RunnerIn, RunnerOut


def encode_run_request(data: bytes) -> bytes:
    return RunnerIn.from_bytes(data).to_bytes()


def decode_run_request(ptr: int, nbytes: int) -> tuple[bytes, bool]:
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    runner_in = RunnerIn.from_bytes(bytes(buf))
    return runner_in.to_bytes(), bool(runner_in.is_prefill)


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

    return RunnerOut(token_ids, logprobs, server_handler_ns).to_bytes()


def decode_run_result(data: bytes):
    """Decode a worker result.

    Returns ``(list[list[int]], list[list[float]] | None)``. The Rust decoder
    keeps the per-seq/per-token loop out of Python on the per-step hot path.
    """
    return RunnerOut.from_bytes(data).result


def decode_runner_out(data: bytes) -> RunnerOut:
    return RunnerOut.from_bytes(data)


def server_handler_ns(data: bytes) -> int:
    """Read the remote decode + forward duration from a RunnerOut payload."""
    return int(RunnerOut.from_bytes(data).server_handler_ns)


class ModelRunnerRpcService:
    def __init__(self, runner=None):
        self._runner = runner
        self._prepared = {}
        self._prepare_ids = itertools.count(1)

    @method(raw=True)
    def run_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        # Always measure the handler duration (decode + forward); it is shipped
        # back in RunnerOut so the client can isolate network latency.
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
    def prepare_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        data, is_prefill = decode_run_request(ptr, nbytes)
        handle = next(self._prepare_ids)
        self._prepared[handle] = self._runner.prepare_from_bytes(data, is_prefill)
        return struct.pack("<Q", handle)

    @method(raw=True)
    def run_prepared(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        if nbytes != 8:
            raise ValueError(f"run_prepared expects an 8-byte handle, got {nbytes}")
        t0 = _time.perf_counter()
        handle = struct.unpack("<Q", (ctypes.c_char * nbytes).from_address(ptr))[0]
        prepared = self._prepared.pop(handle)
        result = self._runner.run_prepared(prepared)
        handler_ns = int((_time.perf_counter() - t0) * 1e9)
        return encode_run_result(result, handler_ns)

    @method(raw=True)
    def migrate_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        self._runner.migrate_from_bytes(bytes(buf))
        return b""
