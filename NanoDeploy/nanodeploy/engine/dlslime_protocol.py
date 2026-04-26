import ctypes
import os
import pickle
import time as _time

from dlslime.rpc import method


_DLSLIME_TIMING = "1"


def encode_run_request(data: bytes, is_prefill: bool) -> bytes:
    return bytes((1 if is_prefill else 0,)) + data


def decode_run_request(ptr: int, nbytes: int) -> tuple[bytes, bool]:
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    payload = bytes(buf)
    if not payload:
        raise ValueError("Empty run request payload")
    return payload[1:], bool(payload[0])


def encode_run_result(result: list[list[int]]) -> bytes:
    return pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)


def decode_run_result(data: bytes) -> list[list[int]]:
    return pickle.loads(data)


class ModelRunnerRpcService:
    def __init__(self, runner=None):
        self._runner = runner

    @method(raw=True)
    def run_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        if _DLSLIME_TIMING:
            t0 = _time.perf_counter()
            data, is_prefill = decode_run_request(ptr, nbytes)
            t1 = _time.perf_counter()
            result = self._runner.run_from_bytes(data, is_prefill)
            t2 = _time.perf_counter()
            encoded = encode_run_result(result)
            t3 = _time.perf_counter()
            if not is_prefill:
                from nanodeploy.logging import get_logger

                get_logger().info(
                    f"[dlslime worker] decode_req={(t1-t0)*1000:.2f}ms "
                    f"forward={(t2-t1)*1000:.2f}ms "
                    f"encode={(t3-t2)*1000:.2f}ms "
                    f"total={(t3-t0)*1000:.2f}ms "
                    f"resp_bytes={len(encoded)}"
                )
            return encoded
        data, is_prefill = decode_run_request(ptr, nbytes)
        result = self._runner.run_from_bytes(data, is_prefill)
        return encode_run_result(result)

    @method(raw=True)
    def migrate_batch(self, channel, ptr: int, nbytes: int) -> bytes:
        if self._runner is None:
            raise RuntimeError("ModelRunnerRpcService is not attached to a runner")
        buf = (ctypes.c_char * nbytes).from_address(ptr)
        self._runner.migrate_from_bytes(bytes(buf))
        return b""
