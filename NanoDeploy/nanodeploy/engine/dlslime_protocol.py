import ctypes
import time as _time

import flatbuffers
from dlslime.rpc import method

from nanodeploy.fbs.RunBatchOutput import (
    RunBatchOutput,
    RunBatchOutputAddHasLogprobs,
    RunBatchOutputAddSequences,
    RunBatchOutputEnd,
    RunBatchOutputStart,
    RunBatchOutputStartSequencesVector,
)
from nanodeploy.fbs.RunSequenceOutput import (
    RunSequenceOutputAddLogprobs,
    RunSequenceOutputAddTokenIds,
    RunSequenceOutputCreateLogprobsVector,
    RunSequenceOutputCreateTokenIdsVector,
    RunSequenceOutputEnd,
    RunSequenceOutputStart,
)


_DLSLIME_TIMING = "1"


def _create_run_result_seq(
    builder: flatbuffers.Builder,
    token_ids,
    logprobs=None,
) -> int:
    token_ids_vec = RunSequenceOutputCreateTokenIdsVector(builder, token_ids)
    logprobs_vec = (
        RunSequenceOutputCreateLogprobsVector(builder, logprobs)
        if logprobs is not None
        else 0
    )

    RunSequenceOutputStart(builder)
    if logprobs_vec:
        RunSequenceOutputAddLogprobs(builder, logprobs_vec)
    RunSequenceOutputAddTokenIds(builder, token_ids_vec)
    return RunSequenceOutputEnd(builder)


def encode_run_request(data: bytes, is_prefill: bool) -> bytes:
    return bytes((1 if is_prefill else 0,)) + data


def decode_run_request(ptr: int, nbytes: int) -> tuple[bytes, bool]:
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    payload = bytes(buf)
    if not payload:
        raise ValueError("Empty run request payload")
    return payload[1:], bool(payload[0])


def encode_run_result(result) -> bytes:
    """Encode the per-step worker result as raw FlatBuffers bytes.

    Accepts either the legacy ``list[list[int]]`` (token_ids only) or the
    tuple ``(token_ids: list[list[int]], logprobs: list[list[float]] | None)``
    shipped when SamplingParams.return_completion_logprobs is on.
    """
    if isinstance(result, tuple):
        token_ids, logprobs = result
    else:
        token_ids, logprobs = result, None

    has_logprobs = logprobs is not None
    builder = flatbuffers.Builder(256)
    seq_offsets = [
        _create_run_result_seq(
            builder,
            seq_token_ids,
            logprobs[i] if has_logprobs and i < len(logprobs) else None,
        )
        for i, seq_token_ids in enumerate(token_ids)
    ]

    RunBatchOutputStartSequencesVector(builder, len(seq_offsets))
    for seq_offset in reversed(seq_offsets):
        builder.PrependUOffsetTRelative(seq_offset)
    seqs_vec = builder.EndVector()

    RunBatchOutputStart(builder)
    RunBatchOutputAddHasLogprobs(builder, has_logprobs)
    RunBatchOutputAddSequences(builder, seqs_vec)
    root = RunBatchOutputEnd(builder)
    builder.Finish(root)
    return bytes(builder.Output())


def decode_run_result(data: bytes):
    """Decode a worker result.

    Returns either ``list[list[int]]`` (token_ids only) or
    ``(list[list[int]], list[list[float]])``.
    """
    output = RunBatchOutput.GetRootAs(data, 0)
    has_logprobs = output.HasLogprobs()
    token_ids = []
    logprobs = []
    for i in range(output.SequencesLength()):
        seq = output.Sequences(i)
        token_ids.append([seq.TokenIds(j) for j in range(seq.TokenIdsLength())])
        if has_logprobs:
            logprobs.append([seq.Logprobs(j) for j in range(seq.LogprobsLength())])

    return (token_ids, logprobs) if has_logprobs else token_ids


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

                get_logger().debug(
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
