import argparse
import time
from collections import defaultdict

from nanodeploy import LLM, SamplingParams
from nanodeploy.engine.hierarchical_contract import (
    FinishEvent,
    validate_execution_trace_set,
)
from nanodeploy.engine.sequence import Sequence


def make_sequence(prompt_len: int, max_tokens: int, offset: int) -> Sequence:
    prompt = [
        (offset + token_idx) % 10_000
        for token_idx in range(prompt_len)
    ]
    return Sequence(
        prompt,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=max_tokens,
            ignore_eos=True,
        ),
    )


def wait_for_terminals(
    decode: LLM,
    request_ids: set[int],
    timeout_s: float,
) -> dict[int, FinishEvent]:
    deadline = time.monotonic() + timeout_s
    terminal: dict[int, FinishEvent] = {}
    while set(terminal) != request_ids:
        if time.monotonic() >= deadline:
            missing = request_ids.difference(terminal)
            raise TimeoutError(f"timed out waiting for requests {sorted(missing)}")
        for event in decode.poll():
            if event.request_id in request_ids:
                terminal[event.request_id] = event
        time.sleep(0.001)
    return terminal


def validate_trace_stage(
    decode: LLM,
    label: str,
    *,
    expected_ranks: int = 8,
) -> tuple[dict, ...]:
    traces = decode.drain_execution_traces()
    steps = validate_execution_trace_set(traces, range(expected_ranks))
    print(
        f"{label}: trace ranks={expected_ranks} "
        f"global_steps={len(steps)} records={len(traces)}"
    )
    return traces


def batch_kinds_by_step(
    traces: tuple[dict, ...],
) -> dict[tuple[int, int], dict[int, str]]:
    by_step: dict[tuple[int, int], dict[int, str]] = defaultdict(dict)
    for trace in traces:
        step = (trace["wave_id"], trace["quantum_id"])
        by_step[step][trace["global_rank"]] = trace["batch_kind"]
    return dict(by_step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the DP2SP4 hierarchical request lifecycle."
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/nvme1n1/ml_research/linbinbin1/DeepSeek-V3",
    )
    parser.add_argument(
        "--ray-address",
        default="10.102.243.60:8776",
    )
    parser.add_argument(
        "--master-address",
        default="10.102.243.60:26444",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    decode = LLM(
        args.model_path,
        enforce_eager=True,
        attention_dp=2,
        attention_sp=4,
        attention_tp=1,
        ffn_dp=1,
        ffn_ep=8,
        ffn_tp=1,
        mode="decode",
        master_address=args.master_address,
        ray_address=args.ray_address,
        dummy_prefill=True,
        dummy_weight=True,
        perfect_eplb=True,
        max_num_seqs=8,
        max_model_len=1024,
        max_num_batched_tokens=1024,
        loop_count=1,
        max_num_send_seqs=128,
        max_num_recv_seqs=130,
        kvcache_block_size=64,
        gpu_memory_utilization=args.gpu_memory_utilization,
        scheduler_arch="hierarchical",
        hierarchical_execution_trace=True,
    )
    try:
        # One real DP engine forces the other engine to execute only control
        # dummies while preserving deployment-wide EP collective order.
        single = make_sequence(prompt_len=64, max_tokens=5, offset=0)
        single_add = decode.add_request(single)
        if not single_add.accepted or single_add.engine_id != 0:
            raise RuntimeError(f"unexpected single-engine ADD result: {single_add}")
        single_terminal = wait_for_terminals(
            decode, {single.seq_id}, args.timeout
        )[single.seq_id]
        if (
            single_terminal.status != "FINISHED"
            or single_terminal.generated_count != 5
        ):
            raise RuntimeError(
                f"final-overrun commit mismatch: {single_terminal}"
            )
        single_traces = validate_trace_stage(decode, "single-engine")
        single_kinds = batch_kinds_by_step(single_traces)
        if not single_kinds or any(
            any(kinds[rank] != "real_or_mixed" for rank in range(4))
            or any(
                kinds[rank] != "all_control_dummy"
                for rank in range(4, 8)
            )
            for kinds in single_kinds.values()
        ):
            raise RuntimeError(
                "single-engine trace did not isolate real work to DP0"
            )

        # Global idle must pause all workers. Polling and load reporting may
        # continue, but the decode quantum counter and trace must not advance.
        before_idle = decode.hierarchical_metrics()["decode_quantum_count"]
        time.sleep(0.5)
        decode.poll()
        after_idle = decode.hierarchical_metrics()["decode_quantum_count"]
        if after_idle != before_idle or decode.drain_execution_traces():
            raise RuntimeError("global idle advanced decode execution")
        print("idle-pause: no forward progress while globally idle")

        # The next request starts a fresh wave. Two consecutive ADDs land on
        # different round-robin owners and must overlap in at least one step.
        pair = (
            make_sequence(prompt_len=64, max_tokens=32, offset=1_000),
            make_sequence(prompt_len=64, max_tokens=32, offset=2_000),
        )
        pair_add = decode.add_request(list(pair))
        if (
            not all(result.accepted for result in pair_add)
            or {result.engine_id for result in pair_add} != {0, 1}
        ):
            raise RuntimeError(f"unexpected dual-engine ADD results: {pair_add}")
        pair_terminal = wait_for_terminals(
            decode,
            {sequence.seq_id for sequence in pair},
            args.timeout,
        )
        if any(
            event.status != "FINISHED" or event.generated_count != 32
            for event in pair_terminal.values()
        ):
            raise RuntimeError(
                f"dual-engine terminal mismatch: {pair_terminal}"
            )
        pair_traces = validate_trace_stage(decode, "dual-engine")
        if not any(
            all(kind == "real_or_mixed" for kind in kinds.values())
            for kinds in batch_kinds_by_step(pair_traces).values()
        ):
            raise RuntimeError(
                "dual-engine requests never overlapped in a global step"
            )

        # Wait until one quantum has completed, then abort during a later
        # in-flight quantum. The in-flight result must be discarded at the
        # boundary and a single ABORTED terminal emitted.
        aborted = make_sequence(
            prompt_len=64,
            max_tokens=128,
            offset=3_000,
        )
        abort_add = decode.add_request(aborted)
        if not abort_add.accepted:
            raise RuntimeError(f"abort test ADD failed: {abort_add}")
        observed_traces: list[dict] = []
        deadline = time.monotonic() + args.timeout
        while not observed_traces:
            if time.monotonic() >= deadline:
                raise TimeoutError("abort test did not execute a first quantum")
            observed_traces.extend(decode.drain_execution_traces())
            time.sleep(0.001)
        time.sleep(0.2)
        abort_result = decode.abort_request(aborted.seq_id)
        if abort_result.status != "abort_pending":
            raise RuntimeError(
                f"abort was not observed in flight: {abort_result}"
            )
        abort_terminal = wait_for_terminals(
            decode, {aborted.seq_id}, args.timeout
        )[aborted.seq_id]
        if abort_terminal.status != "ABORTED":
            raise RuntimeError(f"abort terminal mismatch: {abort_terminal}")
        observed_traces.extend(decode.drain_execution_traces())
        validate_execution_trace_set(observed_traces, range(8))
        if not 0 < abort_terminal.generated_count < aborted.max_tokens:
            raise RuntimeError(
                f"abort committed an invalid prefix: {abort_terminal}"
            )
        print(
            "in-flight-abort:",
            f"generated_prefix={abort_terminal.generated_count}",
            f"trace_records={len(observed_traces)}",
        )

        metrics = decode.hierarchical_metrics()
        if metrics["waiting_requests"] or metrics["running_requests"]:
            raise RuntimeError(f"lifecycle left active requests: {metrics}")
        print("DP2SP4 hierarchical lifecycle validation passed")
    finally:
        decode.exit()


if __name__ == "__main__":
    main()
