from time import perf_counter

from dlengine.logging import get_logger

logger = get_logger()


def generate(
    engine,
    use_tqdm: bool = True,
    log_metrics_interval: int = 10,
    return_serialized: bool = False,
) -> list[dict] | list[bytes]:
    """Drive an engine synchronously until all queued requests finish."""
    _ = log_metrics_interval
    num_reqs = engine.scheduler.num_waiting()
    pbar = None
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)
        except Exception:
            use_tqdm = False

    outputs_by_seq: dict[int, dict] = {}
    serialized_outputs: list[bytes] = []
    completed_seq_ids: set[int] = set()
    prefill_throughput = decode_throughput = 0.0
    step_count = 0

    window_start = perf_counter()
    window_tokens = 0
    window_interval = 5.0
    last_tqdm_update = perf_counter()
    tqdm_interval = 1.0

    while not engine.is_finished():
        t = perf_counter()
        result = engine.step()
        step_count += 1
        step_duration = perf_counter() - t

        if result.prefill_tokens > 0:
            prefill_throughput = result.prefill_tokens / step_duration
            engine.metrics_manager.server_metric.record_prefill_throughput(
                result.prefill_tokens, step_duration
            )
        if result.decode_tokens > 0:
            engine.metrics_manager.server_metric.record_decode_throughput(
                result.decode_tokens, step_duration
            )
            window_tokens += result.decode_tokens

        now = perf_counter()
        window_elapsed = now - window_start
        if window_elapsed >= window_interval and window_tokens > 0:
            decode_throughput = window_tokens / window_elapsed
            logger.info(
                f"[Throughput] {decode_throughput:.0f} tok/s "
                f"({window_tokens} tokens in {window_elapsed:.1f}s, "
                f"bs={result.real_bs}, step={step_count})"
            )
            window_start = now
            window_tokens = 0

        if use_tqdm and (now - last_tqdm_update >= tqdm_interval):
            last_tqdm_update = now
            pbar.set_postfix(
                {
                    "bs": f"{result.real_bs}",
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                    "step": f"{step_count}",
                }
            )
        for event in result.outputs:
            seq_id = int(event["seq_id"])
            if event["num_tokens"] > 0 and not event["is_to_be_migrated"]:
                output = outputs_by_seq.setdefault(
                    seq_id,
                    {
                        "seq_id": seq_id,
                        "token_ids": [],
                        "is_finished": False,
                    },
                )
                output["token_ids"].append(int(event["last_token"]))
            if event["is_to_be_migrated"] and event.get("migration_payload"):
                serialized_outputs.append(bytes(event["migration_payload"]))
            if event["is_finished"] or event["is_to_be_migrated"]:
                outputs_by_seq.setdefault(
                    seq_id,
                    {
                        "seq_id": seq_id,
                        "token_ids": [],
                        "is_finished": bool(event["is_finished"]),
                    },
                )["is_finished"] = bool(event["is_finished"])
                if seq_id in completed_seq_ids:
                    continue
                completed_seq_ids.add(seq_id)
                if use_tqdm:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    engine.metrics_manager.log_final_summary()

    if return_serialized:
        return serialized_outputs

    return list(outputs_by_seq.values())
