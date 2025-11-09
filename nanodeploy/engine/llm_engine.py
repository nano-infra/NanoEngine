import atexit
import time
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Literal

import numpy as np

from tqdm.auto import tqdm

from transformers import AutoTokenizer

from nanodeploy.config import Config
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence
from nanodeploy.logging import get_logger


logger = get_logger()


class LLMEngine:
    def __init__(self, model, **kwargs):
        self.engine_id = str(uuid.uuid4())

        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.config = config
        self.config.engine_id = self.engine_id
        self.ps = []
        self.events = []

        self.executor = RayExecutor(config=config)
        self.update_num_kvcache_blocks()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        del self.executor

    def update_num_kvcache_blocks(self):
        self.config.num_kvcache_blocks = self.executor.update_kvcache_blocks()

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            seq.set_engine_id(
                self.engine_id, self.config.attention_dp, self.config.attention_sp
            )
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self.scheduler.free_to_be_migrated(seqs)

    def step(self):
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        sch_begin = time.time()
        dp_seqs, is_prefill = self.scheduler.schedule()
        dp_sp_seqs = [
            [
                seq
                for seq in seqs
                if seq.block_ctx(self.engine_id).master_sp_rank == sp_idx
            ]
            for seqs in dp_seqs
            for sp_idx in range(self.config.attention_sp)
        ]
        dp_sp_tp_seqs: list[list[Sequence]] = [
            seqs for seqs in dp_sp_seqs for _ in range(tp_size)
        ]
        sch_end = time.time()
        post_sch_begin = 0
        post_sch_end = 0
        if is_prefill and self.config.mode == "decode":
            if not self.config.dummy_prefill:
                logger.info("perform migration")
                for seqs in dp_sp_tp_seqs:
                    for seq in seqs:
                        logger.info(
                            f"{seq.block_ctx(seq.backup_engine_id).block_location, seq.block_ctx(seq.active_engine_id).block_location}"
                        )
                        seq.block_ctx(seq.active_engine_id).num_dispatched_tokens[
                            seq.block_ctx().master_sp_rank
                        ] += 1
                self.executor.migrate(dp_sp_tp_seqs)
            else:
                [[seq.append_token(0) for seq in seqs] for seqs in dp_seqs]
        else:
            token_ids = self.executor.run(dp_sp_tp_seqs, is_prefill)[::tp_size]
            post_sch_begin = time.time()
            token_ids = [
                token_ids[i * sp_size : (i + 1) * sp_size] for i in range(0, dp_size)
            ]
            dp_sp_seqs = [
                dp_sp_seqs[i * sp_size : (i + 1) * sp_size] for i in range(0, dp_size)
            ]
            self.scheduler.postprocess(dp_sp_seqs, token_ids)
            post_sch_end = time.time()
        outputs = []
        num_tokens = 0
        for seqs in dp_seqs:
            outputs.extend(
                [
                    (seq.seq_id, seq.completion_token_ids)
                    for seq in seqs
                    if seq.is_finished
                ]
            )
            num_tokens += (
                sum(len(seq) for seq in seqs)
                if is_prefill
                else -len(seqs) * self.config.loop_count
            )
        return (
            outputs,
            num_tokens,
            [len(seqs) for seqs in dp_seqs],
            (sch_end - sch_begin) * 1000,
            (post_sch_end - post_sch_begin) * 1000,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def p2p_init(
        self, remote_engine_name: str, num_kv_blocks: int, remote_world_size: int
    ):
        return self.executor.p2p_init(
            remote_engine_name, num_kv_blocks, remote_world_size
        )

    def p2p_connect(
        self, remote_engine_name: str, remote_endpoint_infos: list[dict[int, dict]]
    ):
        return self.executor.p2p_connect(remote_engine_name, remote_endpoint_infos)

    def generate(
        self,
        use_tqdm: bool = True,
    ) -> None:
        num_reqs = len(self.scheduler.waiting)
        if use_tqdm:
            pbar = tqdm(total=num_reqs, desc="Generating", dynamic_ncols=True)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, bs, sch_latency, post_sch_latency = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                itl = (perf_counter() - t) * 1000 / self.config.loop_count
                pbar.set_postfix(
                    {
                        "bs": f"{bs}",
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                        "sch_ovhd": f"{sch_latency:.2f}ms",
                        "post_sch_ovhd": f"{post_sch_latency:.2f}ms",
                        "itl": f"{itl:.2f}ms",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()
        return
