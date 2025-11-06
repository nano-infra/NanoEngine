import atexit
import uuid
from dataclasses import fields
from time import perf_counter
from typing import Literal

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy.config import Config
from nanodeploy.engine.ray_executor import RayExecutor
from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence


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
            seq.set_engine_id(self.engine_id, self.config.attention_sp)
            self.scheduler.add(seq)

    def free_to_be_migrated(self, seqs: Sequence | list[Sequence]):
        self.scheduler.free_to_be_migrated(seqs)

    def step(self):
        dp_size = self.config.attention_dp
        sp_size = self.config.attention_sp
        tp_size = self.config.attention_tp
        dp_seqs, is_prefill = self.scheduler.schedule()
        if is_prefill and self.config.mode == "decode":
            self.executor.migrate(dp_seqs)
        else:
            dp_sp_seqs = [
                [
                    seq
                    for seqs in dp_seqs
                    for seq in seqs
                    if seq.block_ctx(self.engine).master_sp_rank == sp_idx
                ]
                for sp_idx in self.config.attention_sp
            ]
            token_ids = self.executor.run(dp_sp_seqs, is_prefill)[::tp_size]
            token_ids = [
                token_ids[i * sp_size : (i + 1) + sp_size] for i in range(0, dp_size)
            ]
            dp_sp_seqs = [
                dp_sp_seqs[i * sp_size : (i + 1) + sp_size] for i in range(0, dp_size)
            ]
            self.scheduler.postprocess(dp_sp_seqs, token_ids)
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
            num_tokens += sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

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
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix(
                    {
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()
        return
