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

        assert config.mode == "hybrid"

        self.config = config
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
        num_kvcache_blocks = self.executor.num_kvcache_blocks()
        self.config.num_kvcache_blocks = min(num_kvcache_blocks)
        print(f"kvcache blocks number updated, {self.config.num_kvcache_blocks=}")

    def add_request(self, seqs: Sequence | list[Sequence]):
        if isinstance(seqs, Sequence):
            seqs = [seqs]
        for seq in seqs:
            self.scheduler.add(seq)

    def prefill(self) -> None:
        tp_size = self.config.attention_tp
        dp_seqs = self.scheduler._schedule_prefill()
        token_ids = self.executor.run(dp_seqs, True)[::tp_size]
        self.scheduler.postprocess(dp_seqs, token_ids)
        return

    def step(self):
        tp_size = self.config.attention_tp
        dp_seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.executor.run(dp_seqs, is_prefill)[::tp_size]
        self.scheduler.postprocess(dp_seqs, token_ids)
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
