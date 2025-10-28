import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp

from nanovllm.engine.llm_engine import LLMEngine as NanoVLLMLLMEngine
from nanovllm.sampling_params import SamplingParams
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanodeploy.config import Config
from nanodeploy.engine.ray_executor import RayExecutor

from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.sequence import Sequence

from nanodeploy.worker.model_runner import ModelRunner


class LLMEngine(NanoVLLMLLMEngine):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.confing = config
        self.ps = []
        self.events = []
        # ctx = mp.get_context("spawn")
        # for i in range(1, config.tensor_parallel_size):
        #     event = ctx.Event()
        #     process = ctx.Process(target=ModelRunner, args=(config, i, event))
        #     process.start()
        #     self.ps.append(process)
        #     self.events.append(event)

        self.executor = RayExecutor(config=config)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        dp_stride = self.confing.tensor_parallel_size
        dp_seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.executor.run(dp_seqs, is_prefill)[::dp_stride]
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
