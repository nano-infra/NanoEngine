import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence

from nanovllm.engine.llm_engine import LLMEngine as NanoVLLMLLMEngine

from nanodeploy.worker.model_runner import ModelRunner
from nanodeploy.config import Config

from nanodeploy.engine.scheduler import Scheduler
from nanodeploy.engine.ray_executor import RayExecutor


class LLMEngine(NanoVLLMLLMEngine):
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
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

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.executor.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids[0])
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens
