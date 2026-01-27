from nanodeploy._cpp import SamplingParams as _CppSamplingParams


class SamplingParams(_CppSamplingParams):
    def __init__(
        self,
        n: int = 1,
        best_of: int = 1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: bool = False,
        stop: list[str] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: int = 0,
    ):
        super().__init__()
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.stop = stop if stop is not None else []
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs
