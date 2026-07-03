__all__ = ["Config", "LLM", "LLMComponent", "SamplingParams"]


def __getattr__(name):
    if name == "Config":
        from .config import Config

        return Config
    if name == "SamplingParams":
        from .sampling_params import SamplingParams

        return SamplingParams
    if name in ("LLM", "LLMComponent"):
        from .llm_component import LLM, LLMComponent

        return {"LLM": LLM, "LLMComponent": LLMComponent}[name]
    raise AttributeError(f"module 'dlengine' has no attribute {name!r}")
