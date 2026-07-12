from typing import TYPE_CHECKING


__all__ = ["LLM", "SamplingParams"]


if TYPE_CHECKING:
    from nanodeploy.llm import LLM
    from nanodeploy.sampling_params import SamplingParams


def __getattr__(name: str):
    if name == "LLM":
        from nanodeploy.llm import LLM

        return LLM
    if name == "SamplingParams":
        from nanodeploy.sampling_params import SamplingParams

        return SamplingParams
    raise AttributeError(f"module 'nanodeploy' has no attribute {name!r}")
