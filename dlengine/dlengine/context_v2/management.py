from dlengine.context_v2 import BaseContext
from dlengine.context_v2.batch import reset_batch_context
from dlengine.context_v2.batch_out import reset_batch_out_context
from dlengine.context_v2.cache.dsa import reset_dsa_context
from dlengine.context_v2.cache.hca import reset_hca_context
from dlengine.context_v2.expert import reset_expert_context


class ContextManagement:
    @staticmethod
    def reset_runtime_contexts() -> None:
        reset_batch_context()
        reset_batch_out_context()
        reset_hca_context()
        reset_dsa_context()
        reset_expert_context()

    @staticmethod
    def clear_context(context: BaseContext) -> None:
        context.clear_context()

    @staticmethod
    def reset_context(context: BaseContext) -> None:
        context.reset_context()


def reset_runtime_contexts() -> None:
    ContextManagement.reset_runtime_contexts()


__all__ = [
    "ContextManagement",
    "reset_runtime_contexts",
]
