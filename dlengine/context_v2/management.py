from dlengine.context_v2 import BaseContext
from dlengine.context_v2.batch import reset_batch_context
from dlengine.context_v2.batch_out import reset_batch_out_context
from dlengine.context_v2.cache.csa import reset_csa_context
from dlengine.context_v2.cache.emb import reset_embedding_pool
from dlengine.context_v2.cache.gdn import reset_gdn_context
from dlengine.context_v2.cache.gqa import reset_gqa_context
from dlengine.context_v2.cache.hca import reset_hca_context
from dlengine.context_v2.cache.hisparse import reset_hisparse_context
from dlengine.context_v2.cache.indexer import reset_indexer_context
from dlengine.context_v2.cache.mla import reset_mla_context
from dlengine.context_v2.expert import reset_expert_context
from dlengine.disagg.p2p import reset_p2p_cache_transfer


class ContextManagement:
    @staticmethod
    def reset_runtime_contexts() -> None:
        reset_batch_context()
        reset_batch_out_context()
        reset_gqa_context()
        reset_mla_context()
        reset_hca_context()
        reset_csa_context()
        reset_gdn_context()
        reset_indexer_context()
        reset_hisparse_context()
        reset_embedding_pool()
        reset_p2p_cache_transfer()
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
