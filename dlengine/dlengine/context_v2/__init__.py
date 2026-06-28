# A series of persistent resources during the lifecycle of the runtime

# Transformation and discovery
# PeerAgentContext: P2P communication and discovery
# DistrbutedContext: Collective communication group

# For Attention Caches
# GQAContext: GQA attention cache management
# MLAContext: MLA attention cache management
# GDNContext: GDN attention cache management
# DSAContext: DSA attention cache management

# For MoE
# ExpertContext: Expert-level context for the runtime

# For Vision Language
# EmbeddingPoolContext: Embedding pool context for the runtime

# For Parameter
# ParameterContext: Weight parameters

import abc


class BaseContext(abc.ABC):
    @abc.abstractmethod
    def clear_context(self) -> None:
        pass

    @abc.abstractmethod
    def reset_context(self) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def get_context_type(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def get_context_name(cls) -> str:
        pass


class ContextManagerMixin:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.clear_context()
