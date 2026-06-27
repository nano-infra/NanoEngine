# A series of persistent resources during the lifecycle of the runtime

# Transformation and discovery
# PeerAgentContext: P2P communication
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
    def __init__(self):
        pass

    @abc.abstractmethod
    def __del__(self):
        pass

    @abc.abstractmethod
    def __enter__(self):
        pass

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @abc.abstractmethod
    def get_context() -> "BaseContext":
        pass

    @abc.abstractmethod
    def set_context(context: "BaseContext"):
        pass

    @abc.abstractmethod
    def clear_context():
        pass

    @abc.abstractmethod
    def reset_context():
        pass

    @abc.abstractmethod
    def get_context_type() -> str:
        pass

    @abc.abstractmethod
    def get_context_name() -> str:
        pass
