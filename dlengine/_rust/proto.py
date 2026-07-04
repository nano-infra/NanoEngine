"""Cross-process RPC protocol helpers."""

from __future__ import annotations

from .wrapper import export

# Shared request payload fragments. These are embedded by multiple protocol
# messages, so keep them stable before changing any socket-facing message.
_SAMPLER_PROTO = [
    "SamplingParams",
]

# Frontend/server <-> engine control-plane messages over ZMQ IPC.
_ENGINE_PROTO = [
    "RequestIn",
    "RequestMigrate",
    "StepOut",
]

# Engine/scheduler <-> model-runner messages. These are on the hot path; keep
# bytes ownership and from_bytes/to_bytes symmetry explicit.
_RUNNER_PROTO = [
    "MigrationIn",
    "RunnerIn",
    "RunnerOut",
]

__all__ = [
    *_SAMPLER_PROTO,
    *_ENGINE_PROTO,
    *_RUNNER_PROTO,
]

globals().update(export(tuple(__all__)))
