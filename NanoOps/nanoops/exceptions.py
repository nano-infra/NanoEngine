"""Custom exceptions for NanoOps."""


class NanoOpsError(Exception):
    """Base exception for NanoOps."""

    pass


class SessionExistsError(NanoOpsError):
    """Session already exists."""

    pass


class SessionNotFoundError(NanoOpsError):
    """Session not found."""

    pass


class ConfigError(NanoOpsError):
    """Configuration error."""

    pass


class ComponentSpawnError(NanoOpsError):
    """Failed to spawn component."""

    pass


class HealthCheckTimeout(NanoOpsError):
    """Health check timed out."""

    pass


class NanoCtrlError(NanoOpsError):
    """NanoCtrl communication error."""

    pass


class RayJobError(NanoOpsError):
    """Ray job submission/management error."""

    pass
