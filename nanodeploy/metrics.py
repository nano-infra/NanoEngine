from dataclasses import dataclass


@dataclass
class SeqMetrics:
    TTFT: float | None = None


@dataclass
class ServerMetrics:
    pass
