from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Input types ──────────────────────────────────────────────────────


class SequenceEntry(BaseModel):
    """Single entity in the input. Mirrors Protenix JSON schema."""

    model_config = {"extra": "allow"}


class CovalentBond(BaseModel):
    entity1: str
    copy_id1: int
    atom_name1: str
    entity2: str
    copy_id2: int
    atom_name2: str


class EmbedRequest(BaseModel):
    """Run pairformer trunk and cache the pair/single representations."""

    name: str
    sequences: list[dict[str, Any]]
    covalent_bonds: list[CovalentBond] = Field(default_factory=list)

    model_name: str = "protenix_base_default_v1.0.0"
    use_msa: bool = False
    use_template: bool = False
    dtype: str = "bf16"
    n_cycle: int = 10
    trimul_kernel: str = "cuequivariance"
    triatt_kernel: str = "cuequivariance"
    enable_cache: bool = True

    scope: str | None = None


class SampleRequest(BaseModel):
    """Run diffusion from a previously cached embedding."""

    embed_id: str
    seeds: list[int] = Field(default_factory=lambda: [101])
    n_sample: int = 1
    n_step: int = 200
    scope: str | None = None


class PredictRequest(BaseModel):
    """Convenience: trunk + diffusion in one call."""

    name: str
    sequences: list[dict[str, Any]]
    covalent_bonds: list[CovalentBond] = Field(default_factory=list)

    model_name: str = "protenix_base_default_v1.0.0"
    use_msa: bool = False
    use_template: bool = False
    dtype: str = "bf16"
    n_cycle: int = 10
    trimul_kernel: str = "cuequivariance"
    triatt_kernel: str = "cuequivariance"
    enable_cache: bool = True

    seeds: list[int] = Field(default_factory=lambda: [101])
    n_sample: int = 1
    n_step: int = 200

    scope: str | None = None


# ── Output types ─────────────────────────────────────────────────────


class StructureResult(BaseModel):
    seed: int
    sample_index: int
    format: str = "cif"
    content: str  # base64-encoded CIF text


class ConfidenceResult(BaseModel):
    seed: int
    sample_index: int
    plddt: float
    ptm: float
    iptm: float
    gpde: float
    ranking_score: float
    has_clash: bool


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    structures: list[StructureResult] | None = None
    confidence: list[ConfidenceResult] | None = None
    error: str | None = None


class EmbedStatus(BaseModel):
    embed_id: str
    status: str  # queued | running | done | error
    n_token: int | None = None
    error: str | None = None


# ── Simple responses ─────────────────────────────────────────────────


class EmbedResponse(BaseModel):
    embed_id: str
    status: str = "queued"


class SampleResponse(BaseModel):
    job_id: str
    status: str = "queued"


class PredictResponse(BaseModel):
    job_id: str
    status: str = "queued"
