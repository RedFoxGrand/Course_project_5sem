"""Internal contracts; all messages exchanged between agents remain dict."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

class DTO(BaseModel):
    model_config = ConfigDict(extra="ignore")

class GPU(DTO):
    name: str = "Unknown GPU"
    vram_gb: float = Field(gt=0)
    count: int = Field(default=1, ge=1)

class Hardware(DTO):
    ram_gb: float = Field(gt=0)
    cpu: str = "Unknown CPU"
    os: str = "Linux"
    gpus: list[GPU] = Field(default_factory=list)
    reserved_ram_gb: float = Field(default=4, ge=0)
    reserved_vram_gb: float = Field(default=0, ge=0)

class Requirements(DTO):
    language: str = "unspecified"
    domain: str = "general"
    local_only: bool = True
    context_length: int = Field(default=4096, ge=128)
    parallel_sessions: int = Field(default=1, ge=1)
    min_context_length: int = Field(default=1024, ge=128)
    min_parallel_sessions: int = Field(default=1, ge=1)
    allow_reduce_context: bool = False
    allow_reduce_parallelism: bool = False
    allow_cpu_offload: bool = True
    priority: Literal["quality", "balanced", "speed", "memory"] = "balanced"
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    reranker_batch_size: int = Field(default=8, ge=1, le=512)
    license_allowlist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_limits(self):
        if self.min_context_length > self.context_length:
            raise ValueError("min_context_length > context_length")
        if self.min_parallel_sessions > self.parallel_sessions:
            raise ValueError("min_parallel_sessions > parallel_sessions")
        return self

class Model(DTO):
    id: str = Field(min_length=1)
    parameters_b: float | None = Field(default=None, gt=0)
    weight_sizes_gb: dict[str, float] = Field(default_factory=dict)
    weights_gb: float | None = Field(default=None, gt=0)
    weights_quantization: str | None = None
    quantizations: list[str] = Field(default_factory=lambda: ["fp16"], min_length=1)
    formats: list[str] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=list)
    max_context: int | None = Field(default=None, ge=128)
    num_hidden_layers: int | None = Field(default=None, ge=1)
    num_key_value_heads: int | None = Field(default=None, ge=1)
    head_dim: int | None = Field(default=None, ge=1)
    kv_cache_gb_per_1k: float | None = Field(default=None, gt=0)
    evidence_status: str = "unknown"
    languages: list[str] = Field(default_factory=list)
    local_available: bool | None = None
    license: str | None = None
    sources: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_weight_sizes(self):
        if any(size <= 0 for size in self.weight_sizes_gb.values()):
            raise ValueError("weight_sizes_gb must contain positive numbers")
        return self

class ModelSet(DTO):
    id: str
    score: float = 0
    llm: Model
    embedding: Model
    reranker: Model

class Evidence(DTO):
    name: str
    value_gb: float | None
    formula: str
    inputs: dict[str, Any]
    source: str
    status: Literal["catalog", "declared", "estimated", "unknown"]

class InferenceConfig(DTO):
    role: str
    model_id: str
    device: Literal["gpu", "cpu"]
    engine: str
    precision: str
    quantization: str
    weight_format: str
    parameters: dict[str, Any]
    command: list[str]
    notes: list[str] = Field(default_factory=list)

class MemoryEstimate(DTO):
    units: str = "GiB (binary units)"
    factors: list[Evidence]
    required_vram_gb: float | None
    available_vram_gb: float
    required_ram_gb: float | None
    available_ram_gb: float
    fit_by_estimate: bool | None
    requires_verification: bool = True
    assumptions: list[str]

class Option(DTO):
    id: str
    model_set_id: str
    model_set_score: float
    context_length: int
    parallel_sessions: int
    llm: InferenceConfig
    embedding: InferenceConfig
    reranker: InferenceConfig
    memory: MemoryEstimate
    status: Literal["estimated_fit", "requires_verification"]
    explanation: str
    warnings: list[str]
    sources: list[str]

class DeploymentPlan(DTO):
    project_id: str
    status: Literal["planned", "no_feasible_solution"]
    primary: Option | None = None
    fallback: Option | None = None
    rejected: list[dict[str, Any]]
    warnings: list[str]
    sources: list[str]
    created_at: str
    verification_required: bool = True
    constraints: dict[str, Any]
