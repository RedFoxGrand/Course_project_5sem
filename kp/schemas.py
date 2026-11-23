import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

EngineType = Literal["vllm", "sglang", "llama.cpp", "ollama", "tei"]
RoleType = Literal["llm", "embedding", "reranker"]
ProvenanceStatus = Literal["catalog", "declared", "benchmark", "estimated", "unknown"]
EpistemicStatus = Literal["confirmed_fact", "hypothesis", "requires_verification"]
PlacementType = Literal["gpu", "cpu", "hybrid"]
PriorityType = Literal["quality", "speed", "memory", "balanced"]

_URL_REGEX = re.compile(
    r"^https?://"
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\.?|"
    r"localhost|"
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
    r"(?::\d+)?"
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def _normalize_lower(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


class BaseIsolatedModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra="ignore")


class TaskRequirementInput(BaseIsolatedModel):
    raw_query: str = Field(default="")
    language: str = Field(default="ru")
    domain: str = Field(default="general")
    max_context_length: int = Field(default=8192, gt=0)
    parallel_sessions: int = Field(default=1, gt=0)
    privacy_required: bool = Field(default=True)
    priority: PriorityType = Field(default="quality")

    @field_validator("language", "domain", "priority", mode="before")
    @classmethod
    def normalize_enums(cls, v: Any) -> Any:
        return _normalize_lower(v)


class HardwareInput(BaseIsolatedModel):
    gpu_model: str = Field(...)
    gpu_count: int = Field(default=1, ge=0)
    vram_gb: float = Field(..., ge=0.0)
    ram_gb: float = Field(..., gt=0.0)
    cpu_model: str = Field(default="x86_64")
    os_name: str = Field(default="Linux")

    @field_validator("gpu_model", "os_name")
    @classmethod
    def validate_not_empty(cls, v: str, info: ValidationInfo) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"Поле '{info.field_name}' не может быть пустым")
        return v_stripped

    @property
    def fingerprint(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", self.gpu_model.lower()).strip("_")
        return f"{self.gpu_count}x_{slug}_{int(self.vram_gb)}gb_vram_{int(self.ram_gb)}gb_ram_{self.os_name.lower()}"


class ProvenanceValue(BaseIsolatedModel):
    value: float = Field(..., ge=0.0)
    source: str = Field(default="unknown")
    status: ProvenanceStatus = Field(default="estimated")

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        return _normalize_lower(v)

    @field_validator("source", mode="before")
    @classmethod
    def sanitize_source(cls, v: Any) -> str:
        if v is None:
            return "unknown"
        text = str(v).strip()
        if not text or text in ("—", "-", "None", "null"):
            return "unknown"
        return text


class ComponentRunConfig(BaseIsolatedModel):
    role: RoleType = Field(...)
    model_id: str = Field(...)
    engine: EngineType = Field(...)
    weight_format: str = Field(default="safetensors")
    quantization: str = Field(default="fp16")
    placement: PlacementType = Field(default="gpu")
    max_model_len: int = Field(default=8192, gt=0)
    max_num_seqs: int = Field(default=1, gt=0)
    keep_loaded: bool = Field(default=True)
    source_url: str = Field(default="")

    @field_validator("role", "engine", "placement", mode="before")
    @classmethod
    def normalize_literals(cls, v: Any) -> Any:
        return _normalize_lower(v)

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Поле 'model_id' не может быть пустым")
        return v_stripped

    @field_validator("source_url", mode="before")
    @classmethod
    def sanitize_url(cls, v: Any) -> str:
        if v is None:
            return ""
        url_stripped = str(v).strip()
        if not url_stripped or url_stripped in ("—", "-", "None", "null"):
            return ""
        if not _URL_REGEX.match(url_stripped):
            return ""
        return url_stripped


class MemoryBreakdownInput(BaseIsolatedModel):
    llm_weights: ProvenanceValue
    llm_kv_cache: ProvenanceValue
    embedding_resident_memory: ProvenanceValue
    reranker_resident_memory: ProvenanceValue
    runtime_overhead: ProvenanceValue
    safety_reserve: ProvenanceValue
    total_vram_gb: float = Field(..., ge=0.0)
    total_ram_gb: float = Field(..., ge=0.0)


class RejectedOptionInput(BaseIsolatedModel):
    candidate_id: str = Field(...)
    role: Literal["llm", "embedding", "reranker", "model_set"] = Field(...)
    reason: str = Field(...)

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Any:
        return _normalize_lower(v)

    @field_validator("candidate_id", "reason")
    @classmethod
    def validate_non_empty_fields(cls, v: str, info: ValidationInfo) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"Поле '{info.field_name}' не может быть пустым")
        return v_stripped


class ExperienceCategoryOutput(BaseIsolatedModel):
    normalized_summary: str = Field(...)
    task_class: str = Field(...)
    language_category: str = Field(...)
    domain_category: str = Field(...)
    hardware_category: str = Field(...)
    priority_category: PriorityType = Field(...)

    @field_validator("priority_category", mode="before")
    @classmethod
    def normalize_priority(cls, v: Any) -> Any:
        return _normalize_lower(v)

    @field_validator("normalized_summary", "task_class", "language_category", "domain_category", "hardware_category")
    @classmethod
    def validate_strings(cls, v: str, info: ValidationInfo) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"Поле '{info.field_name}' не может быть пустым")
        return v_stripped


class UncertaintyStatement(BaseIsolatedModel):
    statement_id: str = Field(...)
    parameter_name: str = Field(...)
    description: str = Field(...)
    source: str = Field(...)
    provenance_status: ProvenanceStatus = Field(...)
    epistemic_status: EpistemicStatus = Field(...)


class ReflectionRecord(BaseIsolatedModel):
    reflection_id: str = Field(...)
    project_id: str = Field(...)
    catalog_version: str = Field(...)
    deduplication_hash: str = Field(...)
    similar_project_id: str | None = Field(default=None)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    category: ExperienceCategoryOutput
    requirements: TaskRequirementInput
    hardware: HardwareInput

    primary_set_id: str = Field(...)
    alternative_set_id: str | None = Field(default=None)
    components: list[ComponentRunConfig] = Field(..., min_length=1)
    memory_breakdown: MemoryBreakdownInput
    decision_rationale: str = Field(...)

    rejected_options: list[RejectedOptionInput] = Field(default_factory=list)
    uncertainties: list[UncertaintyStatement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    overall_epistemic_status: EpistemicStatus = Field(default="confirmed_fact")


class AgentTraceEvent(BaseIsolatedModel):
    agent: str = Field(default="ReflectionAgent")
    from_node: str = Field(default="")
    to_node: str = Field(default="")
    step: str = Field(...)
    details: str = Field(...)
    updated_keys: list[str] = Field(default_factory=list)
    candidates_count: int = Field(default=0, ge=0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())