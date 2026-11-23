import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

EngineType = Literal["vllm", "sglang", "llama.cpp", "ollama", "tei"]
RoleType = Literal["llm", "embedding", "reranker"]
ProvenanceStatus = Literal["catalog", "declared", "benchmark", "estimated", "unknown"]
DomainType = Literal["general", "code", "technical", "medical"]


class AssistantProjectRequest(BaseModel):
    task_description: str = Field(..., min_length=1)
    user_clarifications: dict[str, str] = Field(default_factory=dict)


class HardwareProfile(BaseModel):
    cpu: str = Field(default="Generic x86_64")
    ram_gb: float = Field(..., gt=0)
    gpu_name: Optional[str] = None
    gpu_count: int = Field(default=0, ge=0)
    vram_gb: float = Field(default=0.0, ge=0.0)
    os: str = Field(default="Linux")
    allow_cpu_offload: bool = Field(default=False)


class ClarificationQuestion(BaseModel):
    field_name: str
    question: str
    importance: Literal["critical", "optional"] = "critical"
    reason_rule_id: str


class RequirementProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    language: str = Field(default="ru", pattern=r"^[a-z]{2}$")
    domain: DomainType = Field(default="general")
    max_context_length: int = Field(default=8192, gt=0)
    privacy_required: bool = Field(default=True)
    parallel_sessions: int = Field(default=1, ge=1, le=64)
    priority_quality: int = Field(default=3, ge=1, le=5)
    priority_speed: int = Field(default=3, ge=1, le=5)
    priority_memory: int = Field(default=3, ge=1, le=5)
    allow_cpu_offload: bool = Field(default=False)
    warnings: list[str] = Field(default_factory=list)

    def to_agent2_payload(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "domain": self.domain,
            "max_context_length": self.max_context_length,
            "privacy_required": self.privacy_required,
        }


class EvidenceItem(BaseModel):
    value: Any = Field(default=None)
    source: str = Field(...)
    status: ProvenanceStatus = Field(...)
    timestamp: Optional[str] = Field(default=None)

    @field_validator("source")
    @classmethod
    def validate_source_not_empty(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Источник (source) не может быть пустой строкой")
        return v_stripped


class ModelCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(...)
    name: str = Field(...)
    organization: str = Field(...)
    role: RoleType = Field(...)
    license: str = Field(...)
    languages: list[str] = Field(default_factory=list)

    context_length: EvidenceItem = Field(...)
    parameters_count_b: EvidenceItem = Field(...)
    active_parameters_count_b: EvidenceItem | None = Field(default=None)
    embedding_dimension: EvidenceItem | None = Field(default=None)

    is_local_compatible: bool = Field(default=True)
    is_context_sufficient: bool = Field(default=True)
    supported_formats: list[str] = Field(default_factory=list)
    supported_quantizations: list[str] = Field(default_factory=list)
    compatible_engines: list[EngineType] = Field(default_factory=list)

    quality_score: EvidenceItem | None = Field(default=None)
    specialization: str = Field(default="")
    domain_match: bool = Field(default=False)
    warnings: list[str] = Field(default_factory=list)
    source_url: str = Field(...)

    @field_validator("model_id", "name", "organization", "license")
    @classmethod
    def validate_non_empty_strings(cls, v: str, info: ValidationInfo) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError(f"Поле '{info.field_name}' не может быть пустым")
        return v_stripped

    @field_validator("source_url")
    @classmethod
    def validate_url_format(cls, v: str) -> str:
        url_stripped = v.strip()
        if not re.match(r"^https?://\S+", url_stripped):
            raise ValueError(f"Некорректный формат URL: '{url_stripped}'")
        return url_stripped

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            if v.value is not None:
                raise ValueError("При статусе unknown значение context_length должно быть None")
            return v
        if not isinstance(v.value, int) or isinstance(v.value, bool) or v.value <= 0:
            raise ValueError("context_length должен быть целым положительным числом")
        return v

    @field_validator("parameters_count_b")
    @classmethod
    def validate_parameters_count(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            if v.value is not None:
                raise ValueError("При статусе unknown значение parameters_count_b должно быть None")
            return v
        if not isinstance(v.value, (int, float)) or isinstance(v.value, bool) or v.value <= 0:
            raise ValueError("parameters_count_b должен быть положительным числом")
        v.value = float(v.value)
        return v

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score(cls, v: EvidenceItem | None) -> EvidenceItem | None:
        if v is None or v.status == "unknown":
            return v
        if not isinstance(v.value, (int, float)) or isinstance(v.value, bool):
            raise ValueError("Метрика качества quality_score должна быть числом")
        if v.value < 0:
            raise ValueError(f"Метрика качества не может быть отрицательной: {v.value}")
        v.value = float(v.value)
        return v


class MemoryEstimate(BaseModel):
    model_config = ConfigDict(extra="allow")

    required_vram_gb: float = Field(default=0.0, ge=0.0)
    required_ram_gb: float = Field(default=0.0, ge=0.0)
    requires_verification: bool = Field(default=False)
    assumptions: list[str] = Field(default_factory=list)


class InferenceConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: RoleType
    model_id: str
    engine: EngineType
    device: Literal["gpu", "cpu", "hybrid"] = "gpu"
    quantization: Optional[str] = None
    parameters: dict[str, Any] = Field(default_factory=dict)


class ModelSet(BaseModel):
    model_config = ConfigDict(extra="allow")

    llm: ModelCandidate
    embedding: ModelCandidate
    reranker: ModelCandidate
    tier: Literal["primary", "economic", "alternative"] = "primary"
    rationale: str = ""
    rejected_models: list[dict[str, Any]] = Field(default_factory=list)


class DeploymentPlan(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary_configs: list[InferenceConfig] = Field(default_factory=list)
    alternative_configs: list[InferenceConfig] = Field(default_factory=list)
    memory_estimate: Optional[MemoryEstimate] = None
    requires_verification: bool = False
    explanation: str = ""


class ReflectionRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    project_id: str
    summary: str = ""
    categories: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class LLMRequirementExtraction(BaseModel):
    language: Optional[str] = None
    domain: Optional[str] = None
    privacy_required: Optional[bool] = None
    max_context_length: Optional[int] = None
    parallel_sessions: Optional[int] = None
    priority_quality: Optional[int] = Field(default=None, ge=1, le=5)
    priority_speed: Optional[int] = Field(default=None, ge=1, le=5)
    priority_memory: Optional[int] = Field(default=None, ge=1, le=5)
    allow_cpu_offload: Optional[bool] = None
    missing_critical_info: list[ClarificationQuestion] = Field(default_factory=list)
    extraction_rationale: str


class AgentTraceEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent: str = "RequirementsAnalyst"
    step: str
    details: str
    candidates_count: int = 0
    status: Literal["started", "completed", "needs_clarification", "error"] = "completed"
    applied_rules: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CreateProjectPayload(BaseModel):
    project_id: Optional[str] = None
    request: AssistantProjectRequest
    hardware: HardwareProfile


class ClarifyProjectPayload(BaseModel):
    answers: dict[str, str]
    updated_hardware: Optional[HardwareProfile] = None


class UpdateHardwarePayload(BaseModel):
    hardware: HardwareProfile
    note: Optional[str] = None
