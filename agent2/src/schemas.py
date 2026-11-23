import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

EngineType = Literal["vllm", "sglang", "llama.cpp", "ollama", "tei"]
RoleType = Literal["llm", "embedding", "reranker"]
ProvenanceStatus = Literal["catalog", "declared", "benchmark", "estimated", "unknown"]


class RequirementProfile(BaseModel):
    """Входной профиль требований от Агента 1."""

    model_config = ConfigDict(extra="ignore")

    language: str = Field(default="ru", description="Целевой язык (ru, en)")
    domain: str = Field(
        default="general", description="Домен (code, technical, medical, general)"
    )
    max_context_length: int = Field(
        default=8192, gt=0, description="Минимальный контекст в токенах"
    )
    privacy_required: bool = Field(
        default=True, description="Только открытые локальные веса"
    )


class EvidenceItem(BaseModel):
    """Элемент доказательной базы параметра модели."""

    value: Any = Field(
        default=None,
        description="Значение характеристики (None, если данные не указаны)",
    )
    source: str = Field(..., description="Первоисточник параметра")
    status: ProvenanceStatus = Field(..., description="Статус достоверности сведений")

    @field_validator("source")
    @classmethod
    def validate_source_not_empty(cls, v: str) -> str:
        v_s = v.strip()
        if not v_s:
            raise ValueError("Источник не может быть пустой строкой")
        return v_s


class ExtractedCardMetadata(BaseModel):
    """Схема Structured Output для извлечения сведений из свободного текста README."""

    detected_license: str | None = Field(
        default=None, description="Лицензия, упомянутая в тексте"
    )
    supported_quantizations: list[str] = Field(
        default_factory=list,
        description="Типы квантования из текста (GGUF, Q4_K_M, AWQ, GPTQ, FP8)",
    )
    compatible_engines: list[EngineType] = Field(
        default_factory=list,
        description="Движки инференса из текста (vLLM, SGLang, Ollama, llama.cpp, TEI)",
    )
    benchmark_name: str | None = Field(
        default=None, description="Название метрики (MMLU, GPQA, MTEB)"
    )
    benchmark_value: float | None = Field(
        default=None, description="Числовой балл бенчмарка"
    )


class ModelCandidate(BaseModel):
    """Итоговая нормализованная карточка модели-кандидата."""

    model_id: str = Field(..., description="Идентификатор модели")
    name: str = Field(..., description="Отображаемое имя")
    organization: str = Field(..., description="Разработчик")
    role: RoleType = Field(..., description="Функциональная роль")
    license: str = Field(..., description="Лицензия")
    languages: list[str] = Field(
        default_factory=list, description="Поддерживаемые языки"
    )

    context_length: EvidenceItem = Field(..., description="Размер контекстного окна")
    parameters_count_b: EvidenceItem = Field(..., description="Параметры в млрд")
    active_parameters_count_b: EvidenceItem | None = None
    embedding_dimension: EvidenceItem | None = None

    is_local_compatible: bool = Field(default=True)
    is_context_sufficient: bool = Field(
        default=True, description="Удовлетворяет ли max_context_length"
    )
    supported_formats: list[str] = Field(default_factory=list)
    supported_quantizations: list[str] = Field(default_factory=list)
    compatible_engines: list[EngineType] = Field(default_factory=list)

    quality_score: EvidenceItem | None = None
    specialization: str = Field(default="")
    domain_match: bool = Field(default=False)
    warnings: list[str] = Field(default_factory=list)
    source_url: str = Field(..., description="Прямой URL репозитория")

    @field_validator("model_id", "name", "organization", "license")
    @classmethod
    def validate_non_empty(cls, v: str, info: ValidationInfo) -> str:
        if not v.strip():
            raise ValueError(f"Поле {info.field_name} не может быть пустым")
        return v.strip()

    @field_validator("source_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not re.match(r"^https?://\S+", v.strip()):
            raise ValueError(f"Некорректный URL: {v}")
        return v.strip()

    @field_validator("context_length")
    @classmethod
    def validate_ctx(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            if v.value is not None:
                raise ValueError(
                    "При статусе unknown значение context_length должно быть None"
                )
            return v
        if not isinstance(v.value, int) or v.value <= 0:
            raise ValueError("context_length должен быть целым положительным числом")
        return v

    @field_validator("parameters_count_b")
    @classmethod
    def validate_params(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            if v.value is not None:
                raise ValueError(
                    "При статусе unknown значение parameters_count_b должно быть None"
                )
            return v
        if not isinstance(v.value, (int, float)) or v.value <= 0:
            raise ValueError("parameters_count_b должен быть положительным числом")
        return v


class AgentTraceEvent(BaseModel):
    """Событие трассировки работы агента для UI и базы знаний OSTIS."""

    agent: str = "ModelResearcher"
    step: str
    details: str
    candidates_count: int = 0
