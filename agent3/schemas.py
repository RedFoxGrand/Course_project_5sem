import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

EngineType = Literal["vllm", "sglang", "llama.cpp", "ollama", "tei"]
RoleType = Literal["llm", "embedding", "reranker"]
ProvenanceStatus = Literal["catalog", "declared", "benchmark", "estimated", "unknown"]


class RequirementProfile(BaseModel):
    """Профиль требований к ИИ-ассистенту."""
    model_config = ConfigDict(extra="ignore")

    language: str = Field(default="ru", description="Целевой язык (ru, en)")
    domain: str = Field(default="general", description="Предметная область (general, code, technical, medical)")
    max_context_length: int = Field(default=8192, gt=0, description="Минимально требуемый контекст в токенах")
    privacy_required: bool = Field(
        default=True, description="Если True — разрешены только локальные open-weight модели"
    )


class HardwareProfile(BaseModel):
    """Профиль аппаратного обеспечения сервера."""
    model_config = ConfigDict(extra="ignore")

    cpu_cores: int = Field(default=8, description="Количество ядер CPU")
    ram_gb: float = Field(default=64.0, description="Объем оперативной памяти в ГБ")
    gpu_count: int = Field(default=1, description="Количество видеокарт")
    gpu_vram_gb: float = Field(default=24.0, description="Суммарный объем VRAM в ГБ")
    os_name: str = Field(default="Linux", description="Операционная система")


class EvidenceItem(BaseModel):
    """Элемент доказательной базы параметра модели."""
    model_config = ConfigDict(extra="ignore")

    value: Any = Field(default=None, description="Фактическое или расчетное значение характеристики")
    source: str = Field(..., description="Первоисточник (catalog, URL репозитория HF, лидерборд)")
    status: ProvenanceStatus = Field(..., description="Статус достоверности сведений")

    @field_validator("source")
    @classmethod
    def validate_source_not_empty(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Источник не может быть пустой строкой")
        return v_stripped


class ModelCandidate(BaseModel):
    """Нормализованная карточка модели-кандидата."""
    model_config = ConfigDict(extra="ignore")

    model_id: str = Field(..., description="Канонический идентификатор модели")
    name: str = Field(..., description="Отображаемое название модели")
    organization: str = Field(..., description="Организация или сообщество-разработчик")
    role: RoleType = Field(..., description="Функциональная роль в контуре RAG")
    license: str = Field(..., description="Лицензия распространения весов")
    languages: list[str] = Field(default_factory=list, description="Поддерживаемые языки")

    context_length: EvidenceItem = Field(..., description="Максимальное контекстное окно в токенах")
    parameters_count_b: EvidenceItem = Field(..., description="Общее число параметров в миллиардах (B)")
    active_parameters_count_b: EvidenceItem | None = Field(default=None, description="Активные параметры в миллиардах")
    embedding_dimension: EvidenceItem | None = Field(default=None, description="Размерность вектора")

    is_local_compatible: bool = Field(default=True, description="Доступны ли веса для локального развертывания")
    is_context_sufficient: bool = Field(default=True, description="Удовлетворяет ли контексту")
    supported_formats: list[str] = Field(default_factory=list, description="Форматы весов")
    supported_quantizations: list[str] = Field(default_factory=list, description="Поддерживаемые квантования")
    compatible_engines: list[EngineType] = Field(default_factory=list, description="Совместимые движки инференса")

    quality_score: EvidenceItem | None = Field(default=None, description="Метрика качества")
    specialization: str = Field(default="", description="Краткое описание специализации модели")
    domain_match: bool = Field(default=False, description="Совпадение предметной области")
    warnings: list[str] = Field(default_factory=list, description="Предупреждения верификации")
    source_url: str = Field(..., description="Прямой URL источника или репозитория модели")

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
        url_regex = re.compile(
            r"^https?://"
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,63}\.?|"
            r"localhost|"
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"
            r"(?::\d+)?"
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )
        if not url_regex.match(url_stripped):
            raise ValueError(f"Некорректный формат URL: '{url_stripped}'")
        return url_stripped

    @field_validator("context_length")
    @classmethod
    def validate_context_length(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            return v
        if not isinstance(v.value, (int, float)) or isinstance(v.value, bool):
            raise ValueError("Значение context_length должно быть числом")
        if v.value <= 0:
            raise ValueError(f"Размер контекстного окна должен быть > 0")
        v.value = int(v.value)
        return v

    @field_validator("parameters_count_b")
    @classmethod
    def validate_parameters_count(cls, v: EvidenceItem) -> EvidenceItem:
        if v.status == "unknown":
            return v
        if not isinstance(v.value, (int, float)) or isinstance(v.value, bool):
            raise ValueError("Значение parameters_count_b должно быть числом")
        if v.value <= 0:
            raise ValueError(f"Число параметров должно быть > 0")
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
            raise ValueError(f"Метрика качества не может быть отрицательной")
        v.value = float(v.value)
        return v


class ModelSet(BaseModel):
    """Спецификация комплекта моделей."""
    set_name: str = Field(..., description="Идентификатор конфигурации сборки")
    llm: ModelCandidate = Field(..., description="Выбранная языковая модель")
    embedding: ModelCandidate = Field(..., description="Выбранная модель векторизации")
    reranker: ModelCandidate = Field(..., description="Выбранная модель ранжирования")
    estimated_total_vram_gb: float = Field(..., description="Предварительная оценка требуемой видеопамяти (ГБ)")
    rationale: str = Field(..., description="Обоснование выбора комплекта")


class Agent3Output(BaseModel):
    """Структурированный вывод результатов подбора комплектов."""
    recommended_sets: List[ModelSet] = Field(..., description="Список рекомендованных конфигураций")
    rejected_reasons: List[str] = Field(..., description="Причины отклонения неоптимальных конфигураций")


class AgentTraceEvent(BaseModel):
    """Событие трассировки логики агента."""
    agent: str = Field(default="ModelSelector")
    step: str
    details: str
    candidates_count: int = Field(default=0)