from typing import Any, Optional, TypedDict
from app.domain.schemas import (
    AssistantProjectRequest,
    ClarificationQuestion,
    HardwareProfile,
)


class DesignState(TypedDict, total=False):
    project_id: str
    request: AssistantProjectRequest
    hardware: HardwareProfile
    requirements: Optional[dict[str, Any]]
    clarification_questions: list[ClarificationQuestion]
    candidates: list[Any]
    is_pipeline_viable: Optional[bool]
    model_sets: list[Any]
    deployment_plan: Optional[Any]
    reflection: Optional[Any]
    trace: list[dict[str, Any]]
    errors: list[str]
