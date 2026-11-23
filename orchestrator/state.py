from typing import Any, Optional, TypedDict


class SharedDesignState(TypedDict, total=False):
    """Единое словарное состояние сквозного графа LangGraph (Agents 1 -> 5)."""

    project_id: str
    request: dict[str, Any]
    hardware: dict[str, Any]
    requirements: Optional[dict[str, Any]]
    clarification_questions: list[dict[str, Any]]

    catalog_version: str
    candidates: list[dict[str, Any]]
    is_pipeline_viable: Optional[bool]

    model_sets: list[dict[str, Any]]
    rejected_reasons: list[str]

    deployment_plan: Optional[dict[str, Any]]

    reflection: Optional[dict[str, Any]]
    scs_code: Optional[str]

    warnings: list[str]
    trace: list[dict[str, Any]]
    errors: list[str]