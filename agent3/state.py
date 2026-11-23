from typing import List, Optional, TypedDict

try:
    from .schemas import (
        AgentTraceEvent,
        HardwareProfile,
        ModelCandidate,
        ModelSet,
        RequirementProfile,
    )
except ImportError:
    from schemas import (
        AgentTraceEvent,
        HardwareProfile,
        ModelCandidate,
        ModelSet,
        RequirementProfile,
    )


class DesignState(TypedDict, total=False):
    """Глобальное состояние графа проектирования ИИ-ассистента."""
    project_id: str
    hardware: HardwareProfile
    requirements: Optional[RequirementProfile]
    candidates: List[ModelCandidate]
    model_sets: List[ModelSet]
    rejected_reasons: List[str]
    warnings: List[str]
    trace: List[AgentTraceEvent]
    errors: List[str]