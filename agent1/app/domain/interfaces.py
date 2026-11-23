from abc import ABC, abstractmethod
from typing import Any
from app.domain.schemas import (
    AssistantProjectRequest,
    HardwareProfile,
    LLMRequirementExtraction,
)


class IRulesRepository(ABC):
    @abstractmethod
    def load_rules(self) -> dict[str, Any]:
        raise NotImplementedError


class ILLMProvider(ABC):
    @abstractmethod
    def extract_requirements(
        self,
        request: AssistantProjectRequest,
        hardware: HardwareProfile,
        rules: dict[str, Any],
    ) -> LLMRequirementExtraction:
        raise NotImplementedError


class IProjectRepository(ABC):
    @abstractmethod
    def save(self, project_id: str, state_data: dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self, project_id: str) -> dict[str, Any] | None:
        raise NotImplementedError
