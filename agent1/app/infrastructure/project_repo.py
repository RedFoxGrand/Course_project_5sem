import json
from pathlib import Path
from typing import Any
from app.domain.interfaces import IProjectRepository


class FileProjectRepository(IProjectRepository):
    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, project_id: str, state_data: dict[str, Any]) -> None:
        target = self._storage_dir / f"{project_id}.json"
        with open(target, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2, default=str)

    def load(self, project_id: str) -> dict[str, Any] | None:
        target = self._storage_dir / f"{project_id}.json"
        if not target.exists():
            return None
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
