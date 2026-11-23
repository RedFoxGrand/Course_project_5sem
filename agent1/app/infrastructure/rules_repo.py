from pathlib import Path
from typing import Any
import yaml
from app.domain.interfaces import IRulesRepository


class YamlRulesRepository(IRulesRepository):
    def __init__(self, rules_path: Path) -> None:
        self._rules_path = rules_path

    def load_rules(self) -> dict[str, Any]:
        with open(self._rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
