"""JSON/YAML export of a validated DeploymentPlan."""
import json
from pathlib import Path
from typing import Any
from .contracts import DeploymentPlan

def _plan(state_or_plan: dict[str, Any]) -> dict:
    if not isinstance(state_or_plan, dict):
        raise TypeError("Expected dictionary")
    raw = state_or_plan.get("deployment_plan", state_or_plan)
    if not isinstance(raw, dict):
        raise ValueError("No deployment_plan available")
    return DeploymentPlan.model_validate(raw).model_dump(mode="json")

def export_json(state_or_plan: dict, path: str | Path | None = None) -> str:
    output = json.dumps(_plan(state_or_plan), ensure_ascii=False, indent=2)
    if path is not None:
        Path(path).write_text(output, encoding="utf-8")
    return output

def export_yaml(state_or_plan: dict, path: str | Path | None = None) -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install PyYAML to export YAML") from exc
    output = yaml.safe_dump(_plan(state_or_plan), allow_unicode=True, sort_keys=False)
    if path is not None:
        Path(path).write_text(output, encoding="utf-8")
    return output
