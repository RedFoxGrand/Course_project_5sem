import uuid
from typing import Any, Literal
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from app.agents.requirements_agent import RequirementsAnalystAgent
from app.config import settings
from app.domain.schemas import (
    AgentTraceEvent,
    AssistantProjectRequest,
    ClarifyProjectPayload,
    CreateProjectPayload,
    HardwareProfile,
    RequirementProfile,
    UpdateHardwarePayload,
)
from app.infrastructure.llm_adapter import OpenAICompatibleRequirementsAdapter
from app.infrastructure.project_repo import FileProjectRepository
from app.infrastructure.rules_repo import YamlRulesRepository
from app.workflow.graph import build_agent1_workflow, serialize_state_value
from app.workflow.state import DesignState

app = FastAPI(title="AI Assistant Designer — Agent 1 (Requirements Analyst API)")

rules_repo = YamlRulesRepository(settings.rules_path)
project_repo = FileProjectRepository(settings.projects_dir)
llm_adapter = OpenAICompatibleRequirementsAdapter(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
    model=settings.llm_model,
    use_mock=settings.use_mock_llm,
    timeout_seconds=settings.llm_timeout_seconds,
)
analyst_agent = RequirementsAnalystAgent(rules_repo=rules_repo, llm_provider=llm_adapter)
workflow = build_agent1_workflow(analyst_agent=analyst_agent, project_repo=project_repo)


def _resolve_status(state: DesignState) -> str:
    if state.get("errors") and not state.get("requirements"):
        return "error"
    if state.get("clarification_questions"):
        return "needs_clarification"
    return "completed"


def _format_response(state: DesignState) -> dict[str, Any]:
    raw_req = serialize_state_value(state.get("requirements"))
    agent2_input = (
        RequirementProfile.model_validate(raw_req).to_agent2_payload()
        if raw_req is not None
        else None
    )
    return {
        "project_id": state["project_id"],
        "status": _resolve_status(state),
        "hardware": serialize_state_value(state["hardware"]),
        "requirements": raw_req,
        "agent2_input": agent2_input,
        "clarification_questions": serialize_state_value(
            state.get("clarification_questions", [])
        ),
        "candidates": serialize_state_value(state.get("candidates", [])),
        "is_pipeline_viable": state.get("is_pipeline_viable"),
        "trace": serialize_state_value(state.get("trace", [])),
        "errors": state.get("errors", []),
    }


async def _load_existing_state(project_id: str) -> DesignState:
    config = {"configurable": {"thread_id": project_id}}
    snapshot = await workflow.aget_state(config)
    if snapshot and snapshot.values:
        return snapshot.values

    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Проект не найден")

    return {
        "project_id": saved["project_id"],
        "request": AssistantProjectRequest.model_validate(saved["request"]),
        "hardware": HardwareProfile.model_validate(saved["hardware"]),
        "requirements": saved.get("requirements"),
        "clarification_questions": [],
        "candidates": saved.get("candidates", []),
        "is_pipeline_viable": saved.get("is_pipeline_viable"),
        "model_sets": saved.get("model_sets", []),
        "deployment_plan": saved.get("deployment_plan"),
        "reflection": saved.get("reflection"),
        "trace": saved.get("trace", []),
        "errors": saved.get("errors", []),
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "agent": "RequirementsAnalyst"}


@app.post("/api/v1/projects/analyze")
async def analyze_project(payload: CreateProjectPayload) -> dict[str, Any]:
    project_id = payload.project_id or f"proj-{uuid.uuid4().hex[:8]}"
    initial_state: DesignState = {
        "project_id": project_id,
        "request": payload.request,
        "hardware": payload.hardware,
        "requirements": None,
        "clarification_questions": [],
        "candidates": [],
        "is_pipeline_viable": None,
        "model_sets": [],
        "deployment_plan": None,
        "reflection": None,
        "trace": [],
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result_state = await workflow.ainvoke(initial_state, config=config)
    return _format_response(result_state)


@app.post("/api/v1/projects/{project_id}/clarify")
async def clarify_project(project_id: str, payload: ClarifyProjectPayload) -> dict[str, Any]:
    current_state = await _load_existing_state(project_id)
    updated_request = current_state["request"].model_copy(deep=True)
    updated_request.user_clarifications.update(payload.answers)

    updated_hardware = (
        payload.updated_hardware
        if payload.updated_hardware is not None
        else current_state["hardware"]
    )

    resumed_state: DesignState = {
        **current_state,
        "request": updated_request,
        "hardware": updated_hardware,
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result_state = await workflow.ainvoke(resumed_state, config=config)
    return _format_response(result_state)


@app.patch("/api/v1/projects/{project_id}/hardware")
async def update_project_hardware(
    project_id: str, payload: UpdateHardwarePayload
) -> dict[str, Any]:
    current_state = await _load_existing_state(project_id)
    updated_request = current_state["request"].model_copy(deep=True)
    if payload.note and payload.note.strip():
        updated_request.user_clarifications["hardware_change_note"] = payload.note.strip()

    trace = list(current_state.get("trace", []))
    prev_hw = current_state["hardware"]
    trace.append(
        AgentTraceEvent(
            agent="WorkflowRouter",
            step="update_hardware",
            details=(
                f"Изменён профиль оборудования: "
                f"{prev_hw.gpu_name or 'CPU'} ({prev_hw.vram_gb} ГБ VRAM) -> "
                f"{payload.hardware.gpu_name or 'CPU'} ({payload.hardware.vram_gb} ГБ VRAM)."
            ),
            status="completed",
            applied_rules=["REUSE_PROJECT_ON_HARDWARE_CHANGE"],
        ).model_dump()
    )

    resumed_state: DesignState = {
        **current_state,
        "request": updated_request,
        "hardware": payload.hardware,
        "requirements": None,
        "clarification_questions": [],
        "candidates": [],
        "is_pipeline_viable": None,
        "model_sets": [],
        "deployment_plan": None,
        "reflection": None,
        "trace": trace,
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result_state = await workflow.ainvoke(resumed_state, config=config)
    return _format_response(result_state)


@app.get("/api/v1/projects/{project_id}")
async def get_saved_project(project_id: str) -> dict[str, Any]:
    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Сохранённый проект не найден")
    return saved


@app.get("/api/v1/projects/{project_id}/export")
async def export_project(
    project_id: str,
    format: Literal["json", "yaml"] = Query(default="json"),
):
    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Сохранённый проект не найден")
    if format == "yaml":
        content = yaml.safe_dump(saved, allow_unicode=True, sort_keys=False)
        return PlainTextResponse(content=content, media_type="application/x-yaml")
    return saved
