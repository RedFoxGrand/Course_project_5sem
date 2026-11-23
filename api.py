import sys
import uuid
from pathlib import Path
from typing import Any, Literal

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

PROJECT_ROOT = Path(__file__).resolve().parent
AGENT1_DIR = PROJECT_ROOT / "agent1"
if str(AGENT1_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT1_DIR))

from app.domain.schemas import (
    ClarifyProjectPayload,
    CreateProjectPayload,
    UpdateHardwarePayload,
)
from app.infrastructure.project_repo import FileProjectRepository

from orchestrator.adapters import normalize_trace_event
from orchestrator.graph import build_master_workflow

app = FastAPI(title="Мультиагентный программный комплекс «Нейро Мастер» (Agents 1-5)")

workflow, ostis_gateway = build_master_workflow(
    agent3_llm=None,  # При необходимости передайте ChatOpenAI(...)
    enable_hf_background_sync=False,
)
project_repo = FileProjectRepository(PROJECT_ROOT / "data" / "projects")
traces_store: dict[str, list[dict[str, Any]]] = {}


def _resolve_status(state: dict[str, Any]) -> str:
    if state.get("clarification_questions"):
        return "needs_clarification"
    plan = state.get("deployment_plan") or {}
    if plan.get("primary") and state.get("reflection"):
        return "completed"
    if state.get("errors"):
        return "failed"
    return "completed"


def _format_response(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": state["project_id"],
        "status": _resolve_status(state),
        "hardware": state.get("hardware"),
        "requirements": state.get("requirements"),
        "clarification_questions": state.get("clarification_questions", []),
        "is_pipeline_viable": state.get("is_pipeline_viable"),
        "candidates_count": len(state.get("candidates", [])),
        "model_sets": state.get("model_sets", []),
        "deployment_plan": state.get("deployment_plan"),
        "reflection": state.get("reflection"),
        "scs_code": state.get("scs_code"),
        "warnings": state.get("warnings", []),
        "trace": state.get("trace", []),
        "errors": state.get("errors", []),
    }


async def _load_state(project_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": project_id}}
    snapshot = await workflow.aget_state(config)
    if snapshot and snapshot.values:
        return dict(snapshot.values)

    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Проект не найден")
    return saved


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "system": "NeuroMaster-Unified-5Agents"}


@app.post("/api/v1/projects/analyze")
async def analyze_project(payload: CreateProjectPayload) -> dict[str, Any]:
    project_id = payload.project_id or f"proj-{uuid.uuid4().hex[:8]}"
    initial_state: dict[str, Any] = {
        "project_id": project_id,
        "request": payload.request.model_dump(),
        "hardware": payload.hardware.model_dump(),
        "requirements": None,
        "clarification_questions": [],
        "candidates": [],
        "is_pipeline_viable": None,
        "model_sets": [],
        "rejected_reasons": [],
        "deployment_plan": None,
        "reflection": None,
        "scs_code": None,
        "warnings": [],
        "trace": [],
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result = await workflow.ainvoke(initial_state, config=config)
    traces_store[project_id] = result.get("trace", [])
    return _format_response(result)


@app.post("/api/v1/projects/{project_id}/clarify")
async def clarify_project(
    project_id: str, payload: ClarifyProjectPayload
) -> dict[str, Any]:
    current = await _load_state(project_id)
    req_dict = dict(current.get("request", {}))
    clarifications = dict(req_dict.get("user_clarifications", {}))
    clarifications.update(payload.answers)
    req_dict["user_clarifications"] = clarifications

    hw_dict = (
        payload.updated_hardware.model_dump()
        if payload.updated_hardware is not None
        else dict(current.get("hardware", {}))
    )

    resumed_state: dict[str, Any] = {
        **current,
        "request": req_dict,
        "hardware": hw_dict,
        "requirements": None,
        "clarification_questions": [],
        "candidates": [],
        "is_pipeline_viable": None,
        "model_sets": [],
        "rejected_reasons": [],
        "deployment_plan": None,
        "reflection": None,
        "scs_code": None,
        "warnings": [],
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result = await workflow.ainvoke(resumed_state, config=config)
    traces_store[project_id] = result.get("trace", [])
    return _format_response(result)


@app.patch("/api/v1/projects/{project_id}/hardware")
async def update_project_hardware(
    project_id: str, payload: UpdateHardwarePayload
) -> dict[str, Any]:
    current = await _load_state(project_id)
    req_dict = dict(current.get("request", {}))
    clarifications = dict(req_dict.get("user_clarifications", {}))
    if payload.note and payload.note.strip():
        clarifications["hardware_change_note"] = payload.note.strip()
    req_dict["user_clarifications"] = clarifications

    prev_hw = current.get("hardware", {})
    new_hw = payload.hardware.model_dump()
    trace = list(current.get("trace", []))
    trace.append(
        normalize_trace_event(
            {
                "agent": "WorkflowRouter",
                "step": "update_hardware",
                "details": (
                    f"Изменён профиль оборудования: "
                    f"{prev_hw.get('gpu_name') or 'CPU'} ({prev_hw.get('vram_gb', 0)} ГБ VRAM) -> "
                    f"{new_hw.get('gpu_name') or 'CPU'} ({new_hw.get('vram_gb', 0)} ГБ VRAM)."
                ),
            },
            "WorkflowRouter",
        )
    )

    resumed_state: dict[str, Any] = {
        **current,
        "request": req_dict,
        "hardware": new_hw,
        "requirements": None,
        "clarification_questions": [],
        "candidates": [],
        "is_pipeline_viable": None,
        "model_sets": [],
        "rejected_reasons": [],
        "deployment_plan": None,
        "reflection": None,
        "scs_code": None,
        "warnings": [],
        "trace": trace,
        "errors": [],
    }
    config = {"configurable": {"thread_id": project_id}}
    result = await workflow.ainvoke(resumed_state, config=config)
    traces_store[project_id] = result.get("trace", [])
    return _format_response(result)


@app.get("/api/v1/projects/{project_id}")
async def get_saved_project(project_id: str) -> dict[str, Any]:
    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Сохранённый проект не найден")
    return saved


@app.get("/api/v1/projects/{project_id}/export")
async def export_project(
    project_id: str, format: Literal["json", "yaml"] = Query(default="json")
):
    saved = project_repo.load(project_id)
    if not saved:
        raise HTTPException(status_code=404, detail="Сохранённый проект не найден")
    if format == "yaml":
        content = yaml.safe_dump(saved, allow_unicode=True, sort_keys=False)
        return PlainTextResponse(content=content, media_type="application/x-yaml")
    return saved


@app.get("/api/ostis/cases")
async def api_list_cases() -> list[dict[str, Any]]:
    return await ostis_gateway.list_all()


@app.get("/api/trace")
async def api_get_trace() -> dict[str, list[dict[str, Any]]]:
    return dict(traces_store)


@app.get("/", response_class=HTMLResponse)
async def render_dashboard() -> str:
    cases = await ostis_gateway.list_all()
    if not cases:
        cases_html = (
            "<p>Кейсов пока нет. Отправьте запрос на "
            "<code>POST /api/v1/projects/analyze</code> через <a href='/docs'>Swagger UI (/docs)</a>.</p>"
        )
    else:
        cases_html = ""
        for c in cases:
            comps_html = "".join(
                f"<li><b>{comp['role'].upper()}</b>: <code>{comp['model_id']}</code> | "
                f"Engine: <b>{comp['engine']}</b> | Quant: {comp['quantization']} | "
                f"Размещение: <b>{comp['placement'].upper()}</b></li>"
                for comp in c["components"]
            )
            cases_html += f"""
            <div class="card">
                <h3>Проект: {c['project_id']} <span class="badge">{c['overall_epistemic_status']}</span></h3>
                <p><b>Описание:</b> {c['category']['normalized_summary']}</p>
                <p><b>Память:</b> VRAM: <b>{c['memory_breakdown']['total_vram_gb']} / {c['hardware']['vram_gb']} ГБ</b> |
                   RAM: <b>{c['memory_breakdown']['total_ram_gb']} / {c['hardware']['ram_gb']} ГБ</b> |
                   <b>Похожий кейс:</b> <code>{c.get('similar_project_id') or 'нет'}</code></p>
                <ul>{comps_html}</ul>
                <details><summary><b>Показать SCs-код OSTIS</b></summary>
                <pre>{c.get('scs_code', '')}</pre></details>
            </div>
            """

    traces_html = ""
    for pid, events in traces_store.items():
        rows = "".join(
            f"<tr><td><b>{t.get('agent', '')}</b></td>"
            f"<td><code>{t.get('step', '')}</code></td>"
            f"<td>{t.get('details', '')}</td>"
            f"<td>{t.get('candidates_count', 0)}</td></tr>"
            for t in events
        )
        traces_html += f"<h3>Проект {pid}</h3><table><tr><th>Агент</th><th>Шаг</th><th>Детали</th><th>Объектов</th></tr>{rows}</table>"

    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>Нейро Мастер — Единый конвейер (Агенты 1–5)</title>
        <style>
            body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f8f9fa; color: #212529; }}
            .card {{ background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
            .badge {{ background: #e9ecef; padding: 3px 8px; border-radius: 4px; font-size: 0.85em; }}
            pre {{ background: #212529; color: #f8f9fa; padding: 12px; border-radius: 6px; overflow-x: auto; }}
            table {{ width: 100%; border-collapse: collapse; background: #fff; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
            th {{ background: #e9ecef; }}
        </style>
    </head>
    <body>
        <h1>Мультиагентный комплекс «Нейро Мастер»</h1>
        <p>Интерактивное тестирование API: <a href="/docs">/docs (Swagger UI)</a></p>
        <h2>1. Накопленные решения в БЗ OSTIS</h2>
        {cases_html}
        <h2>2. Сквозная трасса агентов</h2>
        {traces_html or "<p>Трасса появится после первого запуска анализа.</p>"}
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)