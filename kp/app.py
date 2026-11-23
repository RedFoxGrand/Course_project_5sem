from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from agent import ReflectionAgent
from ostis_ontology import LocalDictionaryOstisGateway
from workflow import build_design_workflow

ostis_gateway = LocalDictionaryOstisGateway(base_dir="./isolated_ostis_kb")
reflection_agent = ReflectionAgent(ostis_gateway=ostis_gateway)
workflow_graph = build_design_workflow(reflection_agent=reflection_agent)

runtime_store: dict[str, Any] = {
    "traces_by_run": {},
}


async def run_all_scenarios() -> None:
    state_5090 = await workflow_graph.invoke({"scenario": "rtx5090"})
    state_5090_dup = await workflow_graph.invoke({"scenario": "rtx5090"})
    state_4090 = await workflow_graph.invoke({"scenario": "rtx4090"})

    runtime_store["traces_by_run"] = {
        "prj_rtx5090_001 (первый запуск)": state_5090["trace"],
        "prj_rtx5090_001 (повторный запуск / дедупликация)": state_5090_dup["trace"],
        "prj_rtx4090_002 (перенос на 24 ГБ VRAM)": state_4090["trace"],
    }


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await run_all_scenarios()
    cases = await ostis_gateway.list_all()
    print(f"Сохранено уникальных кейсов в словарной БЗ OSTIS: {len(cases)}")
    for c in cases:
        print(
            f" - {c['project_id']} ({c['hardware']['gpu_model']}): "
            f"VRAM={c['memory_breakdown']['total_vram_gb']} ГБ, "
            f"статус={c['overall_epistemic_status']}, похож на={c['similar_project_id']}"
        )
    print("\nСквозная трасса и переходы графа для prj_rtx4090_002:")
    for ev in runtime_store["traces_by_run"].get("prj_rtx4090_002 (перенос на 24 ГБ VRAM)", []):
        keys = ", ".join(ev["updated_keys"]) or "—"
        print(
            f"  [{ev['from_node']} -> {ev['to_node']}] {ev['agent']} ({ev['step']}): "
            f"{ev['details']} | поля: {keys}"
        )
    yield


app = FastAPI(
    title="Студент 5 — Рефлексивный агент и сквозная трасса OSTIS",
    lifespan=lifespan,
)


@app.get("/api/ostis/cases")
async def api_list_cases() -> list[dict[str, Any]]:
    return await ostis_gateway.list_all()


@app.get("/api/trace")
async def api_get_trace() -> dict[str, Any]:
    return dict(runtime_store["traces_by_run"])


@app.get("/", response_class=HTMLResponse)
async def render_dashboard() -> str:
    cases = await ostis_gateway.list_all()
    case_cards = ""
    for c in cases:
        comps_html = "".join(
            f"<li><b>{comp['role'].upper()}</b>: <code>{comp['model_id']}</code> "
            f"| Engine: <b>{comp['engine']}</b> | Quant: {comp['quantization']} "
            f"| Размещение: <b>{comp['placement'].upper()}</b> (ctx: {comp['max_model_len']}, seqs: {comp['max_num_seqs']})</li>"
            for comp in c["components"]
        )
        unc_html = "".join(
            f"<li><code>[{u['epistemic_status']}]</code> {u['description']} <i>({u['source']})</i></li>"
            for u in c["uncertainties"]
        )
        case_cards += f"""
        <div class="card">
            <h3>Проект: {c['project_id']} <span class="badge">{c['overall_epistemic_status']}</span></h3>
            <p><b>Описание:</b> {c['category']['normalized_summary']}</p>
            <p><b>Категория OSTIS:</b> <code>{c['category']['task_class']}</code> |
               <b>Оборудование:</b> <code>{c['category']['hardware_category']}</code> |
               <b>Связан с кейсом:</b> <code>{c['similar_project_id'] or 'исходный кейс'}</code></p>
            <p><b>Оценка памяти:</b> VRAM: <b>{c['memory_breakdown']['total_vram_gb']} / {c['hardware']['vram_gb']} ГБ</b> |
               RAM: <b>{c['memory_breakdown']['total_ram_gb']} / {c['hardware']['ram_gb']} ГБ</b></p>
            <p><b>Обоснование:</b> {c['decision_rationale']}</p>
            <h4>Конфигурация моделей:</h4><ul>{comps_html}</ul>
            <h4>Утверждения и гипотезы:</h4><ul>{unc_html}</ul>
            <details><summary><b>Показать SCs-код для БЗ OSTIS</b></summary>
            <pre>{c['scs_code']}</pre></details>
        </div>
        """

    traces_html = ""
    for run_title, events in runtime_store["traces_by_run"].items():
        rows = "".join(
            f"<tr>"
            f"<td><code>{t['from_node']} &rarr; {t['to_node']}</code></td>"
            f"<td><b>{t['agent']}</b></td>"
            f"<td><code>{t['step']}</code></td>"
            f"<td>{t['details']}</td>"
            f"<td><code>{', '.join(t['updated_keys']) or '—'}</code></td>"
            f"<td>{t['candidates_count']}</td>"
            f"</tr>"
            for t in events
        )
        traces_html += f"""
        <h3>{run_title}</h3>
        <table>
            <tr>
                <th>Переход графа</th>
                <th>Агент</th>
                <th>Шаг</th>
                <th>Детали выполнения</th>
                <th>Обновленные поля словаря</th>
                <th>Объектов</th>
            </tr>
            {rows}
        </table>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="utf-8">
        <title>OSTIS — История опыта и единая трасса агентов</title>
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
        <h1>Интеллектуальная среда проектирования ИИ-ассистентов (Модуль Студента 5)</h1>
        <h2>1. Накопленные случаи проектирования в БЗ OSTIS</h2>
        {case_cards}
        <h2>2. Сквозная трасса работы агентов и переходы графа состояний (Trace)</h2>
        {traces_html}
    </body>
    </html>
    """


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8005)