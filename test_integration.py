import pytest
from fastapi.testclient import TestClient

from api import app
from orchestrator.graph import build_master_workflow


@pytest.mark.asyncio
async def test_full_pipeline_5090_dup_and_4090_transfer(tmp_path):
    """Проверка полного конвейера A1->A2->A3->A4->A5: RTX 5090, дедупликация и перенос на RTX 4090."""
    from agent5.ostis_ontology import LocalDictionaryOstisGateway

    gateway = LocalDictionaryOstisGateway(base_dir=tmp_path / "ostis_kb")
    workflow, _ = build_master_workflow(
        agent3_llm=None,
        ostis_gateway=gateway,
        enable_hf_background_sync=False,
    )

    state_5090 = {
        "project_id": "it_5090_001",
        "request": {
            "task_description": (
                "Локальный русскоязычный RAG-ассистент по технической документации. "
                "Данные нельзя передавать во внешние API. Приоритет качество ответа. "
                "5 параллельных сессий, допускается перенос embedding или reranker на CPU."
            ),
            "user_clarifications": {},
        },
        "hardware": {
            "cpu": "AMD Ryzen 9 9950X",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "os": "Linux",
            "allow_cpu_offload": True,
        },
        "trace": [],
        "errors": [],
        "warnings": [],
        "rejected_reasons": [],
    }

    # 1. Первый прогон RTX 5090
    res1 = await workflow.ainvoke(state_5090, config={"configurable": {"thread_id": "it_5090_001"}})
    assert res1["is_pipeline_viable"] is True
    assert len(res1["model_sets"]) >= 1
    assert res1["deployment_plan"]["status"] == "planned"
    assert res1["deployment_plan"]["primary"] is not None
    assert res1["reflection"] is not None
    assert res1["scs_code"] and "concept_ai_assistant_design_case" in res1["scs_code"]
    assert res1["reflection"]["similar_project_id"] is None

    # Проверяем, что в сквозной трассе отметились все 5 агентов
    agents_in_trace = {ev["agent"] for ev in res1["trace"]}
    assert {
        "RequirementsAnalyst",
        "ModelResearcher",
        "ModelSelector",
        "inference_configurator",
        "ReflectionAgent",
    }.issubset(agents_in_trace)

    # 2. Повторный прогон того же конфига -> дедупликация в OSTIS
    state_dup = {**state_5090, "project_id": "it_5090_dup"}
    res_dup = await workflow.ainvoke(state_dup, config={"configurable": {"thread_id": "it_5090_dup"}})
    assert any(ev["step"] == "prevent_duplication" for ev in res_dup["trace"])
    assert len(await gateway.list_all()) == 1

    # 3. Перенос на RTX 4090 -> новый кейс со связью similar_project_id
    state_4090 = {
        "project_id": "it_4090_002",
        "request": {
            "task_description": (
                "Локальный русскоязычный RAG-ассистент по технической документации на RTX 4090. "
                "3 параллельные сессии, допускается перенос embedding и reranker на CPU."
            ),
            "user_clarifications": {},
        },
        "hardware": {
            "cpu": "AMD Ryzen 9 9950X",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "vram_gb": 24.0,
            "os": "Linux",
            "allow_cpu_offload": True,
        },
        "trace": [],
        "errors": [],
        "warnings": [],
        "rejected_reasons": [],
    }
    res2 = await workflow.ainvoke(state_4090, config={"configurable": {"thread_id": "it_4090_002"}})
    assert res2["deployment_plan"]["status"] == "planned"
    assert res2["reflection"]["similar_project_id"] == "it_5090_001"
    assert len(await gateway.list_all()) == 2


def test_unified_api_clarify_and_hardware_patch():
    """Проверка объединённого FastAPI: /health, уточнение вопросов, смена GPU, /export и OSTIS."""
    client = TestClient(app)

    assert client.get("/health").status_code == 200

    # Неполный запрос -> статус needs_clarification
    r1 = client.post(
        "/api/v1/projects/analyze",
        json={
            "project_id": "api-test-01",
            "request": {"task_description": "Нужен ассистент для поиска по файлам."},
            "hardware": {
                "ram_gb": 64.0,
                "gpu_name": "NVIDIA GeForce RTX 5090",
                "gpu_count": 1,
                "vram_gb": 0.0,
                "os": "Linux",
                "allow_cpu_offload": True,
            },
        },
    )
    assert r1.status_code == 200
    assert r1.json()["status"] == "needs_clarification"

    # Доотправка ответов -> полный прогон всех 5 агентов
    r2 = client.post(
        "/api/v1/projects/api-test-01/clarify",
        json={"answers": {"language": "ru", "domain": "technical", "vram_gb": "32"}},
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "completed"
    assert d2["deployment_plan"]["status"] == "planned"
    assert d2["reflection"] is not None

    # Смена оборудования на RTX 4090 через PATCH
    r3 = client.patch(
        "/api/v1/projects/api-test-01/hardware",
        json={
            "hardware": {
                "ram_gb": 64.0,
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "gpu_count": 1,
                "vram_gb": 24.0,
                "os": "Linux",
                "allow_cpu_offload": True,
            },
            "note": "Перенос на 24 ГБ VRAM, допускается CPU offload.",
        },
    )
    assert r3.status_code == 200
    assert r3.json()["status"] == "completed"

    # Проверка экспорта YAML, списка OSTIS и HTML-дашборда
    assert client.get("/api/v1/projects/api-test-01/export?format=yaml").status_code == 200
    assert len(client.get("/api/ostis/cases").json()) >= 1
    assert client.get("/").status_code == 200