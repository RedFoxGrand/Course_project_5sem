import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field
from app.api.main import app, analyst_agent, project_repo
from app.domain.schemas import EvidenceItem, ModelCandidate, RequirementProfile
from app.workflow.graph import build_agent1_workflow

client = TestClient(app)


def test_typical_scenario_1_rtx_5090():
    payload = {
        "project_id": "test-rtx-5090",
        "request": {
            "task_description": (
                "Требуется локальный русскоязычный RAG-ассистент по внутренней технической "
                "документации. Одновременно работают LLM, embedding и reranker. Данные "
                "нельзя передавать во внешние API. Приоритет качество ответа при сохранении "
                "интерактивной скорости. Ожидается до пяти параллельных пользовательских сессий."
            )
        },
        "hardware": {
            "cpu": "AMD Ryzen 9",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "os": "Linux",
        },
    }
    response = client.post("/api/v1/projects/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["clarification_questions"] == []

    req_profile = RequirementProfile(**data["requirements"])
    assert req_profile.language == "ru"
    assert req_profile.domain == "technical"
    assert req_profile.privacy_required is True
    assert req_profile.max_context_length == 8192
    assert req_profile.parallel_sessions == 5
    assert req_profile.priority_quality == 5
    assert data["agent2_input"] == {
        "language": "ru",
        "domain": "technical",
        "max_context_length": 8192,
        "privacy_required": True,
    }


def test_typical_scenario_2_reuse_project_on_rtx_4090():
    analyze_payload = {
        "project_id": "test-transfer-5090-to-4090",
        "request": {
            "task_description": (
                "Требуется локальный русскоязычный RAG-ассистент по внутренней технической "
                "документации. Данные нельзя передавать во внешние API. "
                "Ожидается до пяти параллельных пользовательских сессий."
            )
        },
        "hardware": {
            "cpu": "AMD Ryzen 9",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "os": "Linux",
            "allow_cpu_offload": False,
        },
    }
    res1 = client.post("/api/v1/projects/analyze", json=analyze_payload)
    assert res1.status_code == 200
    assert res1.json()["requirements"]["parallel_sessions"] == 5

    patch_payload = {
        "hardware": {
            "cpu": "Intel Core i9",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "vram_gb": 24.0,
            "os": "Linux",
            "allow_cpu_offload": True,
        },
        "note": "Допускается снижение параллелизма и использование CPU для embedding или reranker.",
    }
    res2 = client.patch("/api/v1/projects/test-transfer-5090-to-4090/hardware", json=patch_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "completed"
    assert data2["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert data2["hardware"]["allow_cpu_offload"] is True
    assert data2["requirements"]["allow_cpu_offload"] is True
    assert data2["requirements"]["parallel_sessions"] < 5
    assert any(t["step"] == "update_hardware" for t in data2["trace"])


def test_boundary_clarification_loop_and_vram_answer_sync():
    payload = {
        "project_id": "test-clarify-loop",
        "request": {
            "task_description": "Нужен корпоративный ассистент для поиска по базе файлов компании."
        },
        "hardware": {
            "cpu": "x86_64",
            "ram_gb": 32.0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "vram_gb": 0.0,
            "os": "Linux",
        },
    }
    res1 = client.post("/api/v1/projects/analyze", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "needs_clarification"
    missing_fields = {q["field_name"] for q in data1["clarification_questions"]}
    assert {"language", "domain", "vram_gb"}.issubset(missing_fields)

    clarify_payload = {
        "answers": {
            "language": "ru",
            "domain": "technical",
            "vram_gb": "24",
        }
    }
    res2 = client.post("/api/v1/projects/test-clarify-loop/clarify", json=clarify_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["status"] == "completed"
    assert data2["hardware"]["vram_gb"] == 24.0
    assert data2["requirements"]["language"] == "ru"
    assert data2["requirements"]["domain"] == "technical"


def test_cpu_negation_not_mistaken_for_offload():
    payload = {
        "project_id": "test-no-cpu",
        "request": {
            "task_description": (
                "Нужен локальный русскоязычный RAG-ассистент по технической документации, "
                "запуск только GPU без CPU."
            )
        },
        "hardware": {
            "cpu": "x86_64",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "os": "Linux",
            "allow_cpu_offload": False,
        },
    }
    res = client.post("/api/v1/projects/analyze", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["requirements"]["allow_cpu_offload"] is False
    assert data["hardware"]["allow_cpu_offload"] is False


def test_export_yaml_and_no_html_ui():
    assert client.get("/").status_code == 404
    res_yaml = client.get("/api/v1/projects/test-rtx-5090/export?format=yaml")
    assert res_yaml.status_code == 200
    assert "project_id: test-rtx-5090" in res_yaml.text


def test_agent2_schema_compatibility():
    candidate = ModelCandidate(
        model_id="Qwen/Qwen2.5-14B-Instruct",
        name="Qwen2.5-14B-Instruct",
        organization="Qwen",
        role="llm",
        license="apache-2.0",
        languages=["ru", "en"],
        context_length=EvidenceItem(
            value=32768,
            source="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
            status="declared",
        ),
        parameters_count_b=EvidenceItem(
            value=14.7,
            source="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
            status="declared",
        ),
        is_context_sufficient=True,
        supported_quantizations=["Q4_K_M", "FP16"],
        compatible_engines=["vllm", "ollama"],
        domain_match=True,
        source_url="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct",
    )
    assert candidate.context_length.value == 32768
    assert candidate.parameters_count_b.value == 14.7


@pytest.mark.asyncio
async def test_agent1_to_agent2_async_handoff():
    class Agent2RequirementProfile(BaseModel):
        model_config = ConfigDict(extra="ignore")
        language: str = "ru"
        domain: str = "general"
        max_context_length: int = Field(default=8192, gt=0)
        privacy_required: bool = True

    class Agent2TraceEvent(BaseModel):
        agent: str = "ModelResearcher"
        step: str
        details: str
        candidates_count: int = 0

    received = {}

    async def mock_agent2_collect_candidates_node(state: dict) -> dict:
        raw_req = state.get("requirements") or {}
        req_profile = (
            raw_req
            if isinstance(raw_req, Agent2RequirementProfile)
            else (
                Agent2RequirementProfile.model_validate(raw_req)
                if isinstance(raw_req, dict)
                else Agent2RequirementProfile()
            )
        )
        received["profile"] = req_profile
        existing_trace = list(state.get("trace", []))
        existing_trace.append(
            Agent2TraceEvent(
                step="pipeline_integrity_check",
                details="Контур RAG жизнеспособен!",
                candidates_count=3,
            ).model_dump()
        )
        return {
            "candidates": [],
            "trace": existing_trace,
            "is_pipeline_viable": True,
        }

    integrated_workflow = build_agent1_workflow(
        analyst_agent=analyst_agent,
        project_repo=project_repo,
        collect_candidates_node=mock_agent2_collect_candidates_node,
    )
    from app.domain.schemas import AssistantProjectRequest, HardwareProfile

    result = await integrated_workflow.ainvoke(
        {
            "project_id": "test-handoff-a2",
            "request": AssistantProjectRequest(
                task_description="Нужен локальный англоязычный ассистент для разработки кода с контекстом 32k."
            ),
            "hardware": HardwareProfile(
                ram_gb=64.0,
                gpu_name="NVIDIA GeForce RTX 5090",
                gpu_count=1,
                vram_gb=32.0,
            ),
            "requirements": None,
            "clarification_questions": [],
            "candidates": [],
            "is_pipeline_viable": None,
            "model_sets": [],
            "deployment_plan": None,
            "reflection": None,
            "trace": [],
            "errors": [],
        },
        config={"configurable": {"thread_id": "test-handoff-a2"}},
    )
    assert received["profile"].language == "en"
    assert received["profile"].domain == "code"
    assert received["profile"].max_context_length == 32768
    assert received["profile"].privacy_required is True
    assert result["is_pipeline_viable"] is True
    assert any(t["agent"] == "ModelResearcher" for t in result["trace"])
