import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENT1_DIR = PROJECT_ROOT / "agent1"
if str(AGENT1_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT1_DIR))

from app.agents.requirements_agent import RequirementsAnalystAgent
from app.config import settings as agent1_settings
from app.domain.schemas import AssistantProjectRequest, HardwareProfile as Agent1Hardware
from app.infrastructure.llm_adapter import OpenAICompatibleRequirementsAdapter
from app.infrastructure.project_repo import FileProjectRepository
from app.infrastructure.rules_repo import YamlRulesRepository

from agent2.src.agent import ModelResearcherAgent
from agent2.src.schemas import RequirementProfile as Agent2Requirement
from agent3.agent3_selector import agent3_select_model_sets
from agent4.inference_agent import configure_inference as run_agent4
from agent5.agent import ReflectionAgent
from agent5.ostis_ontology import AbstractOstisGateway, LocalDictionaryOstisGateway

from .adapters import (
    adapt_for_agent3,
    adapt_for_agent4,
    adapt_for_agent5,
    normalize_trace_event,
)
from .state import SharedDesignState


def build_master_workflow(
    agent3_llm: Any = None,
    ostis_gateway: AbstractOstisGateway | None = None,
    enable_hf_background_sync: bool = False,
):
    """Собирает единый сквозной граф LangGraph: Agent 1 -> Agent 2 -> Agent 3 -> Agent 4 -> Agent 5."""
    rules_repo = YamlRulesRepository(agent1_settings.rules_path)
    project_repo = FileProjectRepository(PROJECT_ROOT / "data" / "projects")
    llm_adapter_a1 = OpenAICompatibleRequirementsAdapter(
        api_key=agent1_settings.llm_api_key,
        base_url=agent1_settings.llm_base_url,
        model=agent1_settings.llm_model,
        use_mock=agent1_settings.use_mock_llm,
        timeout_seconds=agent1_settings.llm_timeout_seconds,
    )
    agent1 = RequirementsAnalystAgent(rules_repo=rules_repo, llm_provider=llm_adapter_a1)
    agent2 = ModelResearcherAgent(enable_background_sync=enable_hf_background_sync)
    gateway = ostis_gateway or LocalDictionaryOstisGateway(
        base_dir=PROJECT_ROOT / "isolated_ostis_kb"
    )
    agent5 = ReflectionAgent(ostis_gateway=gateway)

    # --- Узел 1: Аналитик требований ---
    def node_agent1(state: SharedDesignState) -> dict[str, Any]:
        req_obj = AssistantProjectRequest.model_validate(state["request"])
        hw_obj = Agent1Hardware.model_validate(state["hardware"])
        res = agent1.invoke(
            {
                "project_id": state["project_id"],
                "request": req_obj,
                "hardware": hw_obj,
                "trace": list(state.get("trace", [])),
                "errors": list(state.get("errors", [])),
            }
        )
        return {
            "hardware": res["hardware"].model_dump(),
            "requirements": res["requirements"],
            "clarification_questions": [
                q.model_dump() for q in res.get("clarification_questions", [])
            ],
            "trace": [normalize_trace_event(ev, "RequirementsAnalyst") for ev in res["trace"]],
            "errors": res["errors"],
        }

    # --- Узел 2: Исследователь моделей (БЗ SQLite) ---
    def node_agent2(state: SharedDesignState) -> dict[str, Any]:
        req2 = Agent2Requirement.model_validate(state["requirements"])
        candidates, new_traces, is_viable = agent2.collect_candidates(
            req2, max_candidates_per_role=30
        )
        trace = list(state.get("trace", [])) + [
            normalize_trace_event(t.model_dump(), "ModelResearcher") for t in new_traces
        ]
        errors = list(state.get("errors", []))
        if not is_viable:
            errors.append("Агент 2: контур RAG не может быть собран под заданные требования.")
        return {
            "catalog_version": "2026.11.1",
            "candidates": [c.model_dump() for c in candidates],
            "is_pipeline_viable": is_viable,
            "trace": trace,
            "errors": errors,
        }

    # --- Узел 3: Подборщик комплекта моделей ---
    def node_agent3(state: SharedDesignState) -> dict[str, Any]:
        req3, hw3, candidates3 = adapt_for_agent3(state)
        try:
            res = agent3_select_model_sets(candidates3, req3, hw3, llm_client=agent3_llm)
            trace = list(state.get("trace", [])) + [
                normalize_trace_event(t, "ModelSelector") for t in res.get("trace", [])
            ]
            return {
                "model_sets": [ms.model_dump() for ms in res["model_sets"]],
                "rejected_reasons": list(state.get("rejected_reasons", []))
                + res.get("rejected_reasons", []),
                "warnings": list(state.get("warnings", [])) + res.get("warnings", []),
                "trace": trace,
            }
        except Exception as exc:
            err = f"Агент 3: {exc}"
            trace = list(state.get("trace", [])) + [
                normalize_trace_event(
                    {"agent": "ModelSelector", "step": "select_error", "details": err},
                    "ModelSelector",
                )
            ]
            return {
                "model_sets": [],
                "errors": list(state.get("errors", [])) + [err],
                "trace": trace,
            }

    # --- Узел 4: Конфигуратор совместного инференса ---
    def node_agent4(state: SharedDesignState) -> dict[str, Any]:
        payload4 = adapt_for_agent4(state)
        res4 = run_agent4(payload4)
        new_trace = [
            normalize_trace_event(ev, "InferenceConfigurator") for ev in res4.get("trace", [])
        ]
        plan = res4.get("deployment_plan")
        errors = list(state.get("errors", [])) + list(res4.get("errors", []))
        if plan and plan.get("status") != "planned":
            errors.append("Агент 4: не найдена допустимая конфигурация совместного инференса.")
        return {
            "deployment_plan": plan,
            "trace": list(state.get("trace", [])) + new_trace,
            "errors": errors,
        }

    # --- Узел 5: Рефлексивный агент и БЗ OSTIS ---
    async def node_agent5(state: SharedDesignState) -> dict[str, Any]:
        payload5 = adapt_for_agent5(state)
        res5 = await agent5.reflect_and_store_experience(payload5)
        return {
            "reflection": res5.get("reflection"),
            "scs_code": res5.get("scs_code"),
            "trace": res5.get("trace", state.get("trace", [])),
            "errors": res5.get("errors", state.get("errors", [])),
        }

    # --- Узел сохранения итогового проекта ---
    def node_save(state: SharedDesignState) -> dict[str, Any]:
        project_repo.save(state["project_id"], dict(state))
        return {}

    # --- Маршрутизаторы ---
    def route_after_a1(state: SharedDesignState) -> str:
        if state.get("errors") and not state.get("requirements"):
            return "save_project"
        if any(q.get("importance") == "critical" for q in state.get("clarification_questions", [])):
            return "save_project"
        return "collect_candidates"

    def route_after_a2(state: SharedDesignState) -> str:
        return "select_model_set" if state.get("is_pipeline_viable") else "save_project"

    def route_after_a3(state: SharedDesignState) -> str:
        return "configure_inference" if state.get("model_sets") else "save_project"

    def route_after_a4(state: SharedDesignState) -> str:
        plan = state.get("deployment_plan") or {}
        return "reflect_and_store" if plan.get("primary") else "save_project"

    builder = StateGraph(SharedDesignState)
    builder.add_node("analyze_requirements", node_agent1)
    builder.add_node("collect_candidates", node_agent2)
    builder.add_node("select_model_set", node_agent3)
    builder.add_node("configure_inference", node_agent4)
    builder.add_node("reflect_and_store", node_agent5)
    builder.add_node("save_project", node_save)

    builder.add_edge(START, "analyze_requirements")
    builder.add_conditional_edges(
        "analyze_requirements",
        route_after_a1,
        {"collect_candidates": "collect_candidates", "save_project": "save_project"},
    )
    builder.add_conditional_edges(
        "collect_candidates",
        route_after_a2,
        {"select_model_set": "select_model_set", "save_project": "save_project"},
    )
    builder.add_conditional_edges(
        "select_model_set",
        route_after_a3,
        {"configure_inference": "configure_inference", "save_project": "save_project"},
    )
    builder.add_conditional_edges(
        "configure_inference",
        route_after_a4,
        {"reflect_and_store": "reflect_and_store", "save_project": "save_project"},
    )
    builder.add_edge("reflect_and_store", "save_project")
    builder.add_edge("save_project", END)

    return builder.compile(checkpointer=MemorySaver()), gateway