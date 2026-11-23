from typing import Any, Callable, Optional
from pydantic import BaseModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from app.agents.requirements_agent import RequirementsAnalystAgent
from app.domain.interfaces import IProjectRepository
from app.domain.schemas import AgentTraceEvent
from app.workflow.state import DesignState


def serialize_state_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [serialize_state_value(item) for item in value]
    if isinstance(value, dict):
        return {k: serialize_state_value(v) for k, v in value.items()}
    return value


def build_agent1_workflow(
    analyst_agent: RequirementsAnalystAgent,
    project_repo: IProjectRepository,
    collect_candidates_node: Optional[Callable[[dict[str, Any]], Any]] = None,
):
    checkpointer = MemorySaver()
    builder = StateGraph(DesignState)

    def analyze_requirements_node(state: DesignState) -> dict:
        return analyst_agent.invoke(state)

    def clarify_if_needed_node(state: DesignState) -> dict:
        trace = list(state.get("trace", []))
        trace.append(
            AgentTraceEvent(
                agent="WorkflowRouter",
                step="clarify_if_needed",
                details="Ожидание ответов пользователя на уточняющие вопросы.",
                status="needs_clarification",
                applied_rules=["LANGGRAPH_INTERRUPT_CLARIFICATION"],
            ).model_dump()
        )
        return {"trace": trace}

    def save_project_node(state: DesignState) -> dict:
        trace = list(state.get("trace", []))
        errors = list(state.get("errors", []))

        if state.get("is_pipeline_viable") is False:
            viability_err = "Агент 2: контур RAG не может быть собран под заданные требования."
            if viability_err not in errors:
                errors.append(viability_err)

        trace.append(
            AgentTraceEvent(
                agent="ProjectStorage",
                step="save_project",
                details=f"Состояние проекта {state['project_id']} сохранено.",
                status="completed",
                applied_rules=["PERSIST_PROJECT_STATE"],
            ).model_dump()
        )
        serializable = {
            "project_id": state["project_id"],
            "request": serialize_state_value(state["request"]),
            "hardware": serialize_state_value(state["hardware"]),
            "requirements": serialize_state_value(state.get("requirements")),
            "clarification_questions": serialize_state_value(
                state.get("clarification_questions", [])
            ),
            "candidates": serialize_state_value(state.get("candidates", [])),
            "is_pipeline_viable": state.get("is_pipeline_viable"),
            "model_sets": serialize_state_value(state.get("model_sets", [])),
            "deployment_plan": serialize_state_value(state.get("deployment_plan")),
            "reflection": serialize_state_value(state.get("reflection")),
            "trace": serialize_state_value(trace),
            "errors": errors,
        }
        project_repo.save(state["project_id"], serializable)
        return {"trace": trace, "errors": errors}

    def route_after_analysis(state: DesignState) -> str:
        if state.get("errors") and not state.get("requirements"):
            return "error_end"
        questions = state.get("clarification_questions", [])
        if any(q.importance == "critical" for q in questions):
            return "clarify_if_needed"
        if collect_candidates_node is not None:
            return "collect_candidates"
        return "save_project"

    builder.add_node("analyze_requirements", analyze_requirements_node)
    builder.add_node("clarify_if_needed", clarify_if_needed_node)
    builder.add_node("save_project", save_project_node)

    routes: dict[str, Any] = {
        "clarify_if_needed": "clarify_if_needed",
        "save_project": "save_project",
        "error_end": END,
    }
    if collect_candidates_node is not None:
        builder.add_node("collect_candidates", collect_candidates_node)
        routes["collect_candidates"] = "collect_candidates"
        builder.add_edge("collect_candidates", "save_project")

    builder.add_edge(START, "analyze_requirements")
    builder.add_conditional_edges(
        "analyze_requirements",
        route_after_analysis,
        routes,
    )
    builder.add_edge("clarify_if_needed", END)
    builder.add_edge("save_project", END)

    return builder.compile(checkpointer=checkpointer)
