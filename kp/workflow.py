from collections.abc import Awaitable, Callable
from typing import Any

from agent import ReflectionAgent
from agent1_data import get_agent1_data
from agent2_data import get_agent2_data
from agent3_data import get_agent3_data
from agent4_data import get_agent4_data
from schemas import AgentTraceEvent

START = "__start__"
END = "__end__"

NodeHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class DesignStateGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, NodeHandler] = {}
        self._edges: dict[str, str] = {}

    def add_node(self, node_name: str, handler: NodeHandler) -> None:
        self._nodes[node_name] = handler

    def add_edge(self, from_node: str, to_node: str) -> None:
        self._edges[from_node] = to_node

    async def invoke(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = dict(initial_state)
        state.setdefault("trace", [])
        state.setdefault("errors", [])

        current_node = START

        while current_node in self._edges:
            next_node = self._edges[current_node]
            if next_node == END:
                break

            handler = self._nodes.get(next_node)
            if handler is None:
                err_msg = f"Узел графа '{next_node}' не зарегистрирован."
                state["errors"] = list(state.get("errors", [])) + [err_msg]
                break

            prev_trace_len = len(state.get("trace", []))
            before_keys = set(state.keys())
            before_snapshot = {k: repr(v) for k, v in state.items()}

            try:
                updated_state = await handler(state)
            except Exception as exc:
                err_msg = f"Необработанная ошибка в узле '{next_node}': {exc}"
                err_event = AgentTraceEvent(
                    agent=next_node,
                    from_node=current_node,
                    to_node=next_node,
                    step="node_exception",
                    details=err_msg,
                    updated_keys=["errors"],
                    candidates_count=0,
                ).model_dump()
                state["errors"] = list(state.get("errors", [])) + [err_msg]
                state["trace"] = list(state.get("trace", [])) + [err_event]
                break

            changed_keys = sorted(
                k for k in updated_state.keys()
                if k != "trace" and (k not in before_keys or repr(updated_state[k]) != before_snapshot.get(k))
            )

            trace_list = list(updated_state.get("trace", []))
            new_events = trace_list[prev_trace_len:]
            for idx, ev in enumerate(new_events):
                ev_copy = dict(ev)
                ev_copy["from_node"] = current_node
                ev_copy["to_node"] = next_node
                if idx == len(new_events) - 1:
                    ev_copy["updated_keys"] = changed_keys
                trace_list[prev_trace_len + idx] = ev_copy

            updated_state["trace"] = trace_list
            state = updated_state
            current_node = next_node

        return state


def build_design_workflow(reflection_agent: ReflectionAgent) -> DesignStateGraph:
    async def analyze_requirements_node(state: dict[str, Any]) -> dict[str, Any]:
        scenario = str(state.get("scenario", "rtx5090"))
        data = get_agent1_data(scenario)
        next_state = dict(state)
        next_state["project_id"] = data["project_id"]
        next_state["request"] = data["request"]
        next_state["requirements"] = data["requirements"]
        next_state["hardware"] = data["hardware"]
        ev = AgentTraceEvent.model_validate(data["trace_event"]).model_dump()
        next_state["trace"] = list(state.get("trace", [])) + [ev]
        return next_state

    async def collect_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
        data = get_agent2_data()
        next_state = dict(state)
        next_state["catalog_version"] = data["catalog_version"]
        next_state["candidates"] = data["candidates"]
        ev = AgentTraceEvent.model_validate(data["trace_event"]).model_dump()
        next_state["trace"] = list(state.get("trace", [])) + [ev]
        return next_state

    async def select_model_set_node(state: dict[str, Any]) -> dict[str, Any]:
        scenario = str(state.get("scenario", "rtx5090"))
        data = get_agent3_data(scenario)
        next_state = dict(state)
        next_state["primary_set_id"] = data["primary_set_id"]
        next_state["alternative_set_id"] = data["alternative_set_id"]
        next_state["model_sets"] = data["model_sets"]
        next_state["rejected_options"] = data["rejected_options"]
        ev = AgentTraceEvent.model_validate(data["trace_event"]).model_dump()
        next_state["trace"] = list(state.get("trace", [])) + [ev]
        return next_state

    async def configure_inference_node(state: dict[str, Any]) -> dict[str, Any]:
        scenario = str(state.get("scenario", "rtx5090"))
        data = get_agent4_data(scenario)
        next_state = dict(state)
        next_state["components"] = data["components"]
        next_state["memory_breakdown"] = data["memory_breakdown"]
        next_state["decision_rationale"] = data["decision_rationale"]
        next_state["warnings"] = data["warnings"]
        ev = AgentTraceEvent.model_validate(data["trace_event"]).model_dump()
        next_state["trace"] = list(state.get("trace", [])) + [ev]
        return next_state

    async def reflect_and_store_experience_node(state: dict[str, Any]) -> dict[str, Any]:
        return await reflection_agent.reflect_and_store_experience(state)

    graph = DesignStateGraph()
    graph.add_node("analyze_requirements", analyze_requirements_node)
    graph.add_node("collect_candidates", collect_candidates_node)
    graph.add_node("select_model_set", select_model_set_node)
    graph.add_node("configure_inference", configure_inference_node)
    graph.add_node("reflect_and_store_experience", reflect_and_store_experience_node)

    graph.add_edge(START, "analyze_requirements")
    graph.add_edge("analyze_requirements", "collect_candidates")
    graph.add_edge("collect_candidates", "select_model_set")
    graph.add_edge("select_model_set", "configure_inference")
    graph.add_edge("configure_inference", "reflect_and_store_experience")
    graph.add_edge("reflect_and_store_experience", END)

    return graph