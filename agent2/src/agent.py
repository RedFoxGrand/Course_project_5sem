from typing import Any

from .catalog_adapter import CatalogAdapter
from .hf_adapter import BackgroundSyncScheduler, HuggingFaceSyncService
from .kb_storage import UnifiedKBStorage
from .schemas import AgentTraceEvent, ModelCandidate, RequirementProfile, RoleType


class ModelResearcherAgent:
    """Агент-исследователь моделей: мгновенный поиск по актуальной БЗ с фоновой синхронизацией."""

    def __init__(
        self,
        db_path: str | None = None,
        enable_background_sync: bool = True,
        sync_interval_hours: float = 24.0,
    ):
        self.storage = UnifiedKBStorage(db_path) if db_path else UnifiedKBStorage()
        self.catalog_tool = CatalogAdapter()

        self.storage.upsert_models(self.catalog_tool.get_all_catalog_models())

        self.sync_service = HuggingFaceSyncService(storage=self.storage)
        self.scheduler = BackgroundSyncScheduler(
            sync_service=self.sync_service,
            interval_hours=sync_interval_hours,
        )
        if enable_background_sync:
            self.scheduler.start()

    def collect_candidates(
        self, requirements: RequirementProfile, max_candidates_per_role: int = 10
    ) -> tuple[list[ModelCandidate], list[AgentTraceEvent], bool]:
        all_candidates: list[ModelCandidate] = []
        traces: list[AgentTraceEvent] = []
        seen_ids: set[str] = set()

        roles: list[RoleType] = ["llm", "embedding", "reranker"]
        role_counts: dict[str, int] = {"llm": 0, "embedding": 0, "reranker": 0}

        sync_meta = self.storage.get_sync_metadata()
        last_sync_str = (
            str(sync_meta["last_sync_at"])
            if sync_meta and sync_meta.get("last_sync_at")
            else "ожидается фоновая синхронизация (используется базовый каталог)"
        )
        total_kb_models = sum(self.storage.count_by_role().values())
        traces.append(
            AgentTraceEvent(
                step="kb_status_check",
                details=f"Единая локальная БЗ активна (всего моделей: {total_kb_models}; обновление HF Hub: {last_sync_str})",
                candidates_count=total_kb_models,
            )
        )

        for role in roles:
            matched_models = self.storage.query_models(role=role, req=requirements, strict_context=True)
            added_count = 0

            for m in matched_models:
                if m.model_id not in seen_ids:
                    all_candidates.append(m)
                    seen_ids.add(m.model_id)
                    added_count += 1
                    if added_count >= max_candidates_per_role:
                        break

            role_counts[role] = added_count
            target_ctx = (
                requirements.max_context_length
                if role == "llm"
                else min(requirements.max_context_length, 8192 if role == "embedding" else 512)
            )
            traces.append(
                AgentTraceEvent(
                    step=f"kb_search:{role}",
                    details=(
                        f"БЗ: для роли '{role}' отобрано {added_count} кандидатов "
                        f"(lang={requirements.language}, min_ctx={target_ctx})"
                    ),
                    candidates_count=added_count,
                )
            )

        missing_roles = [r for r, count in role_counts.items() if count == 0]
        is_pipeline_viable = len(missing_roles) == 0

        if not is_pipeline_viable:
            reasons = []
            for r in missing_roles:
                if r == "llm":
                    reasons.append(
                        f"нет LLM с контекстом >= {requirements.max_context_length} токенов под язык '{requirements.language}'"
                    )
                elif r == "embedding":
                    reasons.append(f"нет Embedding под язык '{requirements.language}'")
                elif r == "reranker":
                    reasons.append(f"нет Reranker под язык '{requirements.language}'")

            error_details = (
                f"[Ошибка контура] Тройка RAG не может быть собрана. Отсутствуют роли: {missing_roles}. "
                f"Причины: {'; '.join(reasons)}."
            )
            traces.append(
                AgentTraceEvent(
                    step="pipeline_integrity_check",
                    details=error_details,
                    candidates_count=0,
                )
            )
        else:
            traces.append(
                AgentTraceEvent(
                    step="pipeline_integrity_check",
                    details=(
                        f"Контур RAG жизнеспособен! Отобрано из актуальной БЗ: "
                        f"LLM={role_counts['llm']}, Emb={role_counts['embedding']}, Rerank={role_counts['reranker']}."
                    ),
                    candidates_count=len(all_candidates),
                )
            )

        return all_candidates, traces, is_pipeline_viable


_DEFAULT_AGENT: ModelResearcherAgent | None = None


async def collect_candidates_node(state: dict[str, Any]) -> dict[str, Any]:
    """Узел для графа LangGraph с поддержкой dict-контракта и настраиваемого лимита."""
    global _DEFAULT_AGENT
    if _DEFAULT_AGENT is None:
        enable_sync = bool(state.get("enable_background_sync", True))
        _DEFAULT_AGENT = ModelResearcherAgent(enable_background_sync=enable_sync)

    raw_req = state.get("requirements") or {}
    req_profile = (
        raw_req
        if isinstance(raw_req, RequirementProfile)
        else (
            RequirementProfile.model_validate(raw_req)
            if isinstance(raw_req, dict)
            else RequirementProfile()
        )
    )

    max_per_role = int(state.get("max_candidates_per_role", 25))
    candidates, new_traces, is_pipeline_viable = _DEFAULT_AGENT.collect_candidates(
        req_profile, max_candidates_per_role=max_per_role
    )

    existing_trace = list(state.get("trace", []))
    existing_trace.extend([t.model_dump() for t in new_traces])

    return {
        "candidates": [c.model_dump() for c in candidates],
        "trace": existing_trace,
        "is_pipeline_viable": is_pipeline_viable,
    }