import hashlib
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import ValidationError

try:
    from .ostis_ontology import AbstractOstisGateway, ScsOntologyGenerator
    from .schemas import (
        AgentTraceEvent,
        ComponentRunConfig,
        EpistemicStatus,
        ExperienceCategoryOutput,
        HardwareInput,
        MemoryBreakdownInput,
        ProvenanceValue,
        ReflectionRecord,
        RejectedOptionInput,
        TaskRequirementInput,
        UncertaintyStatement,
    )
except ImportError:
    from ostis_ontology import AbstractOstisGateway, ScsOntologyGenerator
    from schemas import (
        AgentTraceEvent,
        ComponentRunConfig,
        EpistemicStatus,
        ExperienceCategoryOutput,
        HardwareInput,
        MemoryBreakdownInput,
        ProvenanceValue,
        ReflectionRecord,
        RejectedOptionInput,
        TaskRequirementInput,
        UncertaintyStatement,
    )


def _canonical_dict_repr(data: Any) -> str:
    if isinstance(data, dict):
        items = ",".join(f"{k}:{_canonical_dict_repr(data[k])}" for k in sorted(data.keys()))
        return "{" + items + "}"
    if isinstance(data, (list, tuple)):
        return "[" + ",".join(_canonical_dict_repr(x) for x in data) + "]"
    return repr(data)


class AbstractCategoryLLM(ABC):
    @abstractmethod
    async def generate_category(
        self,
        req: TaskRequirementInput,
        hw: HardwareInput,
        components: list[ComponentRunConfig],
    ) -> dict[str, Any]:
        pass


class DeterministicLocalCategorizer(AbstractCategoryLLM):
    async def generate_category(
        self,
        req: TaskRequirementInput,
        hw: HardwareInput,
        components: list[ComponentRunConfig],
    ) -> dict[str, Any]:
        has_cpu_offload = any(c.placement == "cpu" for c in components)
        suffix = "_cpu_offload" if has_cpu_offload else "_all_gpu"
        hw_cat = f"{hw.gpu_count}xgpu_{int(hw.vram_gb)}gb{suffix}"

        category = ExperienceCategoryOutput(
            normalized_summary=(
                f"Локальный {req.language.upper()} RAG-контур ({req.domain}) "
                f"на {hw.gpu_model} ({int(hw.vram_gb)} ГБ VRAM, {int(hw.ram_gb)} ГБ RAM)"
            ),
            task_class=f"rag_{req.language}_{req.domain}",
            language_category=req.language,
            domain_category=req.domain,
            hardware_category=hw_cat,
            priority_category=req.priority,
        )
        return category.model_dump()


class OpenAICompatibleCategorizer(AbstractCategoryLLM):
    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    async def generate_category(
        self,
        req: TaskRequirementInput,
        hw: HardwareInput,
        components: list[ComponentRunConfig],
    ) -> dict[str, Any]:
        prompt = (
            f"Сформируй структуру категории опыта проектирования RAG-ассистента.\n"
            f"Язык: {req.language}, Домен: {req.domain}, Контекст: {req.max_context_length}.\n"
            f"Оборудование: {hw.gpu_model} ({hw.vram_gb} ГБ VRAM, {hw.ram_gb} ГБ RAM).\n"
            f"Схема: {ExperienceCategoryOutput.model_json_schema()}"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            validated = ExperienceCategoryOutput.model_validate_json(content)
            return validated.model_dump()


class ReflectionAgent:
    def __init__(
        self,
        ostis_gateway: AbstractOstisGateway,
        llm_categorizer: AbstractCategoryLLM | None = None,
    ) -> None:
        self.ostis = ostis_gateway
        self.categorizer = llm_categorizer or DeterministicLocalCategorizer()
        self._fallback_categorizer = DeterministicLocalCategorizer()

    @staticmethod
    def _compute_dedup_hash(
        req: TaskRequirementInput,
        hw: HardwareInput,
        components: list[ComponentRunConfig],
    ) -> str:
        comp_sig = sorted(
            f"{c.role}:{c.model_id}:{c.engine}:{c.quantization}:{c.placement}:{c.max_model_len}:{c.max_num_seqs}"
            for c in components
        )
        canonical_dict = {
            "lang": req.language.lower(),
            "domain": req.domain.lower(),
            "ctx": req.max_context_length,
            "hw": hw.fingerprint,
            "components": comp_sig,
        }
        canonical_str = _canonical_dict_repr(canonical_dict)
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @staticmethod
    def _analyze_uncertainties(
        mem: MemoryBreakdownInput,
        components: list[ComponentRunConfig],
        warnings: list[str],
    ) -> tuple[list[UncertaintyStatement], EpistemicStatus]:
        checks: list[tuple[str, str, ProvenanceValue]] = [
            ("llm_weights", "Память весов LLM", mem.llm_weights),
            ("llm_kv_cache", "Память KV cache LLM", mem.llm_kv_cache),
            ("embedding_mem", "Память embedding-модели", mem.embedding_resident_memory),
            ("reranker_mem", "Память reranker-модели", mem.reranker_resident_memory),
            ("runtime_overhead", "Служебная память движка инференса", mem.runtime_overhead),
            ("safety_reserve", "Запас устойчивости по памяти", mem.safety_reserve),
        ]

        statements: list[UncertaintyStatement] = []
        overall_status: EpistemicStatus = "confirmed_fact"

        for code, title, item in checks:
            if item.status == "unknown":
                ep_status: EpistemicStatus = "requires_verification"
                overall_status = "requires_verification"
                desc = f"{title}: точные данные отсутствуют, запуск требует проверки (requires_verification)."
            elif item.status == "estimated":
                ep_status = "hypothesis"
                if overall_status == "confirmed_fact":
                    overall_status = "hypothesis"
                desc = f"{title}: аналитическая оценка ({item.value} ГБ) записана как гипотеза."
            else:
                ep_status = "confirmed_fact"
                desc = f"{title}: подтверждённый факт ({item.value} ГБ)."

            statements.append(
                UncertaintyStatement(
                    statement_id=code,
                    parameter_name=title,
                    description=desc,
                    source=item.source,
                    provenance_status=item.status,
                    epistemic_status=ep_status,
                )
            )

        required_roles = {"llm", "embedding", "reranker"}
        present_roles = {c.role for c in components}
        missing_roles = sorted(required_roles - present_roles)
        if missing_roles:
            overall_status = "requires_verification"
            statements.append(
                UncertaintyStatement(
                    statement_id="missing_mvp_roles",
                    parameter_name="Состав комплекта RAG",
                    description=f"Отсутствуют обязательные роли MVP: {', '.join(missing_roles)}.",
                    source="reflection_validator",
                    provenance_status="unknown",
                    epistemic_status="requires_verification",
                )
            )

        for idx, comp in enumerate(components):
            if not comp.source_url:
                if overall_status == "confirmed_fact":
                    overall_status = "hypothesis"
                statements.append(
                    UncertaintyStatement(
                        statement_id=f"missing_url_{idx}",
                        parameter_name=f"Источник модели {comp.model_id}",
                        description=f"Для модели {comp.model_id} не указан корректный URL первоисточника.",
                        source="catalog_parser",
                        provenance_status="unknown",
                        epistemic_status="hypothesis",
                    )
                )

        for idx, warn_text in enumerate(warnings):
            if overall_status == "confirmed_fact":
                overall_status = "hypothesis"
            statements.append(
                UncertaintyStatement(
                    statement_id=f"warning_{idx}",
                    parameter_name="Предупреждение верификации",
                    description=warn_text,
                    source="pipeline_validator",
                    provenance_status="estimated",
                    epistemic_status="hypothesis",
                )
            )

        return statements, overall_status

    async def reflect_and_store_experience(self, state: dict[str, Any]) -> dict[str, Any]:
        next_state = dict(state)
        errors: list[str] = list(state.get("errors", []))
        trace_list: list[dict[str, Any]] = []

        for ev in state.get("trace", []):
            try:
                trace_list.append(AgentTraceEvent.model_validate(ev).model_dump())
            except ValidationError:
                continue

        required_keys = ("requirements", "hardware", "components", "memory_breakdown")
        missing_keys = [k for k in required_keys if not state.get(k)]
        if missing_keys:
            msg = f"Недостаточно данных в DesignState для рефлексии: отсутствуют {', '.join(missing_keys)}."
            errors.append(msg)
            trace_list.append(
                AgentTraceEvent(
                    agent="ReflectionAgent",
                    step="validate_input_state",
                    details=msg,
                    candidates_count=0,
                ).model_dump()
            )
            next_state["errors"] = errors
            next_state["trace"] = trace_list
            next_state["reflection"] = None
            return next_state

        project_id = str(state.get("project_id", "prj_unknown"))
        catalog_version = str(state.get("catalog_version", "2026.10.1"))

        try:
            req = TaskRequirementInput.model_validate(state["requirements"])
            hw = HardwareInput.model_validate(state["hardware"])
            components = [
                ComponentRunConfig.model_validate(c) for c in state["components"]
            ]
            memory = MemoryBreakdownInput.model_validate(state["memory_breakdown"])
        except (ValidationError, TypeError) as exc:
            msg = f"Ошибка валидации входного состояния DesignState: {exc}"
            errors.append(msg)
            trace_list.append(
                AgentTraceEvent(
                    agent="ReflectionAgent",
                    step="validation_error",
                    details=msg,
                    candidates_count=0,
                ).model_dump()
            )
            next_state["errors"] = errors
            next_state["trace"] = trace_list
            next_state["reflection"] = None
            return next_state

        rejected: list[RejectedOptionInput] = []
        for raw_rej in state.get("rejected_options", []):
            try:
                rejected.append(RejectedOptionInput.model_validate(raw_rej))
            except ValidationError as exc:
                errors.append(f"Пропущена некорректная запись об отклонении: {exc}")

        warnings = [str(w) for w in state.get("warnings", []) if w is not None and str(w).strip()]
        dedup_hash = self._compute_dedup_hash(req, hw, components)

        try:
            existing_record = await self.ostis.get_by_hash(dedup_hash)
        except Exception as exc:
            existing_record = None
            errors.append(f"Ошибка проверки дубликата в OSTIS: {exc}")

        if existing_record is not None:
            trace_list.append(
                AgentTraceEvent(
                    agent="ReflectionAgent",
                    step="prevent_duplication",
                    details=(
                        f"Случай проектирования с хэшем {dedup_hash[:10]}... уже существует в БЗ OSTIS "
                        f"({existing_record.get('reflection_id', 'unknown')}). Дублирование предотвращено."
                    ),
                    candidates_count=len(components),
                ).model_dump()
            )
            next_state["reflection"] = existing_record
            next_state["errors"] = errors
            next_state["trace"] = trace_list
            return next_state

        try:
            similar_case = await self.ostis.find_similar(req.language, req.domain)
            similar_project_id = str(similar_case["project_id"]) if similar_case else None
        except Exception as exc:
            similar_project_id = None
            errors.append(f"Ошибка поиска похожих случаев в OSTIS: {exc}")

        try:
            category_dict = await self.categorizer.generate_category(req, hw, components)
            category = ExperienceCategoryOutput.model_validate(category_dict)
        except Exception as exc:
            errors.append(f"Сбой категоризатора LLM, применен локальный резерв: {exc}")
            fallback_dict = await self._fallback_categorizer.generate_category(req, hw, components)
            category = ExperienceCategoryOutput.model_validate(fallback_dict)

        trace_list.append(
            AgentTraceEvent(
                agent="ReflectionAgent",
                step="categorize_solution",
                details=f"Категоризация завершена: {category.task_class} | {category.hardware_category}",
                candidates_count=len(components),
            ).model_dump()
        )

        uncertainties, overall_epistemic = self._analyze_uncertainties(memory, components, warnings)
        sources = sorted({c.source_url for c in components if c.source_url})

        record = ReflectionRecord(
            reflection_id=f"refl_{project_id}",
            project_id=project_id,
            catalog_version=catalog_version,
            deduplication_hash=dedup_hash,
            similar_project_id=similar_project_id,
            category=category,
            requirements=req,
            hardware=hw,
            primary_set_id=str(state.get("primary_set_id", "set_primary")),
            alternative_set_id=state.get("alternative_set_id"),
            components=components,
            memory_breakdown=memory,
            decision_rationale=str(state.get("decision_rationale", "Выбрана оптимальная конфигурация.")),
            rejected_options=rejected,
            uncertainties=uncertainties,
            warnings=warnings,
            sources=sources,
            overall_epistemic_status=overall_epistemic,
        )

        record_dict = record.model_dump()
        scs_code = ScsOntologyGenerator.generate(record)

        try:
            sc_addr = await self.ostis.store(record_dict, scs_code)
            hypotheses_count = sum(1 for u in uncertainties if u.epistemic_status != "confirmed_fact")
            trace_list.append(
                AgentTraceEvent(
                    agent="ReflectionAgent",
                    step="store_ostis_experience",
                    details=(
                        f"Запись сохранена в БЗ OSTIS ({sc_addr}). "
                        f"Статус: {overall_epistemic}, гипотез зафиксировано: {hypotheses_count}."
                    ),
                    candidates_count=len(components),
                ).model_dump()
            )
        except Exception as exc:
            msg = f"Не удалось сохранить опыт в БЗ OSTIS: {exc}"
            errors.append(msg)
            trace_list.append(
                AgentTraceEvent(
                    agent="ReflectionAgent",
                    step="store_ostis_error",
                    details=msg,
                    candidates_count=len(components),
                ).model_dump()
            )

        next_state["reflection"] = record_dict
        next_state["scs_code"] = scs_code
        next_state["errors"] = errors
        next_state["trace"] = trace_list
        return next_state