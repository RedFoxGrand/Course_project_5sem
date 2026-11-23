import itertools
from typing import Any, Callable, Dict, List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

try:
    from .schemas import (
        Agent3Output,
        HardwareProfile,
        ModelCandidate,
        ModelSet,
        RequirementProfile,
    )
    from .state import DesignState
except ImportError:
    from schemas import (
        Agent3Output,
        HardwareProfile,
        ModelCandidate,
        ModelSet,
        RequirementProfile,
    )
    from state import DesignState


def estimate_vram_rough(model: ModelCandidate) -> float:
    """Предварительная оценка потребления памяти по общему числу параметров и квантованию."""
    if (
        model.parameters_count_b.status == "unknown"
        or not isinstance(model.parameters_count_b.value, (int, float))
    ):
        return 0.0

    # Для MoE и dense-моделей в память загружаются ВСЕ параметры весов
    params = float(model.parameters_count_b.value)

    if model.role == "llm":
        quants_l = {q.lower() for q in model.supported_quantizations}
        if quants_l & {"q4_k_m", "q4_0", "awq", "gptq", "int4"}:
            return round(params * 0.65, 2)
        if quants_l & {"q8_0", "int8", "fp8"}:
            return round(params * 1.15, 2)
    return round(params * 2.0, 2)


def hard_filter_models(
    candidates: List[ModelCandidate], reqs: RequirementProfile
) -> List[ModelCandidate]:
    """Программная фильтрация моделей на основе системных требований с учётом роли."""
    valid_models: List[ModelCandidate] = []
    lang_req = reqs.language.lower().strip()

    for model in candidates:
        model_langs = [lang.lower() for lang in model.languages]
        if lang_req and lang_req not in model_langs and "multilingual" not in model_langs:
            continue

        if reqs.privacy_required and not model.is_local_compatible:
            continue

        if model.role == "llm":
            target_ctx = reqs.max_context_length
        elif model.role == "embedding":
            target_ctx = min(reqs.max_context_length, 8192)
        else:
            target_ctx = min(reqs.max_context_length, 512)

        if (
            model.context_length.status != "unknown"
            and model.context_length.value is not None
            and model.context_length.value < target_ctx
        ):
            continue

        valid_models.append(model)

    return valid_models


def _has_unknown_params(combo: dict[str, Any]) -> int:
    """Возвращает 1, если у любой модели в тройке неизвестно число параметров, иначе 0."""
    return int(
        any(
            combo[r].parameters_count_b.status == "unknown"
            or combo[r].parameters_count_b.value is None
            for r in ("llm", "embedding", "reranker")
        )
    )


def _combo_quality_score(combo: dict[str, Any]) -> float:
    score = 0.0
    for role, weight in (("llm", 1.0), ("embedding", 0.5), ("reranker", 0.3)):
        m: ModelCandidate = combo[role]
        if m.domain_match:
            score += 500.0 * weight
        if m.quality_score and isinstance(m.quality_score.value, (int, float)):
            score += float(m.quality_score.value) * weight
    if _has_unknown_params(combo):
        score -= 1000.0
    return score


def _rebind_candidates(
    recommended: List[ModelSet], lookup: dict[str, ModelCandidate]
) -> List[ModelSet]:
    """Восстанавливает полные метаданные кандидатов (formats/quantizations/engines) после LLM."""
    rebound: List[ModelSet] = []
    for ms in recommended:
        llm = lookup.get(ms.llm.model_id) or lookup.get(ms.llm.name) or ms.llm
        emb = lookup.get(ms.embedding.model_id) or lookup.get(ms.embedding.name) or ms.embedding
        rerank = lookup.get(ms.reranker.model_id) or lookup.get(ms.reranker.name) or ms.reranker
        rebound.append(
            ms.model_copy(update={"llm": llm, "embedding": emb, "reranker": rerank})
        )
    return rebound


def _deterministic_select(
    shortlist: list[dict[str, Any]], hardware: HardwareProfile
) -> tuple[List[ModelSet], List[str]]:
    """Детерминированный резервный отбор (Основная по качеству + Экономичная по памяти)."""
    by_quality = sorted(shortlist, key=_combo_quality_score, reverse=True)
    primary = by_quality[0]

    by_vram = sorted(
        shortlist,
        key=lambda c: (_has_unknown_params(c), c["estimated_vram"], -_combo_quality_score(c)),
    )
    economic = next(
        (
            c
            for c in by_vram
            if (c["llm"].model_id, c["embedding"].model_id, c["reranker"].model_id)
            != (primary["llm"].model_id, primary["embedding"].model_id, primary["reranker"].model_id)
        ),
        None,
    )

    sets: List[ModelSet] = [
        ModelSet(
            set_name="Primary_Quality",
            llm=primary["llm"],
            embedding=primary["embedding"],
            reranker=primary["reranker"],
            estimated_total_vram_gb=round(primary["estimated_vram"], 2),
            rationale=(
                f"Основная конфигурация максимального качества в пределах лимита "
                f"({primary['estimated_vram']:.2f} ГБ при ресурсах VRAM={hardware.gpu_vram_gb} ГБ, "
                f"RAM={hardware.ram_gb} ГБ)."
            ),
        )
    ]

    if economic is not None:
        sets.append(
            ModelSet(
                set_name="Economic_Memory",
                llm=economic["llm"],
                embedding=economic["embedding"],
                reranker=economic["reranker"],
                estimated_total_vram_gb=round(economic["estimated_vram"], 2),
                rationale=(
                    f"Экономичная конфигурация с минимальной оценочной памятью "
                    f"({economic['estimated_vram']:.2f} ГБ) и запасом под KV-кэш."
                ),
            )
        )

    chosen_keys = {
        (s.llm.model_id, s.embedding.model_id, s.reranker.model_id) for s in sets
    }
    rejected_reasons: List[str] = []
    for idx, combo in enumerate(shortlist, 1):
        key = (combo["llm"].model_id, combo["embedding"].model_id, combo["reranker"].model_id)
        if key not in chosen_keys:
            rejected_reasons.append(
                f"Вариант {idx} ({combo['llm'].name} + {combo['embedding'].name} + {combo['reranker'].name}): "
                f"уступает выбранным комплектам по соотношению качества и памяти ({combo['estimated_vram']:.2f} ГБ)."
            )

    return sets, rejected_reasons[:10]


def agent3_select_model_sets(
    candidates: List[ModelCandidate],
    reqs: RequirementProfile,
    hardware: HardwareProfile,
    llm_client: BaseChatModel | None = None,
) -> dict[str, Any]:
    """Выполнение алгоритма подбора и ранжирования комплектов моделей."""
    filtered = hard_filter_models(candidates, reqs)

    llms = [m for m in filtered if m.role == "llm"]
    embeddings = [m for m in filtered if m.role == "embedding"]
    rerankers = [m for m in filtered if m.role == "reranker"]

    local_warnings: List[str] = []
    for model in filtered:
        if (
            model.parameters_count_b.status == "unknown"
            or not isinstance(model.parameters_count_b.value, (int, float))
        ):
            warning_msg = (
                f"Модель {model.name} имеет неизвестное количество параметров. "
                "Расчет VRAM может быть некорректным."
            )
            if warning_msg not in local_warnings:
                local_warnings.append(warning_msg)

    # При CPU-only ориентируемся на 70% RAM; при GPU — на 85% VRAM (+ небольшой запас на CPU offload embedding/reranker)
    is_cpu_only = hardware.gpu_count == 0 or hardware.gpu_vram_gb <= 0
    memory_threshold = hardware.ram_gb * 0.7 if is_cpu_only else hardware.gpu_vram_gb * 0.85

    valid_combinations: List[dict[str, Any]] = []
    for llm, emb, rerank in itertools.product(llms, embeddings, rerankers):
        llm_mem = estimate_vram_rough(llm)
        emb_mem = estimate_vram_rough(emb)
        rerank_mem = estimate_vram_rough(rerank)
        total_vram = round(llm_mem + emb_mem + rerank_mem, 2)

        # Допускаем, если помещается суммарно в VRAM (или LLM в VRAM, а мелкие модели можно вынести на CPU)
        fits_joint = total_vram <= memory_threshold
        fits_hybrid = (
            not is_cpu_only
            and llm_mem <= hardware.gpu_vram_gb * 0.85
            and (emb_mem + rerank_mem) <= hardware.ram_gb * 0.4
        )
        if fits_joint or fits_hybrid:
            valid_combinations.append(
                {
                    "llm": llm,
                    "embedding": emb,
                    "reranker": rerank,
                    "estimated_vram": total_vram,
                }
            )

    if not valid_combinations:
        raise ValueError(
            "Отсутствуют конфигурации, удовлетворяющие требованиям и ограничениям оборудования"
        )

    # Формируем сбалансированный shortlist (топ-10 по качеству + топ-5 самых лёгких)
    by_quality = sorted(valid_combinations, key=_combo_quality_score, reverse=True)
    by_vram = sorted(
        valid_combinations,
        key=lambda c: (_has_unknown_params(c), c["estimated_vram"], -_combo_quality_score(c)),
    )
    shortlist: List[dict[str, Any]] = []
    seen_combos: set[tuple[str, str, str]] = set()
    for combo in by_quality[:10] + by_vram[:5]:
        key = (combo["llm"].model_id, combo["embedding"].model_id, combo["reranker"].model_id)
        if key not in seen_combos:
            seen_combos.add(key)
            shortlist.append(combo)

    lookup: dict[str, ModelCandidate] = {}
    for m in filtered:
        lookup[m.model_id] = m
        lookup[m.name] = m

    used_llm = False
    recommended_sets: List[ModelSet] = []
    rejected_reasons: List[str] = []

    if llm_client is not None:
        combinations_text = ""
        for idx, combo in enumerate(shortlist, 1):
            combinations_text += (
                f"Вариант {idx}:\n"
                f"- LLM: {combo['llm'].name} [id={combo['llm'].model_id}] "
                f"(Параметры: {combo['llm'].parameters_count_b.value}B)\n"
                f"- Embedding: {combo['embedding'].name} [id={combo['embedding'].model_id}] "
                f"(Параметры: {combo['embedding'].parameters_count_b.value}B)\n"
                f"- Reranker: {combo['reranker'].name} [id={combo['reranker'].model_id}] "
                f"(Параметры: {combo['reranker'].parameters_count_b.value}B)\n"
                f"- Расчетная память: {combo['estimated_vram']} ГБ\n\n"
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Вы являетесь ИИ-архитектором. Проанализируйте список допустимых конфигураций "
                    "и выберите две лучшие: одну с максимальной производительностью (Основная), "
                    "вторую с минимальным потреблением памяти (Экономичная). "
                    "В массиве rejected_reasons используйте формат: "
                    "'Вариант [Номер] ([LLM] + [Embedding] + [Reranker]): [Причина]'. "
                    "Ответ должен строго соответствовать переданной схеме JSON.",
                ),
                (
                    "human",
                    "Профиль требований:\nЯзык: {language}\nДомен: {domain}\nКонтекст: {context}\n\n"
                    "Ограничение VRAM: {vram_limit} ГБ, RAM: {ram_limit} ГБ\n\n"
                    "Доступные конфигурации:\n{combinations}",
                ),
            ]
        )

        try:
            structured_llm = llm_client.with_structured_output(Agent3Output)
            chain = prompt | structured_llm
            llm_response: Agent3Output = chain.invoke(
                {
                    "language": reqs.language,
                    "domain": reqs.domain,
                    "context": reqs.max_context_length,
                    "vram_limit": hardware.gpu_vram_gb,
                    "ram_limit": hardware.ram_gb,
                    "combinations": combinations_text,
                }
            )
            recommended_sets = _rebind_candidates(llm_response.recommended_sets, lookup)
            rejected_reasons = llm_response.rejected_reasons
            used_llm = True
        except Exception as exc:
            local_warnings.append(
                f"LLM-ранжировщик Агента 3 недоступен ({exc}); применен детерминированный отбор."
            )

    if not recommended_sets:
        recommended_sets, rejected_reasons = _deterministic_select(shortlist, hardware)

    return {
        "model_sets": recommended_sets,
        "rejected_reasons": rejected_reasons,
        "warnings": local_warnings,
        "trace": [
            {
                "agent": "ModelSelector",
                "step": "rank_combinations_via_llm" if used_llm else "rank_combinations_deterministic",
                "details": (
                    f"Сформировано {len(recommended_sets)} комплектов "
                    f"({'LLM' if used_llm else 'детерминированный отбор'}) из {len(valid_combinations)} допустимых."
                ),
                "candidates_count": len(valid_combinations),
            }
        ],
    }


def create_agent3_node(
    llm_client: BaseChatModel | None = None,
) -> Callable[[DesignState], Dict[str, Any]]:
    """Формирование функции-узла для интеграции агента в граф состояний."""

    def node(state: DesignState) -> Dict[str, Any]:
        raw_reqs = state.get("requirements")
        raw_hw = state.get("hardware")
        raw_candidates = state.get("candidates", [])

        if not raw_reqs or not raw_hw:
            return {
                "errors": list(state.get("errors", []))
                + ["Отсутствуют требования или профиль оборудования"]
            }

        try:
            reqs = (
                raw_reqs
                if isinstance(raw_reqs, RequirementProfile)
                else RequirementProfile.model_validate(raw_reqs)
            )
            hardware = (
                raw_hw
                if isinstance(raw_hw, HardwareProfile)
                else HardwareProfile.model_validate(raw_hw)
            )
            candidates = [
                c if isinstance(c, ModelCandidate) else ModelCandidate.model_validate(c)
                for c in raw_candidates
            ]

            result = agent3_select_model_sets(candidates, reqs, hardware, llm_client)
            return {
                "model_sets": result["model_sets"],
                "rejected_reasons": list(state.get("rejected_reasons", []))
                + result.get("rejected_reasons", []),
                "warnings": list(state.get("warnings", [])) + result.get("warnings", []),
                "trace": list(state.get("trace", [])) + result["trace"],
            }
        except Exception as e:
            return {"errors": list(state.get("errors", [])) + [str(e)]}

    return node