"""Agent 4. Public interface: dict -> dict, including LangGraph node.

Only deterministic and validated feasible strategies can reach the output.
LLM ranking is optional and cannot override compatibility/memory hard checks.
"""
from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from math import ceil
from typing import Any, Callable
from pydantic import BaseModel, ValidationError
from .contracts import DeploymentPlan, Hardware, Requirements, ModelSet, Option
from .rules import (BITS, ENGINE_DOCS, choose_engine, engine_choices, make_config,
                    normalize_hardware, normalize_requirements, normalize_sets)
from .memory import weight_estimate, kv_estimate, calculate_memory

class RankedDecision(BaseModel):
    strategy_id: str
    explanation: str = ""


def levels(value: int, minimum: int, allow_reduce: bool) -> list[int]:
    return list(dict.fromkeys([value, max(minimum, value // 2), minimum])) if allow_reduce else [value]


def schemes(has_gpu: bool, offload: bool) -> list[tuple[str, str, str]]:
    if not has_gpu:
        return [("cpu", "cpu", "cpu")]
    result = [("gpu", "gpu", "gpu")]
    if offload:
        result += [("gpu", "gpu", "cpu"), ("gpu", "cpu", "gpu"),
                   ("gpu", "cpu", "cpu"), ("cpu", "cpu", "cpu")]
    return result


class InferenceConfigurator:
    def __init__(self, ranker: Callable[[dict], dict] | None = None):
        """ranker: optional structured LLM callback returning a plain dictionary."""
        self.ranker = ranker

    def build_option(self, ms: ModelSet, hw: Hardware, req: Requirements,
                     quant: str, devices: tuple[str, str, str],
                     context: int, parallel: int, llm_engine: str) -> tuple[Option | None, str]:
        models = {"llm": ms.llm, "embedding": ms.embedding, "reranker": ms.reranker}
        engine = {"llm": llm_engine,
                  "embedding": choose_engine(ms.embedding, "embedding"),
                  "reranker": choose_engine(ms.reranker, "reranker")}
        if llm_engine not in engine_choices(ms.llm, "llm"):
            return None, "Недопустимый inference engine для LLM"
        for role, model in models.items():
            if req.local_only and model.local_available is False:
                return None, f"Модель {role} запрещает локальное развёртывание"
            if req.license_allowlist and model.license is not None and (
                    model.license.lower() not in [v.lower() for v in req.license_allowlist]):
                return None, f"Лицензия {model.license} ({role}) не входит в разрешённый список"
            if req.language != "unspecified" and model.languages:
                allowed = [lang.lower() for lang in model.languages]
                language = req.language.lower()
                if not (language in allowed or "multilingual" in allowed or "multi" in allowed):
                    return None, f"Модель {role} не поддерживает язык {req.language}"
        unknown = [role for role, value in engine.items() if value is None]
        if unknown:
            return None, "Не подтверждён engine или формат весов: " + ", ".join(unknown)
        if any(models[role].quantizations[0] not in {"fp16", "float16", "bf16"}
               for role in ("embedding", "reranker")):
            return None, "Для TEI в MVP поддержаны только подтверждённые fp16/bf16-веса"
        if quant not in ms.llm.quantizations or quant not in BITS:
            return None, "Квантизация отсутствует в подтверждённой карточке модели"
        if ms.llm.max_context is not None and context > ms.llm.max_context:
            return None, f"Контекст {context} больше max_context={ms.llm.max_context}"
        if engine["llm"] == "vllm" and quant not in {"fp16", "float16", "awq", "gptq", "fp8"}:
            return None, "Генератор vLLM не поддерживает данный способ квантования"
        if engine["llm"] in {"llama.cpp", "ollama"} and quant in {"awq", "gptq", "fp8"}:
            return None, "Формат AWQ/GPTQ/FP8 не является готовым GGUF"
        if devices[0] == "cpu" and engine["llm"] not in {"llama.cpp", "ollama"}:
            return None, "CPU-LLM доступен только для llama.cpp/Ollama"
        if engine["llm"] == "vllm" and devices[0] != "gpu":
            return None, "vLLM GPU-only в данной реализации"
        if engine["llm"] == "vllm" and "linux" not in hw.os.lower():
            return None, "vLLM предполагает Linux; другая ОС требует отдельного подтверждения"
        if "win" in hw.os.lower() and any(e == "tei" for e in engine.values()):
            return None, "TEI на Windows требует Linux/WSL; native deployment не поддержан планировщиком"
        vram = max((gpu.vram_gb for gpu in hw.gpus), default=0)
        fraction = None
        if engine["llm"] == "vllm":
            weight = weight_estimate(ms.llm, quant, "llm").value_gb
            kv = kv_estimate(ms.llm, context, parallel).value_gb
            if weight is None or kv is None:
                return None, "Не хватает параметров весов/KV для расчёта gpu_memory_utilization vLLM"
            # Reserve a fraction for vLLM that covers weights, KV and runtime.
            fraction = ceil(((weight + kv + 0.9 + 0.25) / vram) * 100) / 100
            if fraction > 0.95:
                return None, "vLLM требует >95% одной GPU без памяти под остальные модели"
            fraction = max(0.05, fraction)
        mem = calculate_memory(hw, req, models, quant, devices, context, parallel, fraction)
        if mem.fit_by_estimate is False:
            return None, (f"Не помещается одновременно: VRAM {mem.required_vram_gb}/{vram} GiB, "
                          f"RAM {mem.required_ram_gb}/{hw.ram_gb} GiB")
        warnings = []
        if mem.fit_by_estimate is None:
            missing = [f.name for f in mem.factors if f.value_gb is None]
            warnings.append("Неизвестные величины: " + ", ".join(missing) + "; размещение не подтверждено")
        if any(m.evidence_status == "unknown" for m in models.values()):
            warnings.append("Часть метаданных моделей имеет status=unknown: проверить model cards")
        if req.language != "unspecified" and any(not m.languages for m in models.values()):
            warnings.append("Поддержка запрошенного языка подтверждена не для всех компонентов")
        if req.local_only and any(m.local_available is None for m in models.values()):
            warnings.append("Локальная доступность весов указана не для всех моделей")
        if req.license_allowlist and any(m.license is None for m in models.values()):
            warnings.append("Лицензии не всех моделей известны, нужна проверка")
        if quant in {"awq", "gptq", "fp8", "q4_k_m", "q4_0", "q8_0"}:
            variants = ms.llm.metadata.get("variant_model_ids", {})
            if not isinstance(variants, dict) or quant not in variants:
                warnings.append("Не указан идентификатор отдельного квантованного чекпойнта; "
                                "проверить путь к весам для " + quant)
        if len(hw.gpus) > 1 or any(g.count > 1 for g in hw.gpus):
            warnings.append("Проверялась одна самая ёмкая GPU; multi-GPU sharding не моделируется")
        if context < req.context_length:
            warnings.append(f"Контекст сокращён с {req.context_length} до {context} по разрешению")
        if parallel < req.parallel_sessions:
            warnings.append(f"Параллелизм сокращён с {req.parallel_sessions} до {parallel} по разрешению")
        if "cpu" in devices:
            warnings.append("CPU размещение может существенно увеличить latency")
        comps = {}
        for role, device, batch in zip(("llm", "embedding", "reranker"), devices,
                                       (1, req.embedding_batch_size, req.reranker_batch_size)):
            m = models[role]
            role_quant = quant if role == "llm" else m.quantizations[0]
            comps[role] = make_config(m, role, device, engine[role],
                                      role_quant, context, parallel, batch,
                                      fraction if role == "llm" else None)
        sources = list(dict.fromkeys(
            [source for model in models.values() for source in model.sources] +
            [ENGINE_DOCS[e] for e in engine.values()]))
        strategy_id = (f"{ms.id}|{llm_engine}|{quant}|{'.'.join(devices)}|"
                       f"context={context}|parallel={parallel}")
        explanation = (f"Комплект {ms.id}: {ms.llm.id} + {ms.embedding.id} + {ms.reranker.id}. "
                       f"LLM engine={llm_engine}, quant={quant}, размещение {devices}, "
                       f"контекст {context}, сессий {parallel}. "
                       f"Оценка VRAM {mem.required_vram_gb} GiB и RAM {mem.required_ram_gb} GiB; "
                       "фактический запуск требует проверки.")
        option = Option(id=strategy_id, model_set_id=ms.id, model_set_score=ms.score,
                        context_length=context, parallel_sessions=parallel,
                        llm=comps["llm"], embedding=comps["embedding"],
                        reranker=comps["reranker"], memory=mem,
                        status=("estimated_fit" if mem.fit_by_estimate is True and
                                not any(m.evidence_status == "unknown" for m in models.values()) and
                                not (req.language != "unspecified" and any(not m.languages for m in models.values())) and
                                not (req.local_only and any(m.local_available is None for m in models.values())) and
                                not (req.license_allowlist and any(m.license is None for m in models.values()))
                                else "requires_verification"),
                        explanation=explanation, warnings=warnings, sources=sources)
        return option, ""

    @staticmethod
    def utility(x: Option, req: Requirements) -> float:
        # Candidate scores are inherited from Agent 3. No unmeasured speed is invented.
        bits = BITS.get(x.llm.quantization, 4)
        quant_quality = bits / 16
        context = x.context_length / req.context_length
        concurrency = x.parallel_sessions / req.parallel_sessions
        gpu_count = sum(c.device == "gpu" for c in (x.llm, x.embedding, x.reranker))
        relative_vram = (x.memory.required_vram_gb or x.memory.available_vram_gb) / max(
            x.memory.available_vram_gb, 1)
        if req.priority == "memory":
            score = quant_quality * .45 + context + concurrency + gpu_count * .2 - relative_vram * 2
        elif req.priority == "quality":
            score = quant_quality * 3 + context * 2 + concurrency + gpu_count * .4 - relative_vram * .25
        elif req.priority == "speed":
            score = quant_quality * .8 + context + concurrency * 2 + gpu_count * .8 - relative_vram * .25
        else:
            score = quant_quality * 1.5 + context * 1.5 + concurrency * 1.5 + gpu_count * .45 - relative_vram * .3
        engine_bonus = (0.2 if x.llm.engine == "vllm" and x.parallel_sessions > 1
                        else 0.05 if x.llm.engine == "llama.cpp" else 0)
        return score + x.model_set_score * 1.5 + engine_bonus + (
            0 if x.memory.fit_by_estimate is True else -1.5)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Do not mutate Agent 1–3 data; return complete updated dictionary."""
        if not isinstance(state, dict):
            raise TypeError("Agent state must be dict")
        out = deepcopy(state)
        out.setdefault("trace", [])
        out.setdefault("errors", [])
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            hw = normalize_hardware(state["hardware"])
            req = normalize_requirements(state["requirements"])
            model_sets = normalize_sets(state["model_sets"])
        except (KeyError, ValueError, TypeError, ValidationError) as exc:
            message = f"Agent 4 input validation error: {exc}"
            out["errors"].append(message)
            out["deployment_plan"] = None
            out["trace"].append({"agent": "inference_configurator", "event": "validation_failed",
                                 "status": "error", "timestamp": stamp, "message": message})
            return out
        out["trace"].append({"agent": "inference_configurator", "event": "started",
                             "status": "running", "timestamp": stamp,
                             "received_sets": len(model_sets)})
        variants: list[Option] = []
        rejected, seen = [], set()
        for ms in model_sets:
            quantizations = [q for q in ms.llm.quantizations if q in BITS]
            quantizations.sort(key=lambda q: BITS[q], reverse=req.priority != "memory")
            for llm_engine, q, placement, ctx, sessions in product(
                    engine_choices(ms.llm, "llm"), quantizations,
                    schemes(bool(hw.gpus), req.allow_cpu_offload),
                    levels(req.context_length, req.min_context_length, req.allow_reduce_context),
                    levels(req.parallel_sessions, req.min_parallel_sessions, req.allow_reduce_parallelism)):
                option, reason = self.build_option(ms, hw, req, q, placement, ctx, sessions,
                                                   llm_engine)
                if option:
                    variants.append(option)
                elif (ms.id, reason) not in seen:
                    rejected.append({"model_set_id": ms.id, "reason": reason})
                    seen.add((ms.id, reason))
        variants.sort(key=lambda v: self.utility(v, req), reverse=True)
        warnings = []
        if variants and self.ranker:
            # LLM may only rank a deterministic shortlist. Structured output validated.
            try:
                shortlist = variants[:12]
                reply = RankedDecision.model_validate(self.ranker({
                    "options": [{"strategy_id": v.id, "description": v.explanation}
                                for v in shortlist]}))
                approved = next((v for v in shortlist if v.id == reply.strategy_id), None)
                if approved is None:
                    raise ValueError("LLM selected a strategy outside the shortlist")
                if reply.explanation.strip():
                    approved = approved.model_copy(update={
                        "explanation": approved.explanation + " " + reply.explanation.strip()})
                variants = [approved] + [v for v in variants if v.id != approved.id]
            except (ValidationError, ValueError, TypeError) as exc:
                warnings.append(f"LLM ranking ignored; deterministic fallback used: {exc}")
        primary = variants[0] if variants else None
        remainder = [v for v in variants[1:] if v.id != primary.id] if primary else []
        verifiable_fallbacks = [v for v in remainder if v.memory.fit_by_estimate is True]
        pool = verifiable_fallbacks or remainder
        fallback = min(pool, key=lambda v: (
            float("inf") if v.memory.required_vram_gb is None else v.memory.required_vram_gb,
            float("inf") if v.memory.required_ram_gb is None else v.memory.required_ram_gb,
            -self.utility(v, req))) if pool else None
        if not primary:
            warnings.append("Подходящая конфигурация не найдена. Проверьте каталог, CPU-offload, "
                            "уменьшение контекста/параллелизма, GGUF/квантизацию или unload policy.")
        if primary and not fallback:
            warnings.append("Резервной resident-конфигурации в переданных комплектах нет")
        if len(hw.gpus) > 1 or any(g.count > 1 for g in hw.gpus):
            warnings.append("Multi-GPU не рассчитывается: используется VRAM одной GPU")
        if primary:
            warnings.extend(primary.warnings)
        plan = DeploymentPlan(project_id=str(state.get("project_id", "unknown-project")),
            status="planned" if primary else "no_feasible_solution",
            primary=primary, fallback=fallback, rejected=rejected,
            warnings=list(dict.fromkeys(warnings)),
            sources=list(dict.fromkeys([s for v in (primary, fallback) if v for s in v.sources])),
            created_at=datetime.now(timezone.utc).isoformat(),
            constraints={"hardware": hw.model_dump(mode="json"),
                         "requirements": req.model_dump(mode="json")})
        out["deployment_plan"] = plan.model_dump(mode="json")
        out["trace"].append({"agent": "inference_configurator", "event": "finished",
                             "status": "success" if primary else "no_solution",
                             "timestamp": datetime.now(timezone.utc).isoformat(),
                             "variants_passed": len(variants),
                             "primary_id": primary.id if primary else None,
                             "fallback_id": fallback.id if fallback else None})
        return out


def configure_inference(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph-friendly node; receives and returns plain dictionaries."""
    return InferenceConfigurator().run(state)
