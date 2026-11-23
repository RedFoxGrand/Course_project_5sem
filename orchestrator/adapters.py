from datetime import datetime, timezone
from typing import Any

from agent3.schemas import (
    HardwareProfile as Agent3Hardware,
    ModelCandidate as Agent3Candidate,
    RequirementProfile as Agent3Requirement,
)


def derive_priority(req: dict[str, Any]) -> str:
    """Переводит числовые приоритеты Агента 1 (1..5) в строку для Агентов 4 и 5."""
    if isinstance(req.get("priority"), str):
        return req["priority"].lower()
    pq = int(req.get("priority_quality", 3))
    ps = int(req.get("priority_speed", 3))
    pm = int(req.get("priority_memory", 3))
    if pq == ps == pm:
        return "balanced"
    best = max((pq, "quality"), (ps, "speed"), (pm, "memory"), key=lambda x: x[0])
    return best[1]


def normalize_trace_event(raw: dict[str, Any], default_agent: str) -> dict[str, Any]:
    """Приводит события трассы любого агента (включая Агента 4 с ключом event) к единому формату."""
    step = raw.get("step") or raw.get("event") or "step"
    details = raw.get("details") or raw.get("message")
    if not details:
        parts = [f"status={raw.get('status', 'ok')}"]
        if "variants_passed" in raw:
            parts.append(f"variants={raw['variants_passed']}")
        if raw.get("primary_id"):
            parts.append(f"primary={raw['primary_id']}")
        details = ", ".join(parts)

    return {
        "agent": str(raw.get("agent") or default_agent),
        "step": str(step),
        "details": str(details),
        "candidates_count": int(
            raw.get("candidates_count")
            or raw.get("received_sets")
            or raw.get("variants_passed")
            or 0
        ),
        "timestamp": str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
    }


def adapt_for_agent3(
    state: dict[str, Any],
) -> tuple[Agent3Requirement, Agent3Hardware, list[Agent3Candidate]]:
    """Адаптер SharedDesignState (Agent 1 + Agent 2) -> схемы Agent 3."""
    req_d = state.get("requirements") or {}
    hw_d = state.get("hardware") or {}

    req3 = Agent3Requirement.model_validate(req_d)
    hw3 = Agent3Hardware(
        cpu_cores=int(hw_d.get("cpu_cores", 16)),
        ram_gb=float(hw_d.get("ram_gb", 64.0)),
        gpu_count=int(hw_d.get("gpu_count", 1 if hw_d.get("vram_gb", 0) > 0 else 0)),
        gpu_vram_gb=float(hw_d.get("vram_gb", hw_d.get("gpu_vram_gb", 0.0))),
        os_name=str(hw_d.get("os", hw_d.get("os_name", "Linux"))),
    )
    candidates3 = [Agent3Candidate.model_validate(c) for c in state.get("candidates", [])]
    return req3, hw3, candidates3


def _estimate_kv_gb_per_1k(params_b: float | None) -> float | None:
    """Консервативная эвристика KV cache (GiB на 1K токенов, FP16 GQA) для vLLM при отсутствии config.json."""
    if params_b is None or params_b <= 0:
        return None
    if params_b <= 4.5:
        return 0.04
    if params_b <= 10.0:
        return 0.07
    if params_b <= 16.0:
        return 0.11
    if params_b <= 35.0:
        return 0.18
    return 0.30


def _candidate_to_agent4_model(raw: dict[str, Any], role: str) -> dict[str, Any]:
    """Преобразует карточку ModelCandidate (с EvidenceItem) в плоский словарь для Agent 4."""
    params_ev = raw.get("parameters_count_b") or {}
    ctx_ev = raw.get("context_length") or {}

    params_b = params_ev.get("value") if isinstance(params_ev, dict) else params_ev
    ctx_val = ctx_ev.get("value") if isinstance(ctx_ev, dict) else ctx_ev
    ev_status = params_ev.get("status", "unknown") if isinstance(params_ev, dict) else "unknown"

    quants = [str(q).lower() for q in (raw.get("supported_quantizations") or ["fp16"])]
    if role in ("embedding", "reranker"):
        # Agent 4 требует, чтобы fp16/bf16 стояло первым для TEI
        if "fp16" in quants:
            quants = ["fp16"] + [q for q in quants if q != "fp16"]
        elif "bf16" in quants:
            quants = ["bf16"] + [q for q in quants if q != "bf16"]

    engines = [str(e).lower() for e in (raw.get("compatible_engines") or [])]
    formats = [str(f).lower() for f in (raw.get("supported_formats") or [])]
    source_url = raw.get("source_url")
    sources = [source_url] if source_url else []

    out: dict[str, Any] = {
        "id": raw.get("model_id") or raw.get("name"),
        "parameters_b": params_b,
        "quantizations": quants or ["fp16"],
        "formats": formats,
        "engines": engines,
        "max_context": ctx_val,
        "evidence_status": ev_status,
        "languages": raw.get("languages") or [],
        "local_available": raw.get("is_local_compatible", True),
        "license": raw.get("license"),
        "sources": sources,
        "metadata": {},
    }

    if role == "llm":
        kv_est = _estimate_kv_gb_per_1k(params_b)
        if kv_est is not None:
            out["kv_cache_gb_per_1k"] = kv_est

    return out


def adapt_for_agent4(state: dict[str, Any]) -> dict[str, Any]:
    """Адаптер SharedDesignState (Agents 1-3) -> входной dict для Agent 4."""
    hw_d = state.get("hardware") or {}
    req_d = state.get("requirements") or {}

    vram_gb = float(hw_d.get("vram_gb", 0.0))
    gpu_count = int(hw_d.get("gpu_count", 0))
    gpus = (
        [
            {
                "name": str(hw_d.get("gpu_name") or "GPU"),
                "vram_gb": vram_gb,
                "count": max(gpu_count, 1),
            }
        ]
        if vram_gb > 0 and (gpu_count > 0 or hw_d.get("gpu_name"))
        else []
    )

    hardware4 = {
        "cpu": str(hw_d.get("cpu", "Generic x86_64")),
        "ram_gb": float(hw_d.get("ram_gb", 64.0)),
        "os": str(hw_d.get("os", "Linux")),
        "gpus": gpus,
        "reserved_ram_gb": 4.0,
        "reserved_vram_gb": 0.0,
    }

    requirements4 = {
        "language": str(req_d.get("language", "ru")),
        "domain": str(req_d.get("domain", "general")),
        "local_only": bool(req_d.get("privacy_required", True)),
        "context_length": int(req_d.get("max_context_length", 8192)),
        "parallel_sessions": int(req_d.get("parallel_sessions", 1)),
        "allow_cpu_offload": bool(req_d.get("allow_cpu_offload", hw_d.get("allow_cpu_offload", True))),
        "allow_reduce_context": bool(req_d.get("allow_reduce_context", False)),
        "allow_reduce_parallelism": bool(req_d.get("allow_reduce_parallelism", False)),
        "priority": derive_priority(req_d),
    }

    model_sets4: list[dict[str, Any]] = []
    raw_sets = state.get("model_sets") or []
    for idx, ms in enumerate(raw_sets):
        score = round(max(0.5, 1.0 - idx * 0.15), 2)
        model_sets4.append(
            {
                "id": str(ms.get("set_name") or ms.get("id") or f"set-{idx + 1}"),
                "score": score,
                "llm": _candidate_to_agent4_model(ms["llm"], "llm"),
                "embedding": _candidate_to_agent4_model(ms["embedding"], "embedding"),
                "reranker": _candidate_to_agent4_model(ms["reranker"], "reranker"),
            }
        )

    return {
        "project_id": state.get("project_id", "prj_unknown"),
        "request": state.get("request", {}),
        "hardware": hardware4,
        "requirements": requirements4,
        "model_sets": model_sets4,
        "trace": [],
        "errors": [],
    }


def _factor_to_provenance(
    factors_map: dict[str, dict[str, Any]], key: str, default_source: str
) -> dict[str, Any]:
    f = factors_map.get(key)
    if not f:
        return {"value": 0.0, "source": default_source, "status": "unknown"}
    val = f.get("value_gb")
    if val is None:
        return {"value": 0.0, "source": str(f.get("source", default_source)), "status": "unknown"}
    return {
        "value": float(val),
        "source": str(f.get("source", default_source)),
        "status": str(f.get("status", "estimated")),
    }


def adapt_for_agent5(state: dict[str, Any]) -> dict[str, Any]:
    """Адаптер SharedDesignState (Agents 1-4) -> входной dict для ReflectionAgent (Agent 5)."""
    req_d = state.get("requirements") or {}
    hw_d = state.get("hardware") or {}
    request_d = state.get("request") or {}
    plan = state.get("deployment_plan") or {}
    primary = plan.get("primary") or {}
    fallback = plan.get("fallback") or {}

    # Карта source_url по model_id из кандидатов Агента 2
    url_by_id: dict[str, str] = {
        c["model_id"]: c.get("source_url", "")
        for c in state.get("candidates", [])
        if isinstance(c, dict) and c.get("model_id")
    }

    requirements5 = {
        "raw_query": str(request_d.get("task_description", "")),
        "language": str(req_d.get("language", "ru")),
        "domain": str(req_d.get("domain", "general")),
        "max_context_length": int(req_d.get("max_context_length", 8192)),
        "parallel_sessions": int(req_d.get("parallel_sessions", 1)),
        "privacy_required": bool(req_d.get("privacy_required", True)),
        "priority": derive_priority(req_d),
    }

    hardware5 = {
        "gpu_model": str(hw_d.get("gpu_name") or "CPU-only"),
        "gpu_count": int(hw_d.get("gpu_count", 0)),
        "vram_gb": float(hw_d.get("vram_gb", 0.0)),
        "ram_gb": float(hw_d.get("ram_gb", 64.0)),
        "cpu_model": str(hw_d.get("cpu", "x86_64")),
        "os_name": str(hw_d.get("os", "Linux")),
    }

    components5: list[dict[str, Any]] = []
    ctx_len = int(primary.get("context_length", requirements5["max_context_length"]))
    seqs = int(primary.get("parallel_sessions", requirements5["parallel_sessions"]))
    for role in ("llm", "embedding", "reranker"):
        cfg = primary.get(role)
        if not isinstance(cfg, dict):
            continue
        mid = str(cfg["model_id"])
        components5.append(
            {
                "role": role,
                "model_id": mid,
                "engine": str(cfg["engine"]),
                "weight_format": str(cfg.get("weight_format", "safetensors")),
                "quantization": str(cfg.get("quantization", "fp16")),
                "placement": str(cfg.get("device", "gpu")),
                "max_model_len": ctx_len if role == "llm" else min(ctx_len, 8192),
                "max_num_seqs": seqs,
                "keep_loaded": True,
                "source_url": url_by_id.get(mid, ""),
            }
        )

    mem = primary.get("memory") or {}
    factors_map = {f["name"]: f for f in mem.get("factors", []) if isinstance(f, dict)}

    rt_gpu = (factors_map.get("runtime_gpu") or {}).get("value_gb") or 0.0
    rt_cpu = (factors_map.get("runtime_cpu") or {}).get("value_gb") or 0.0
    vram_safe = (factors_map.get("vram_safety") or {}).get("value_gb") or 0.0
    ram_safe = (factors_map.get("ram_safety") or {}).get("value_gb") or 0.0

    memory5 = {
        "llm_weights": _factor_to_provenance(factors_map, "llm_weights", "agent4_memory"),
        "llm_kv_cache": _factor_to_provenance(factors_map, "llm_kv_cache", "agent4_memory"),
        "embedding_resident_memory": _factor_to_provenance(
            factors_map, "embedding_weights", "agent4_memory"
        ),
        "reranker_resident_memory": _factor_to_provenance(
            factors_map, "reranker_weights", "agent4_memory"
        ),
        "runtime_overhead": {
            "value": round(float(rt_gpu) + float(rt_cpu), 4),
            "source": "agent4_runtime_heuristic",
            "status": "estimated",
        },
        "safety_reserve": {
            "value": round(float(vram_safe) + float(ram_safe), 4),
            "source": "agent4_safety_heuristic",
            "status": "estimated",
        },
        "total_vram_gb": float(mem.get("required_vram_gb") or 0.0),
        "total_ram_gb": float(mem.get("required_ram_gb") or 0.0),
    }

    rejected5: list[dict[str, Any]] = []
    for idx, reason in enumerate(state.get("rejected_reasons", []), 1):
        if str(reason).strip():
            rejected5.append(
                {
                    "candidate_id": f"agent3_option_{idx}",
                    "role": "model_set",
                    "reason": str(reason),
                }
            )
    for rej in plan.get("rejected", []):
        if isinstance(rej, dict) and rej.get("reason"):
            rejected5.append(
                {
                    "candidate_id": str(rej.get("model_set_id", "unknown_set")),
                    "role": "model_set",
                    "reason": str(rej["reason"]),
                }
            )

    warnings5 = list(
        dict.fromkeys(
            list(state.get("warnings", []))
            + list(req_d.get("warnings", []))
            + list(plan.get("warnings", []))
        )
    )

    return {
        "project_id": state.get("project_id", "prj_unknown"),
        "catalog_version": state.get("catalog_version", "2026.11.1"),
        "requirements": requirements5,
        "hardware": hardware5,
        "primary_set_id": str(primary.get("model_set_id", "set_primary")),
        "alternative_set_id": fallback.get("model_set_id") if isinstance(fallback, dict) else None,
        "components": components5,
        "memory_breakdown": memory5,
        "decision_rationale": str(
            primary.get("explanation") or "Выбрана оптимальная конфигурация инференса."
        ),
        "rejected_options": rejected5,
        "warnings": warnings5,
        "trace": [normalize_trace_event(ev, "Pipeline") for ev in state.get("trace", [])],
        "errors": list(state.get("errors", [])),
    }