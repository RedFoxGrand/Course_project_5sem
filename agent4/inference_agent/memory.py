"""All memory calculations are explicitly estimates; nothing is benchmarked."""
from __future__ import annotations
from math import isfinite
from .contracts import Hardware, Model, Requirements, Evidence, MemoryEstimate
from .rules import BITS

GIB = 1024 ** 3

def evidence(name: str, value: float | None, formula: str, inputs: dict,
             source: str, status: str) -> Evidence:
    if value is not None and (value < 0 or not isfinite(value)):
        raise ValueError(f"Invalid memory quantity: {name}")
    return Evidence(name=name, value_gb=round(value, 5) if value is not None else None,
                    formula=formula, inputs=inputs, source=source, status=status)

def weight_estimate(model: Model, quant: str, role: str) -> Evidence:
    if quant in model.weight_sizes_gb:
        size = model.weight_sizes_gb[quant]
        return evidence(f"{role}_weights", size, "weight_sizes_gb[quant]",
                        {"quant": quant}, model.sources[0] if model.sources else "model card", "catalog")
    if model.weights_gb is not None and (model.weights_quantization == quant or
            (model.weights_quantization is None and model.quantizations == [quant])):
        return evidence(f"{role}_weights", model.weights_gb, "declared weights_gb",
                        {"quant": quant}, model.sources[0] if model.sources else "model card", "declared")
    if model.parameters_b is not None and quant in BITS:
        bits = BITS[quant]
        size = model.parameters_b * 1e9 * bits / 8 / GIB * 1.12
        return evidence(f"{role}_weights", size,
                        "parameters_b * 1e9 * bits / 8 / 2**30 * 1.12",
                        {"params_b": model.parameters_b, "bits": bits, "overhead": 1.12},
                        "heuristic: +12% serialization/packing overhead", "estimated")
    return evidence(f"{role}_weights", None, "missing weight size and parameter_count",
                    {"quant": quant}, "model card missing", "unknown")

def kv_estimate(model: Model, context: int, parallel: int) -> Evidence:
    if model.num_hidden_layers and model.num_key_value_heads and model.head_dim:
        raw = (2 * model.num_hidden_layers * model.num_key_value_heads * model.head_dim *
               2 * context * parallel / GIB)  # fp16: 2 bytes per element
        return evidence("llm_kv_cache", raw * 1.1,
                        "2(K,V) * layers * kv_heads * head_dim * 2(bytes) * context * parallel / 2**30 * 1.1",
                        {"layers": model.num_hidden_layers, "kv_heads": model.num_key_value_heads,
                         "head_dim": model.head_dim, "context": context, "parallel": parallel,
                         "kv_precision": "fp16", "overhead": 1.1},
                        "model architecture + 10% estimated overhead", "estimated")
    if model.kv_cache_gb_per_1k is not None:
        return evidence("llm_kv_cache", model.kv_cache_gb_per_1k * context / 1000 * parallel * 1.1,
                        "kv_gb_per_1k * context / 1000 * parallel * 1.1",
                        {"kv_gb_per_1k": model.kv_cache_gb_per_1k,
                         "context": context, "parallel": parallel},
                        "candidate KV metadata +10%", "estimated")
    return evidence("llm_kv_cache", None, "architecture or KV coefficient not provided",
                    {"context": context, "parallel": parallel}, "model card missing", "unknown")

def sum_known(values: list[float | None]) -> float | None:
    return round(sum(values), 4) if all(v is not None for v in values) else None

def calculate_memory(hw: Hardware, req: Requirements, models: dict[str, Model],
                     quant: str, placement: tuple[str, str, str],
                     context: int, parallel: int,
                     vllm_fraction: float | None = None) -> MemoryEstimate:
    gpu_vram = max((g.vram_gb for g in hw.gpus), default=0.0)
    gpu_reserve = (max(1.0, 0.12 * gpu_vram) + hw.reserved_vram_gb) if hw.gpus else 0.0
    ram_reserve = max(2.0, 0.10 * hw.ram_gb) + hw.reserved_ram_gb
    quantities = {
        "llm": weight_estimate(models["llm"], quant, "llm"),
        "embedding": weight_estimate(models["embedding"], models["embedding"].quantizations[0], "embedding"),
        "reranker": weight_estimate(models["reranker"], models["reranker"].quantizations[0], "reranker"),
    }
    kv = kv_estimate(models["llm"], context, parallel)
    batch_emb = 0.1 + req.embedding_batch_size * 0.01
    batch_rerank = 0.1 + req.reranker_batch_size * 0.015
    rt_gpu, rt_cpu = 0.0, 0.0
    for role, device in zip(("llm", "embedding", "reranker"), placement):
        if device == "gpu":
            rt_gpu += 0.9 if role == "llm" else 0.35
        else:
            rt_cpu += 0.6 if role == "llm" else 0.3
    factors = [*quantities.values(), kv,
        evidence("runtime_gpu", rt_gpu, "0.9 per LLM + 0.35 per TEI on GPU",
                 {"placement": placement}, "runtime heuristic", "estimated"),
        evidence("runtime_cpu", rt_cpu, "0.6 per LLM + 0.3 per TEI on CPU",
                 {"placement": placement}, "runtime heuristic", "estimated"),
        evidence("embedding_batch", batch_emb, "0.1 + batch_size * 0.01",
                 {"batch_size": req.embedding_batch_size}, "batch heuristic", "estimated"),
        evidence("reranker_batch", batch_rerank, "0.1 + batch_size * 0.015",
                 {"batch_size": req.reranker_batch_size}, "batch heuristic", "estimated"),
        evidence("vram_safety", gpu_reserve, "max(1, 0.12 * largest_gpu_vram) + reserved_vram",
                 {"vram": gpu_vram, "reserved_vram": hw.reserved_vram_gb}, "safety heuristic", "estimated"),
        evidence("ram_safety", ram_reserve, "max(2, 0.10 * RAM) + reserved_ram",
                 {"ram": hw.ram_gb, "reserved_ram": hw.reserved_ram_gb}, "OS/reserve heuristic", "estimated"),
    ]
    gpu = [gpu_reserve, rt_gpu]
    ram = [ram_reserve, rt_cpu]
    for role, device in zip(("llm", "embedding", "reranker"), placement):
        (gpu if device == "gpu" else ram).append(quantities[role].value_gb)
    (gpu if placement[0] == "gpu" else ram).append(kv.value_gb)
    (gpu if placement[1] == "gpu" else ram).append(batch_emb)
    (gpu if placement[2] == "gpu" else ram).append(batch_rerank)
    # vLLM reserves a fraction of available GPU memory, not just raw weights+KV.
    if vllm_fraction is not None:
        if gpu_vram <= 0:
            raise ValueError("vLLM requires GPU")
        budget = vllm_fraction * gpu_vram
        original = sum_known([quantities["llm"].value_gb, kv.value_gb, 0.9])
        if original is None:
            gpu.append(None)
        else:
            gpu.append(max(0, budget - original))
        factors.append(evidence("vllm_reserved_pool", budget,
                                "gpu_memory_utilization * largest_gpu_vram",
                                {"utilization": vllm_fraction, "vram": gpu_vram},
                                "vLLM memory reservation parameter", "estimated"))
    vram_required, ram_required = sum_known(gpu), sum_known(ram)
    # Even with unknown components, known subtotals already exceeding a device
    # disprove feasibility. They must not be presented as 'unknown but possible'.
    gpu_lower_bound = sum(x for x in gpu if x is not None)
    ram_lower_bound = sum(x for x in ram if x is not None)
    if gpu_lower_bound > gpu_vram + 1e-8 or ram_lower_bound > hw.ram_gb + 1e-8:
        fits = False
    elif vram_required is None or ram_required is None:
        fits = None
    else:
        fits = True
    return MemoryEstimate(factors=factors,
        required_vram_gb=vram_required, available_vram_gb=gpu_vram,
        required_ram_gb=ram_required, available_ram_gb=hw.ram_gb,
        fit_by_estimate=fits, requires_verification=True,
        assumptions=["GiB (not decimal GB); input *_gb assumed GiB",
                     "All 3 models loaded concurrently; no model replication",
                     "One physical GPU; no unverified summing of multiple GPUs",
                     "KV cache FP16; batch/runtime/safety are heuristics",
                     "Measurements and model-engine compatibility must be checked on hardware"])
