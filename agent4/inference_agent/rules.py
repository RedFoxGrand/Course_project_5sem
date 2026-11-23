"""Normalization and engine-compatibility rule base.

Engines are accepted only when model cards explicitly declare support. The
rules do not claim generic architecture compatibility without verification.
"""
from __future__ import annotations
from typing import Any
from .contracts import Hardware, Requirements, Model, ModelSet, InferenceConfig

ENGINE_DOCS = {
    "vllm": "https://docs.vllm.ai/en/stable/cli/serve/",
    "llama.cpp": "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
    "ollama": "https://github.com/ollama/ollama/blob/main/docs/faq.mdx",
    "tei": "https://huggingface.co/docs/text-embeddings-inference/en/cli_arguments",
}
BITS = {"fp32": 32, "bf16": 16, "fp16": 16, "float16": 16,
        "fp8": 8, "int8": 8, "q8_0": 8, "int4": 4, "q4_k_m": 4,
        "q4_0": 4, "awq": 4, "gptq": 4}

def get(d: dict, *names: str, default: Any = None):
    return next((d[n] for n in names if n in d and d[n] is not None), default)

def normalize_hardware(d: dict) -> Hardware:
    if not isinstance(d, dict):
        raise ValueError("hardware must be a dictionary")
    gs = d.get("gpus")
    if gs is None:
        vram = get(d, "vram_gb", "gpu_vram_gb", "vram")
        gs = [] if vram is None else [{"name": get(d, "gpu", "gpu_name", default="GPU"),
                                      "vram_gb": vram, "count": get(d, "gpu_count", default=1)}]
    if isinstance(gs, dict):
        gs = [gs]
    return Hardware.model_validate({
        "cpu": get(d, "cpu", default="Unknown CPU"),
        "os": get(d, "os", "operating_system", default="Linux"),
        "ram_gb": get(d, "ram_gb", "ram"),
        "reserved_ram_gb": get(d, "reserved_ram_gb", default=4),
        "reserved_vram_gb": get(d, "reserved_vram_gb", default=0),
        "gpus": [{"name": get(g, "name", "model", default="GPU"),
                  "vram_gb": get(g, "vram_gb", "vram", "memory_gb"),
                  "count": get(g, "count", default=1)} for g in gs],
    })

def normalize_requirements(d: dict) -> Requirements:
    if not isinstance(d, dict):
        raise ValueError("requirements must be a dictionary")
    context = get(d, "context_length", "max_context", default=4096)
    sessions = get(d, "parallel_sessions", "concurrency", default=1)
    reduce_context = get(d, "allow_reduce_context", "allow_context_reduction", default=False)
    reduce_parallel = get(d, "allow_reduce_parallelism", "allow_parallelism_reduction", default=False)
    return Requirements.model_validate({
        "language": get(d, "language", default="unspecified"),
        "domain": get(d, "domain", default="general"),
        "local_only": get(d, "local_only", default=True),
        "context_length": context,
        "parallel_sessions": sessions,
        "min_context_length": get(d, "min_context_length", "minimum_context_length",
                                  default=min(1024, context) if reduce_context else context),
        "min_parallel_sessions": get(d, "min_parallel_sessions", "minimum_parallel_sessions",
                                    default=1 if reduce_parallel else sessions),
        "allow_reduce_context": reduce_context,
        "allow_reduce_parallelism": reduce_parallel,
        "allow_cpu_offload": get(d, "allow_cpu_offload", default=True),
        "priority": get(d, "priority", default="balanced"),
        "embedding_batch_size": get(d, "embedding_batch_size", default=16),
        "reranker_batch_size": get(d, "reranker_batch_size", default=8),
        "license_allowlist": get(d, "license_allowlist", "allowed_licenses", default=[]),
    })

def normalize_model(d: dict | str) -> Model:
    if isinstance(d, str):
        d = {"id": d}
    if not isinstance(d, dict):
        raise ValueError("Each model must be a dictionary")
    quants = get(d, "quantizations", "supported_quantizations", default=["fp16"])
    fmts = get(d, "formats", "weight_formats", default=[])
    engines = get(d, "engines", "supported_engines", default=[])
    if isinstance(quants, str):
        quants = [quants]
    if isinstance(fmts, str):
        fmts = [fmts]
    if isinstance(engines, str):
        engines = [engines]
    return Model.model_validate({
        "id": get(d, "id", "model_id", "name"),
        "parameters_b": get(d, "parameters_b", "params_b", "parameter_count_b"),
        "weights_gb": get(d, "weights_gb", "weights_size_gb", "size_gb"),
        "weight_sizes_gb": {k.lower(): v for k, v in get(d, "weight_sizes_gb", default={}).items()},
        "weights_quantization": (str(get(d, "weights_quantization", "default_quantization")).lower()
                                 if get(d, "weights_quantization", "default_quantization") else None),
        "quantizations": [str(q).lower() for q in quants],
        "formats": [str(f).lower() for f in fmts],
        "engines": [str(e).lower() for e in engines],
        "max_context": get(d, "max_context", "context_window", "max_model_len"),
        "num_hidden_layers": get(d, "num_hidden_layers", "n_layers"),
        "num_key_value_heads": get(d, "num_key_value_heads", "n_kv_heads"),
        "head_dim": get(d, "head_dim", "kv_head_dim"),
        "kv_cache_gb_per_1k": get(d, "kv_cache_gb_per_1k"),
        "evidence_status": get(d, "evidence_status", default="unknown"),
        "languages": get(d, "languages", "supported_languages", default=[]),
        "local_available": get(d, "local_available", "can_run_locally"),
        "license": get(d, "license"),
        "sources": get(d, "sources", default=[]),
        "metadata": get(d, "metadata", default={}),
    })

def normalize_sets(items: list[dict]) -> list[ModelSet]:
    if not isinstance(items, list) or not items:
        raise ValueError("model_sets must be a nonempty list of dictionaries")
    result = []
    for idx, item in enumerate(items):
        model_dict = item.get("models", item)
        result.append(ModelSet.model_validate({
            "id": get(item, "id", "set_id", default=f"set-{idx+1}"),
            "score": get(item, "score", "ranking_score", default=0),
            "llm": normalize_model(get(model_dict, "llm", "llm_model")),
            "embedding": normalize_model(get(model_dict, "embedding", "embedding_model")),
            "reranker": normalize_model(get(model_dict, "reranker", "rerank", "reranker_model")),
        }))
    return result

def engine_choices(model: Model, role: str) -> list[str]:
    """Only engines explicitly supported by catalogue + available weight formats."""
    engines, formats = set(model.engines), set(model.formats)
    if role == "llm":
        choices = []
        if "vllm" in engines and "safetensors" in formats:
            choices.append("vllm")
        if "llama.cpp" in engines and "gguf" in formats:
            choices.append("llama.cpp")
        if "ollama" in engines and ("gguf" in formats or "ollama" in formats):
            choices.append("ollama")
        return choices
    return ["tei"] if "tei" in engines and (
        "safetensors" in formats or "pytorch" in formats) else []

def choose_engine(model: Model, role: str) -> str | None:
    return next(iter(engine_choices(model, role)), None)

def make_config(model: Model, role: str, device: str, engine: str, quant: str,
                context: int, parallel: int, batch: int, gpu_fraction: float | None) -> InferenceConfig:
    notes = []
    variant_ids = model.metadata.get("variant_model_ids", {})
    actual_model_id = (variant_ids.get(quant, model.id) if isinstance(variant_ids, dict)
                       else model.id)
    if engine == "vllm":
        params = {"dtype": "float16", "max_model_len": context, "max_num_seqs": parallel,
                  "gpu_memory_utilization": gpu_fraction}
        cmd = ["vllm", "serve", actual_model_id, "--dtype", "float16", "--max-model-len", str(context),
               "--max-num-seqs", str(parallel), "--gpu-memory-utilization", str(gpu_fraction)]
        if quant in {"awq", "gptq", "fp8"}:
            cmd += ["--quantization", quant]
        elif quant not in {"fp16", "float16"}:
            notes.append("Проверить dtype модели и фактически доступные веса/квантизацию.")
    elif engine == "llama.cpp":
        params = {"ctx_size": context * parallel, "parallel": parallel,
                  "n_gpu_layers": 99 if device == "gpu" else 0}
        variant_paths = model.metadata.get("variant_paths", {})
        gguf_path = (variant_paths.get(quant) if isinstance(variant_paths, dict) else None)
        cmd = ["llama-server", "--model", str(gguf_path or model.metadata.get("local_path", "<PATH_TO_GGUF>")),
               "--ctx-size", str(context * parallel), "--parallel", str(parallel),
               "--n-gpu-layers", str(params["n_gpu_layers"])]
        notes.append("Для GGUF укажите реальный local_path; контекст в llama.cpp общий для слотов.")
    elif engine == "ollama":
        params = {"OLLAMA_CONTEXT_LENGTH": context, "OLLAMA_NUM_PARALLEL": parallel,
                  "OLLAMA_MAX_LOADED_MODELS": 3}
        cmd = ["ollama", "run", actual_model_id]
        notes.append("Переменные окружения Ollama задаются серверу; модель нужно импортировать заранее.")
    else:  # TEI
        params = {"max_concurrent_requests": max(8, parallel * 2), "max_batch_requests": batch}
        cmd = ["text-embeddings-router", "--model-id", model.id,
               "--max-concurrent-requests", str(params["max_concurrent_requests"]),
               "--max-batch-requests", str(batch)]
        notes.append("Проверить поддержку архитектуры и CPU/CUDA-образа TEI на целевом сервере.")
    return InferenceConfig(role=role, model_id=model.id, device=device,
                           engine=engine, precision="fp16", quantization=quant,
                           weight_format="gguf" if engine in {"llama.cpp", "ollama"} else "safetensors",
                           parameters=params, command=cmd, notes=notes)
