from typing import Any


def get_agent4_data(scenario: str = "rtx5090") -> dict[str, Any]:
    if scenario == "rtx4090":
        return {
            "components": [
                {
                    "role": "llm",
                    "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                    "engine": "vllm",
                    "weight_format": "safetensors",
                    "quantization": "awq_int4",
                    "placement": "gpu",
                    "max_model_len": 8192,
                    "max_num_seqs": 3,
                    "keep_loaded": True,
                    "source_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ",
                },
                {
                    "role": "embedding",
                    "model_id": "BAAI/bge-m3",
                    "engine": "tei",
                    "weight_format": "safetensors",
                    "quantization": "fp16",
                    "placement": "cpu",
                    "max_model_len": 8192,
                    "max_num_seqs": 3,
                    "keep_loaded": True,
                    "source_url": "https://huggingface.co/BAAI/bge-m3",
                },
                {
                    "role": "reranker",
                    "model_id": "BAAI/bge-reranker-v2-m3",
                    "engine": "tei",
                    "weight_format": "safetensors",
                    "quantization": "fp16",
                    "placement": "cpu",
                    "max_model_len": 8192,
                    "max_num_seqs": 3,
                    "keep_loaded": True,
                    "source_url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
                },
            ],
            "memory_breakdown": {
                "llm_weights": {"value": 9.2, "source": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ", "status": "declared"},
                "llm_kv_cache": {"value": 4.6, "source": "vllm_kv_cache_formula_8k_3seqs", "status": "estimated"},
                "embedding_resident_memory": {"value": 0.0, "source": "offloaded_to_system_ram", "status": "declared"},
                "reranker_resident_memory": {"value": 0.0, "source": "offloaded_to_system_ram", "status": "declared"},
                "runtime_overhead": {"value": 2.2, "source": "vllm_cuda_graphs_overhead", "status": "estimated"},
                "safety_reserve": {"value": 2.4, "source": "safety_margin_10_percent", "status": "declared"},
                "total_vram_gb": 18.4,
                "total_ram_gb": 18.5,
            },
            "decision_rationale": (
                "При переносе на RTX 4090 (24 ГБ VRAM) модели embedding и reranker перенесены на CPU (RAM), "
                "контекст LLM уменьшен до 8192 токенов, а параллелизм снижен с 5 до 3 сессий (итого 18.4 ГБ VRAM)."
            ),
            "warnings": [
                "Объем KV-кэша для 3 сессий рассчитан аналитически без запуска benchmark."
            ],
            "trace_event": {
                "agent": "InferenceConfigurator",
                "step": "configure_inference",
                "details": "Перенос embedding/reranker на CPU, снижение параллелизма до 3 сессий (18.4 ГБ VRAM).",
                "candidates_count": 3,
            },
        }

    return {
        "components": [
            {
                "role": "llm",
                "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                "engine": "vllm",
                "weight_format": "safetensors",
                "quantization": "awq_int4",
                "placement": "gpu",
                "max_model_len": 16384,
                "max_num_seqs": 5,
                "keep_loaded": True,
                "source_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ",
            },
            {
                "role": "embedding",
                "model_id": "BAAI/bge-m3",
                "engine": "tei",
                "weight_format": "safetensors",
                "quantization": "fp16",
                "placement": "gpu",
                "max_model_len": 8192,
                "max_num_seqs": 5,
                "keep_loaded": True,
                "source_url": "https://huggingface.co/BAAI/bge-m3",
            },
            {
                "role": "reranker",
                "model_id": "BAAI/bge-reranker-v2-m3",
                "engine": "tei",
                "weight_format": "safetensors",
                "quantization": "fp16",
                "placement": "gpu",
                "max_model_len": 8192,
                "max_num_seqs": 5,
                "keep_loaded": True,
                "source_url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
            },
        ],
        "memory_breakdown": {
            "llm_weights": {"value": 9.2, "source": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ", "status": "declared"},
            "llm_kv_cache": {"value": 8.5, "source": "vllm_kv_cache_formula_16k_5seqs", "status": "estimated"},
            "embedding_resident_memory": {"value": 1.4, "source": "https://huggingface.co/BAAI/bge-m3", "status": "catalog"},
            "reranker_resident_memory": {"value": 1.4, "source": "https://huggingface.co/BAAI/bge-reranker-v2-m3", "status": "catalog"},
            "runtime_overhead": {"value": 2.5, "source": "vllm_cuda_graphs_overhead", "status": "estimated"},
            "safety_reserve": {"value": 3.0, "source": "safety_margin_10_percent", "status": "declared"},
            "total_vram_gb": 26.0,
            "total_ram_gb": 12.0,
        },
        "decision_rationale": (
            "Все три модели постоянно размещены в 32 ГБ GDDR7 на RTX 5090 (суммарно 26.0 ГБ VRAM с учетом KV-кэша на 5 сессий)."
        ),
        "warnings": [
            "Объем KV-кэша рассчитан аналитически без запуска реального benchmark."
        ],
        "trace_event": {
            "agent": "InferenceConfigurator",
            "step": "configure_inference",
            "details": "Рассчитано совместное размещение на GPU: 26.0 ГБ из 32.0 ГБ VRAM.",
            "candidates_count": 3,
        },
    }