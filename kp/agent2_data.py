from typing import Any


def get_agent2_data() -> dict[str, Any]:
    return {
        "catalog_version": "2026.10.1",
        "candidates": [
            {
                "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                "role": "llm",
                "parameters_b": 14.7,
                "context_length": 32768,
                "source_url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ",
            },
            {
                "model_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "role": "llm",
                "parameters_b": 32.5,
                "context_length": 32768,
                "source_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ",
            },
            {
                "model_id": "BAAI/bge-m3",
                "role": "embedding",
                "parameters_b": 0.56,
                "context_length": 8192,
                "source_url": "https://huggingface.co/BAAI/bge-m3",
            },
            {
                "model_id": "BAAI/bge-reranker-v2-m3",
                "role": "reranker",
                "parameters_b": 0.56,
                "context_length": 8192,
                "source_url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
            },
        ],
        "trace_event": {
            "agent": "ModelResearcher",
            "step": "collect_candidates",
            "details": "Собрано и нормализовано 4 модели-кандидата из каталога и Hugging Face.",
            "candidates_count": 4,
        },
    }