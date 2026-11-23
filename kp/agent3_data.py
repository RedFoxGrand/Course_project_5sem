from typing import Any


def get_agent3_data(scenario: str = "rtx5090") -> dict[str, Any]:
    if scenario == "rtx4090":
        return {
            "primary_set_id": "set_qwen14b_gpu_bge_cpu_hybrid",
            "alternative_set_id": "set_qwen7b_all_gpu_eco",
            "model_sets": [
                {
                    "set_id": "set_qwen14b_gpu_bge_cpu_hybrid",
                    "llm": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                    "embedding": "BAAI/bge-m3",
                    "reranker": "BAAI/bge-reranker-v2-m3",
                }
            ],
            "rejected_options": [
                {
                    "candidate_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                    "role": "llm",
                    "reason": "Веса 32B модели и KV-кэш не помещаются в 24 ГБ VRAM RTX 4090.",
                }
            ],
            "trace_event": {
                "agent": "ModelSelector",
                "step": "select_model_set",
                "details": "Выбран гибридный комплект set_qwen14b_gpu_bge_cpu_hybrid для RTX 4090.",
                "candidates_count": 3,
            },
        }

    return {
        "primary_set_id": "set_qwen14b_bge_m3_gpu",
        "alternative_set_id": "set_qwen7b_bge_m3_eco",
        "model_sets": [
            {
                "set_id": "set_qwen14b_bge_m3_gpu",
                "llm": "Qwen/Qwen2.5-14B-Instruct-AWQ",
                "embedding": "BAAI/bge-m3",
                "reranker": "BAAI/bge-reranker-v2-m3",
            }
        ],
        "rejected_options": [
            {
                "candidate_id": "Qwen/Qwen2.5-32B-Instruct-AWQ",
                "role": "llm",
                "reason": "Суммарная потребность 34.8 ГБ VRAM с KV-кэшем и embedding/reranker превышает 32 ГБ RTX 5090.",
            },
            {
                "candidate_id": "meta-llama/Llama-3.1-8B-Instruct",
                "role": "llm",
                "reason": "Уступает Qwen2.5-14B на русскоязычной технической документации.",
            },
        ],
        "trace_event": {
            "agent": "ModelSelector",
            "step": "select_model_set",
            "details": "Отобрана тройка Qwen2.5-14B + bge-m3 + bge-reranker-v2-m3.",
            "candidates_count": 3,
        },
    }