from typing import Any


def get_agent1_data(scenario: str = "rtx5090") -> dict[str, Any]:
    if scenario == "rtx4090":
        return {
            "project_id": "prj_rtx4090_002",
            "request": {
                "raw_query": "Перенос русскоязычного RAG-ассистента на рабочую станцию RTX 4090 (24 ГБ VRAM, 64 ГБ RAM).",
            },
            "requirements": {
                "raw_query": "Перенос русскоязычного RAG-ассистента на рабочую станцию RTX 4090 (24 ГБ VRAM, 64 ГБ RAM).",
                "language": "ru",
                "domain": "technical",
                "max_context_length": 8192,
                "parallel_sessions": 3,
                "privacy_required": True,
                "priority": "balanced",
            },
            "hardware": {
                "gpu_model": "NVIDIA GeForce RTX 4090",
                "gpu_count": 1,
                "vram_gb": 24.0,
                "ram_gb": 64.0,
                "cpu_model": "AMD Ryzen 9 9950X",
                "os_name": "Linux",
            },
            "trace_event": {
                "agent": "RequirementAnalyst",
                "step": "analyze_requirements",
                "details": "Обновлен профиль оборудования: переход на RTX 4090 (24 ГБ VRAM, 64 ГБ RAM), 3 сессии.",
                "candidates_count": 0,
            },
        }

    return {
        "project_id": "prj_rtx5090_001",
        "request": {
            "raw_query": "Локальный русскоязычный RAG-ассистент по внутренней технической документации на RTX 5090.",
        },
        "requirements": {
            "raw_query": "Локальный русскоязычный RAG-ассистент по внутренней технической документации на RTX 5090.",
            "language": "ru",
            "domain": "technical",
            "max_context_length": 16384,
            "parallel_sessions": 5,
            "privacy_required": True,
            "priority": "quality",
        },
        "hardware": {
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "ram_gb": 64.0,
            "cpu_model": "AMD Ryzen 9 9950X",
            "os_name": "Linux",
        },
        "trace_event": {
            "agent": "RequirementAnalyst",
            "step": "analyze_requirements",
            "details": "Формализован профиль: ru, technical, 5 сессий, RTX 5090 (32 ГБ VRAM).",
            "candidates_count": 0,
        },
    }