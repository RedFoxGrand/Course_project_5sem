import argparse
import asyncio
import os
from pathlib import Path

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from orchestrator.graph import build_master_workflow

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_openrouter_api_key() -> str | None:
    """Загружает API-ключ OpenRouter из окружения или локального файла api_key.txt."""
    env_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    key_file = PROJECT_ROOT / "api_key.txt"
    if key_file.exists():
        content = key_file.read_text(encoding="utf-8").strip()
        if content:
            return content

    return None


def create_agent3_llm(mode: str = "openrouter"):
    """Создает LLM-клиент для Агента 3: 'openrouter', 'ollama' или 'none'."""
    normalized_mode = mode.strip().lower()

    if normalized_mode == "none":
        return None

    if normalized_mode == "ollama":
        return ChatOllama(
            model="qwen2.5:3b",
            temperature=0.0,
        )

    if normalized_mode == "openrouter":
        api_key = _load_openrouter_api_key()
        if not api_key:
            print(
                "Предупреждение: ключ OpenRouter не найден в переменных окружения "
                "(OPENROUTER_API_KEY / OPENAI_API_KEY) и в файле api_key.txt. "
                "Агент 3 перейдет на детерминированный отбор."
            )
            return None

        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="nvidia/nemotron-3.5-super-vl:free",
            temperature=0.0,
            timeout=150.0,
            max_retries=1,
        )

    raise ValueError(f"Неизвестный режим запуска LLM для Агента 3: {mode}")


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки для запуска демонстрации."""
    default_mode = os.getenv("AGENT3_LLM_MODE", "openrouter")
    parser = argparse.ArgumentParser(
        description="Запуск сквозного сценария проектирования ИИ-ассистента."
    )
    parser.add_argument(
        "--agent3-llm",
        choices=["openrouter", "ollama", "none"],
        default=default_mode,
        help="Режим LLM для Агента 3: openrouter, ollama или none (детерминированный отбор).",
    )
    return parser.parse_args()


async def run_scenario(workflow, project_id: str, description: str, hardware: dict) -> None:
    initial_state = {
        "project_id": project_id,
        "request": {"task_description": description, "user_clarifications": {}},
        "hardware": hardware,
        "trace": [],
        "errors": [],
        "warnings": [],
        "rejected_reasons": [],
    }
    result = await workflow.ainvoke(
        initial_state, config={"configurable": {"thread_id": project_id}}
    )

    print("=" * 90)
    print(f"ПРОЕКТ: {project_id}")
    print(f"Ошибки: {result.get('errors') or 'нет'}")

    warnings = result.get("warnings") or []
    if warnings:
        print("Предупреждения:")
        for w in warnings:
            print(f"  ! {w}")

    plan = result.get("deployment_plan") or {}
    primary = plan.get("primary")
    if primary:
        print(f"Выбранный план: {primary['id']}")
        print(
            f"Память (GiB): VRAM={primary['memory']['required_vram_gb']}/{primary['memory']['available_vram_gb']} | "
            f"RAM={primary['memory']['required_ram_gb']}/{primary['memory']['available_ram_gb']}"
        )
        print(f"Команда LLM: {' '.join(primary['llm']['command'])}")

    refl = result.get("reflection")
    if refl:
        print(
            f"OSTIS: id={refl['reflection_id']} | статус={refl['overall_epistemic_status']} | "
            f"похожий кейс={refl.get('similar_project_id')}"
        )

    print("Сквозная трасса:")
    for ev in result.get("trace", []):
        print(f"  - [{ev['agent']} / {ev['step']}] {ev['details']}")


async def main(agent3_mode: str) -> None:
    agent3_llm = create_agent3_llm(agent3_mode)

    workflow, ostis_gateway = build_master_workflow(
        agent3_llm=agent3_llm,
        enable_hf_background_sync=False,
    )

    await run_scenario(
        workflow,
        project_id="prj_rtx5090_001",
        description=(
            "Требуется локальный русскоязычный RAG-ассистент по внутренней технической "
            "документации. Данные нельзя передавать во внешние API. Приоритет качество ответа. "
            "Ожидается 5 параллельных сессий, допускается перенос embedding или reranker на CPU."
        ),
        hardware={
            "cpu": "AMD Ryzen 9 9950X",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
            "vram_gb": 32.0,
            "os": "Linux",
            "allow_cpu_offload": True,
        },
    )

    await run_scenario(
        workflow,
        project_id="prj_rtx4090_002",
        description=(
            "Локальный русскоязычный RAG-ассистент по технической документации на RTX 4090. "
            "Данные нельзя передавать во внешние API. Ожидается 3 параллельные сессии, "
            "допускается перенос embedding и reranker на CPU."
        ),
        hardware={
            "cpu": "AMD Ryzen 9 9950X",
            "ram_gb": 64.0,
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "vram_gb": 24.0,
            "os": "Linux",
            "allow_cpu_offload": True,
        },
    )

    saved = await ostis_gateway.list_all()
    print("=" * 90)
    print(f"Итого уникальных кейсов в БЗ OSTIS: {len(saved)} (SCs-файлы сохранены в ./isolated_ostis_kb/)")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.agent3_llm))