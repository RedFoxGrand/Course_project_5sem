import asyncio
from orchestrator.graph import build_master_workflow


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


async def main() -> None:
    workflow, ostis_gateway = build_master_workflow(
        agent3_llm=None,  # Можно передать ChatOpenAI(...) для LLM-ранжирования
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
    asyncio.run(main())