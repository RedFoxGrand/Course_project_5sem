import asyncio
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.agent import collect_candidates_node
from src.schemas import RequirementProfile

DEFAULT_CANDIDATES_PATH = BASE_DIR / "mock_candidates.json"


async def main() -> None:
    requirements = RequirementProfile(
        language="ru",
        domain="technical",
        max_context_length=8192,
        privacy_required=True,
    )

    initial_state = {"requirements": requirements, "trace": []}
    result_state = await collect_candidates_node(initial_state)

    candidates = result_state["candidates"]
    traces = result_state["trace"]
    is_viable = result_state["is_pipeline_viable"]

    print("-" * 98)
    print(f"Агент 2: собрано кандидатов — {len(candidates)}")
    print(
        f"Статус жизнеспособности RAG-тройки: {'готова к сборке' if is_viable else 'невозможно собрать'}"
    )
    print("-" * 98)

    if candidates:
        print(
            f"{'Роль':<10} | {'Модель (ID)':<42} | {'Параметры':<13} | {'Контекст':<12} | {'Статус'}"
        )
        print("-" * 98)
        for c in candidates:
            params_str = (
                f"{c.parameters_count_b.value}B"
                if c.parameters_count_b.value is not None
                else "не указано"
            )
            ctx_str = (
                f"{c.context_length.value}"
                if c.context_length.value is not None
                else "не указано"
            )
            status_str = c.parameters_count_b.status
            print(
                f"{c.role.upper():<10} | {c.model_id[:42]:<42} | {params_str:<13} | {ctx_str:<12} | {status_str}"
            )

    print("\nЖурнал трассировки:")
    for t in traces:
        print(f" -> [{t['step']}] {t['details']}")

    payload_str = json.dumps(
        [c.model_dump() for c in candidates], ensure_ascii=False, indent=4
    )
    await asyncio.to_thread(
        DEFAULT_CANDIDATES_PATH.write_text, payload_str, encoding="utf-8"
    )
    print(f"\n[OK] Результат сохранен в {DEFAULT_CANDIDATES_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
