import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ModelResearcherAgent
from src.schemas import RequirementProfile

SCENARIOS: list[tuple[str, RequirementProfile]] = [
    (
        "1. Кодогенерация (EN, 32K, только локальные)",
        RequirementProfile(
            language="en",
            domain="code",
            max_context_length=32768,
            privacy_required=True,
        ),
    ),
    (
        "2. Экстремальный контекст 1M токенов (RU, локальные)",
        RequirementProfile(
            language="ru",
            domain="general",
            max_context_length=1_000_000,
            privacy_required=True,
        ),
    ),
    (
        "3. Гибридный режим с облачными API (RU, 131K, privacy=False)",
        RequirementProfile(
            language="ru",
            domain="general",
            max_context_length=131072,
            privacy_required=False,
        ),
    ),
    (
        "4. Редкий язык через тег multilingual (DE, 8K, локальные)",
        RequirementProfile(
            language="de",
            domain="general",
            max_context_length=8192,
            privacy_required=True,
        ),
    ),
    (
        "5. Невыполнимый контекст 10^12 (проверка разрыва триплета)",
        RequirementProfile(
            language="ru",
            domain="general",
            max_context_length=10**12,
            privacy_required=True,
        ),
    ),
]


def main() -> None:
    # Отключаем фоновый поток на время замера скорости локального поиска
    agent = ModelResearcherAgent(enable_background_sync=False)
    report: dict[str, object] = {}

    for title, req in SCENARIOS:
        t0 = time.perf_counter()
        candidates, traces, is_viable = agent.collect_candidates(req)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        by_role = {
            r: [c for c in candidates if c.role == r]
            for r in ("llm", "embedding", "reranker")
        }

        print("=" * 95)
        print(f"СЦЕНАРИЙ: {title}")
        print(
            f"Жизнеспособность: {is_viable} | Время отклика: {elapsed_ms} мс | "
            f"LLM={len(by_role['llm'])}, Emb={len(by_role['embedding'])}, Rerank={len(by_role['reranker'])}"
        )

        for role in ("llm", "embedding", "reranker"):
            top3 = by_role[role][:3]
            if top3:
                items_str = ", ".join(
                    f"{c.model_id} (ctx={c.context_length.value}, params={c.parameters_count_b.value}, "
                    f"local={c.is_local_compatible}, dom={c.domain_match})"
                    for c in top3
                )
                print(f"  Топ-{role.upper()}: {items_str}")
            else:
                print(f"  Топ-{role.upper()}: [ПУСТО]")

        if not is_viable:
            print(f"  Диагностика: {traces[-1].details}")

        report[title] = {
            "is_pipeline_viable": is_viable,
            "elapsed_ms": elapsed_ms,
            "counts": {r: len(by_role[r]) for r in ("llm", "embedding", "reranker")},
            "trace": [t.model_dump() for t in traces],
            "top_candidates": [c.model_dump() for c in candidates],
        }

    out_path = PROJECT_ROOT / "scenarios_report.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 95)
    print(f"[OK] Полный отчёт сохранён в {out_path}")


if __name__ == "__main__":
    main()
