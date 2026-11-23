import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.catalog_adapter import CatalogAdapter
from src.hf_adapter import HuggingFaceSyncService
from src.kb_storage import UnifiedKBStorage

TS_LLM_PATH = PROJECT_ROOT / "data" / "raw" / "llmModels.ts"
TS_EMB_PATH = PROJECT_ROOT / "data" / "raw" / "embeddingModels.ts"
JSON_LLM_PATH = PROJECT_ROOT / "data" / "catalog" / "llmCatalog.json"
JSON_EMB_PATH = PROJECT_ROOT / "data" / "catalog" / "embeddingCatalog.json"


def extract_objects_from_ts(content: str, array_name: str) -> list[dict]:
    """Извлекает массив объектов из экспортируемой TS-переменной."""
    pattern = rf"export const {array_name}[^=]*=\s*(\[[\s\S]*?\]);"
    match = re.search(pattern, content)
    if not match:
        return []

    raw_array = match.group(1)
    cleaned = re.sub(r"new Date\((['\"][^'\"]+['\"])\)", r"\1", raw_array)
    cleaned = re.sub(r"\s+as\s+const", "", cleaned)
    cleaned = re.sub(r"([{,]\s*)([a-zA-Z0-9_]+)\s*:", r'\1"\2":', cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return json.loads(cleaned)


def main() -> None:
    # 1. Конвертация TS -> JSON (если исходные TS-файлы присутствуют)
    if TS_LLM_PATH.exists():
        llms = extract_objects_from_ts(
            TS_LLM_PATH.read_text(encoding="utf-8"), "llmKnowledgeBase"
        )
        JSON_LLM_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_LLM_PATH, "w", encoding="utf-8") as f:
            json.dump(llms, f, ensure_ascii=False, indent=4)
        print(
            f"[1/3] Экспортировано {len(llms)} моделей LLM из TS в {JSON_LLM_PATH.name}"
        )

    if TS_EMB_PATH.exists():
        embeddings = extract_objects_from_ts(
            TS_EMB_PATH.read_text(encoding="utf-8"), "sampleModels"
        )
        JSON_EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(JSON_EMB_PATH, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=4)
        print(
            f"[1/3] Экспортировано {len(embeddings)} моделей Embedding из TS в {JSON_EMB_PATH.name}"
        )

    # 2. Загрузка статических каталогов в единую SQLite-БЗ
    storage = UnifiedKBStorage()
    catalog_models = CatalogAdapter().get_all_catalog_models()
    seeded = storage.upsert_models(catalog_models)
    print(
        f"[2/3] Загружено/проверено моделей из локального каталога в SQLite БЗ: {seeded}"
    )

    # 3. Принудительная синхронизация свежих и популярных моделей с Hugging Face Hub
    print("[3/3] Запуск синхронизации с Hugging Face Hub (downloads + trending)...")
    synced = HuggingFaceSyncService(storage=storage).sync_all(per_query_limit=15)
    print(f"[OK] Синхронизировано моделей с HF Hub: {synced}")
    print(f"Итоговое количество моделей в БЗ по ролям: {storage.count_by_role()}")


if __name__ == "__main__":
    main()
