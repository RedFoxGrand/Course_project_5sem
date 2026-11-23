from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import ModelResearcherAgent
from src.hf_adapter import BackgroundSyncScheduler, HuggingFaceSyncService
from src.kb_storage import UnifiedKBStorage
from src.schemas import EvidenceItem, ModelCandidate, RequirementProfile


def test_offline_query_works_without_network(tmp_path):
    """Агент формирует жизнеспособный триплет RAG из локальной БЗ без доступа к сети."""
    db_file = tmp_path / "test_kb.sqlite3"
    agent = ModelResearcherAgent(db_path=str(db_file), enable_background_sync=False)

    req = RequirementProfile(
        language="ru",
        domain="technical",
        max_context_length=8192,
        privacy_required=True,
    )
    candidates, traces, is_viable = agent.collect_candidates(req)

    assert is_viable is True
    roles = {c.role for c in candidates}
    assert roles == {"llm", "embedding", "reranker"}
    assert any(t.step == "kb_status_check" for t in traces)


def test_catalog_priority_protection(tmp_path):
    """Синхронизация с HF Hub не перезаписывает проверенные модели из статического каталога."""
    storage = UnifiedKBStorage(tmp_path / "priority_kb.sqlite3")

    catalog_model = ModelCandidate(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        name="Qwen2.5-7B-Instruct",
        organization="Qwen",
        role="llm",
        license="Apache-2.0",
        languages=["ru", "en"],
        context_length=EvidenceItem(
            value=32768, source="catalog:llmCatalog", status="catalog"
        ),
        parameters_count_b=EvidenceItem(
            value=7.6, source="catalog:llmCatalog", status="catalog"
        ),
        source_url="https://huggingface.co/Qwen/Qwen2.5-7B-Instruct",
    )
    hf_overwrite_attempt = catalog_model.model_copy(
        update={
            "context_length": EvidenceItem(value=None, source="hf", status="unknown"),
            "parameters_count_b": EvidenceItem(
                value=None, source="hf", status="unknown"
            ),
        }
    )

    storage.upsert_models([catalog_model])
    storage.upsert_models([hf_overwrite_attempt])

    result = storage.query_models(
        "llm", RequirementProfile(language="ru", max_context_length=8192)
    )
    assert len(result) == 1
    assert result[0].context_length.value == 32768
    assert result[0].context_length.status == "catalog"


def test_hf_sync_network_failure_preserves_kb(tmp_path):
    """При обрыве интернета фоновая синхронизация переходит в offline_fallback без потери данных."""
    storage = UnifiedKBStorage(tmp_path / "offline_sync.sqlite3")
    sync_service = HuggingFaceSyncService(storage=storage)

    with patch.object(
        sync_service.api, "list_models", side_effect=ConnectionError("No internet")
    ):
        synced_count = sync_service.sync_all(per_query_limit=5)

    assert synced_count == 0
    meta = storage.get_sync_metadata()
    assert meta is not None
    assert meta["status"] == "offline_fallback"
    assert meta["last_sync_at"] is None


def test_background_scheduler_24h_interval(tmp_path):
    """Фоновый планировщик запускает обновление только по истечении 24 часов."""
    storage = UnifiedKBStorage(tmp_path / "scheduler.sqlite3")
    sync_service = HuggingFaceSyncService(storage=storage)
    scheduler = BackgroundSyncScheduler(sync_service=sync_service, interval_hours=24.0)

    assert scheduler.is_sync_due() is True

    storage.update_sync_metadata(status="success", models_synced=10)
    assert scheduler.is_sync_due() is False

    old_time = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    with storage._connect() as conn:
        conn.execute(
            "UPDATE sync_metadata SET last_sync_at = ? WHERE id = 1", (old_time,)
        )
        conn.commit()

    assert scheduler.is_sync_due() is True


def test_triplet_integrity_failure_on_impossible_context(tmp_path):
    """При невыполнимых требованиях контур помечается нежизнеспособным с диагностикой."""
    agent = ModelResearcherAgent(
        db_path=str(tmp_path / "impossible.sqlite3"), enable_background_sync=False
    )
    req = RequirementProfile(language="ru", max_context_length=10**12)

    _, traces, is_viable = agent.collect_candidates(req)
    assert is_viable is False
    assert any("ошибка контура" in t.details.lower() for t in traces)


def test_missing_metadata_stored_as_unknown_not_faked(tmp_path):
    """Если параметры и контекст не указаны в карточке и config.json, фиксируется None и статус 'unknown'."""
    storage = UnifiedKBStorage(tmp_path / "unknown_meta.sqlite3")
    sync_service = HuggingFaceSyncService(storage=storage)

    raw_hf_model = SimpleNamespace(
        id="org/custom-undocumented-model",
        tags=["text-generation", "languages:ru"],
        card_data=SimpleNamespace(description="Новая модель без указания весов и окна"),
        safetensors=None,
    )

    with patch.object(sync_service, "_fetch_config_context", return_value=None):
        parsed = sync_service._parse_hf_model(raw_hf_model, role="llm")

    assert parsed is not None
    assert parsed.parameters_count_b.value is None
    assert parsed.parameters_count_b.status == "unknown"
    assert parsed.context_length.value is None
    assert parsed.context_length.status == "unknown"
    assert len(parsed.warnings) == 2

    storage.upsert_models([parsed])

    # При строгом требовании к контексту модель с неизвестным окном не выдаётся как подтверждённая
    strict_res = storage.query_models(
        "llm",
        RequirementProfile(language="ru", max_context_length=8192),
        strict_context=True,
    )
    assert len(strict_res) == 0

    # При нестрогом поиске она возвращается с флагом is_context_sufficient=False
    relaxed_res = storage.query_models(
        "llm",
        RequirementProfile(language="ru", max_context_length=8192),
        strict_context=False,
    )
    assert len(relaxed_res) == 1
    assert relaxed_res[0].is_context_sufficient is False
