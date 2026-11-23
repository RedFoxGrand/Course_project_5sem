import json
import logging
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import HfApi, hf_hub_download

from .kb_storage import UnifiedKBStorage
from .llm_extractor import FreeTextCardExtractor
from .schemas import (
    EngineType,
    EvidenceItem,
    ModelCandidate,
    ProvenanceStatus,
    RoleType,
)

logger = logging.getLogger(__name__)

PIPELINE_MAP: dict[RoleType, str] = {
    "llm": "text-generation",
    "embedding": "feature-extraction",
    "reranker": "text-classification",
}


class HuggingFaceSyncService:
    """Сервис периодической синхронизации открытых моделей с Hugging Face Hub в единую БЗ."""

    def __init__(
        self,
        storage: UnifiedKBStorage | None = None,
        llm_instance: Any = None,
        timeout_seconds: float = 8.0,
    ):
        self.api = HfApi()
        self.storage = storage or UnifiedKBStorage()
        self.timeout = timeout_seconds
        self.free_text_extractor = FreeTextCardExtractor(llm_instance=llm_instance)

    def _execute_hf_query(
        self,
        tag_filters: list[str],
        search_query: str | None,
        limit: int,
        sort_by: str = "downloads",
    ) -> list[Any]:
        try:
            return list(
                self.api.list_models(
                    filter=tag_filters,
                    search=search_query,
                    sort=sort_by,
                    limit=limit,
                    cardData=True,
                    full=True,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[HF Sync Warning] Запрос к HF Hub (%s, sort=%s) завершился ошибкой: %s",
                tag_filters,
                sort_by,
                e,
            )
            return []

    @staticmethod
    def _fetch_config_context(model_id: str) -> int | None:
        """Пытается извлечь реальную длину контекста из файла config.json репозитория."""
        try:
            cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:  # noqa: BLE001
            return None

        if not isinstance(cfg, dict):
            return None

        ctx_keys = (
            "max_position_embeddings",
            "n_positions",
            "seq_length",
            "max_seq_len",
            "model_max_length",
        )
        sources = [cfg]
        if isinstance(cfg.get("text_config"), dict):
            sources.append(cfg["text_config"])

        for src in sources:
            for key in ctx_keys:
                val = src.get(key)
                if isinstance(val, int) and 128 <= val <= 10_000_000:
                    return val
        return None

    def sync_all(
        self, per_query_limit: int = 15, prune_older_than_days: int = 60
    ) -> int:
        """
        Выполняет полный цикл синхронизации популярнейших и свежих (last_modified) моделей
        по ролям LLM, Embedding и Reranker в локальную SQLite-БЗ.
        """
        queries: list[tuple[RoleType, list[str], str | None]] = [
            ("llm", ["text-generation", "languages:ru"], None),
            ("llm", ["text-generation", "multilingual"], None),
            ("llm", ["text-generation", "languages:en"], "code"),
            ("embedding", ["feature-extraction", "multilingual"], None),
            ("embedding", ["feature-extraction", "languages:ru"], None),
            ("reranker", ["text-classification"], "rerank"),
        ]

        collected: dict[str, ModelCandidate] = {}
        had_network_success = False

        for role, tags, search_q in queries:
            for sort_by in ("downloads", "last_modified"):
                raw_models = self._execute_hf_query(
                    tag_filters=tags,
                    search_query=search_q,
                    limit=per_query_limit,
                    sort_by=sort_by,
                )
                if raw_models:
                    had_network_success = True
                for raw_m in raw_models:
                    parsed = self._parse_hf_model(raw_m, role)
                    if parsed and parsed.model_id not in collected:
                        collected[parsed.model_id] = parsed

        if not had_network_success and not collected:
            self.storage.update_sync_metadata(
                status="offline_fallback",
                models_synced=0,
                last_error="Сеть недоступна или превышен таймаут HF Hub API",
            )
            return 0

        synced_count = self.storage.upsert_models(list(collected.values()))
        self.storage.prune_stale_hf_models(max_age_days=prune_older_than_days)
        self.storage.update_sync_metadata(
            status="success", models_synced=synced_count, last_error=None
        )
        return synced_count

    def _parse_hf_model(self, model: Any, role: RoleType) -> ModelCandidate | None:
        model_id = getattr(model, "id", None)
        if not model_id or not isinstance(model_id, str):
            return None

        tags: list[str] = getattr(model, "tags", None) or []
        tags_lower = [t.lower() for t in tags]
        model_id_lower = model_id.lower()

        excluded_tags = {
            "ocr",
            "image-to-text",
            "image-text-to-text",
            "visual-question-answering",
            "audio-to-text",
            "automatic-speech-recognition",
            "text-to-speech",
            "image-classification",
            "object-detection",
        }
        excluded_substrings = [
            "ocr",
            "tinystories",
            "internvl",
            "siglip",
            "clip-",
            "caption",
            "vision",
            "whisper",
            "codec",
            "speech",
        ]
        if any(t in tags_lower for t in excluded_tags) or any(
            x in model_id_lower for x in excluded_substrings
        ):
            return None

        if role != "reranker" and "rerank" in model_id_lower:
            return None

        card_data = (
            getattr(model, "card_data", None) or getattr(model, "cardData", None) or {}
        )
        warnings: list[str] = []
        source_url = f"https://huggingface.co/{model_id}"

        safetensors_info = getattr(model, "safetensors", None)
        st_total = (
            getattr(safetensors_info, "total", None) if safetensors_info else None
        )
        if isinstance(safetensors_info, dict):
            st_total = safetensors_info.get("total")

        param_match_b = re.search(r"(\d+(?:\.\d+)?)b", model_id_lower)
        param_match_m = re.search(r"(\d+(?:\.\d+)?)m(?![a-z])", model_id_lower)
        has_b_weight = param_match_b is not None

        if isinstance(st_total, (int, float)) and st_total > 0:
            params_val: float | None = round(float(st_total) / 1_000_000_000, 3)
            params_status: ProvenanceStatus = "declared"
        elif param_match_b:
            params_val = float(param_match_b.group(1))
            params_status = "declared"
        elif param_match_m and not any(
            k in model_id_lower for k in ["context", "ctx", "1m"]
        ):
            params_val = round(float(param_match_m.group(1)) / 1000.0, 4)
            params_status = "declared"
        else:
            params_val = None
            params_status = "unknown"
            warnings.append(
                "данные о количестве параметров не указаны в карточке модели"
            )

        if role == "llm" and params_val is not None and params_val < 0.5:
            return None

        raw_ctx = getattr(card_data, "max_position_embeddings", None)
        ctx_source = source_url
        if isinstance(raw_ctx, (int, float)) and raw_ctx > 0:
            ctx_val: int | None = int(raw_ctx)
            ctx_status: ProvenanceStatus = "declared"
        elif (
            has_b_weight
            and re.search(r"[-_](?:1m|1000k|1048k)(?:[-_]|$)", model_id_lower)
        ) or re.search(
            r"(?:context[-_]?1m|1m[-_]?context|long[-_]?1m)", model_id_lower
        ):
            ctx_val = 1048576
            ctx_status = "declared"
        elif re.search(r"[-_](?:512k|256k|128k|32k)(?:[-_]|$)", model_id_lower):
            val_k = int(re.search(r"[-_](\d+)k", model_id_lower).group(1))  # type: ignore[union-attr]
            ctx_val = val_k * 1024
            ctx_status = "declared"
        else:
            cfg_ctx = self._fetch_config_context(model_id)
            if cfg_ctx is not None:
                ctx_val = cfg_ctx
                ctx_status = "declared"
                ctx_source = f"{source_url}/blob/main/config.json"
            else:
                ctx_val = None
                ctx_status = "unknown"
                warnings.append("данные о размере контекстного окна не указаны")

        declared_langs: list[str] = []
        for t in tags:
            if t.startswith(("languages:", "language:")):
                code = t.split(":")[-1].strip().lower()
                if code not in declared_langs:
                    declared_langs.append(code)
        if "multilingual" in tags_lower and "multilingual" not in declared_langs:
            declared_langs.append("multilingual")
        if not declared_langs:
            declared_langs = ["en"]

        raw_description = str(getattr(card_data, "description", "") or " ".join(tags))
        extracted = self.free_text_extractor.extract_from_markdown(raw_description)

        lic = getattr(card_data, "license", None)
        if not lic:
            for t in tags:
                if t.startswith("license:"):
                    lic = t.split(":", 1)[1]
                    break
        license_str = str(
            lic[0]
            if isinstance(lic, list) and lic
            else (lic or extracted.detected_license or "unknown")
        )

        formats = ["safetensors"]
        if "GGUF" in extracted.supported_quantizations or any(
            "gguf" in t for t in tags_lower
        ):
            formats.append("gguf")

        engines: list[EngineType] = list(extracted.compatible_engines)
        if role == "llm" and "vllm" not in engines:
            engines.extend(["vllm", "sglang"])
        elif role in ["embedding", "reranker"] and "tei" not in engines:
            engines.append("tei")

        return ModelCandidate(
            model_id=model_id,
            name=model_id.split("/")[-1],
            organization=model_id.split("/")[0] if "/" in model_id else "Community",
            role=role,
            license=license_str,
            languages=declared_langs,
            context_length=EvidenceItem(
                value=ctx_val, source=ctx_source, status=ctx_status
            ),
            parameters_count_b=EvidenceItem(
                value=params_val, source=source_url, status=params_status
            ),
            is_local_compatible=True,
            is_context_sufficient=ctx_val is not None,
            supported_formats=formats,
            supported_quantizations=extracted.supported_quantizations or ["FP16"],
            compatible_engines=sorted(set(engines)),
            quality_score=(
                EvidenceItem(
                    value=extracted.benchmark_value,
                    source=f"{source_url}#readme",
                    status="benchmark",
                )
                if extracted.benchmark_value
                else None
            ),
            specialization=f"Hugging Face ({PIPELINE_MAP[role]})",
            domain_match=True,
            warnings=warnings,
            source_url=source_url,
        )


class BackgroundSyncScheduler:
    """Фоновый демон автоматического обновления БЗ с периодом раз в 24 часа."""

    def __init__(
        self, sync_service: HuggingFaceSyncService, interval_hours: float = 24.0
    ):
        self.sync_service = sync_service
        self.interval = timedelta(hours=interval_hours)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def is_sync_due(self) -> bool:
        info = self.sync_service.storage.get_sync_metadata()
        if not info or not info.get("last_sync_at"):
            return True
        last_dt = datetime.fromisoformat(str(info["last_sync_at"]))
        return datetime.now(UTC) - last_dt >= self.interval

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="kb-hf-sync-daemon", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.is_sync_due():
                try:
                    self.sync_service.sync_all()
                except Exception as exc:  # noqa: BLE001
                    logger.error("[KB Background Sync Error] %s", exc)
            self._stop_event.wait(timeout=3600.0)
