import json
import re
from pathlib import Path
from typing import Any

from .kb_storage import check_domain_match
from .schemas import (
    EngineType,
    EvidenceItem,
    ModelCandidate,
    ProvenanceStatus,
    RequirementProfile,
)

CURRENT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_LLM_PATH = CURRENT_DIR / "data" / "catalog" / "llmCatalog.json"
DEFAULT_EMB_PATH = CURRENT_DIR / "data" / "catalog" / "embeddingCatalog.json"

DEFAULT_RERANKER_CATALOG: list[dict[str, Any]] = [
    {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "name": "bge-reranker-v2-m3",
        "organization": "BAAI",
        "license": "Apache 2.0",
        "languages": ["ru", "en"],
        "context_length": 8192,
        "parameters_b": 0.568,
        "mteb_rerank": 69.2,
        "specialization": "Мультиязычный cross-encoder реранкер для гибридного RAG",
        "source_url": "https://huggingface.co/BAAI/bge-reranker-v2-m3",
    },
    {
        "model_id": "Alibaba-NLP/gte-multilingual-reranker-base",
        "name": "gte-multilingual-reranker-base",
        "organization": "Alibaba",
        "license": "Apache 2.0",
        "languages": ["ru", "en"],
        "context_length": 8192,
        "parameters_b": 0.306,
        "mteb_rerank": 67.4,
        "specialization": "Легковесный мультиязычный реранкер с поддержкой длинного контекста",
        "source_url": "https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base",
    },
    {
        "model_id": "BAAI/bge-reranker-v2-gemma",
        "name": "bge-reranker-v2-gemma",
        "organization": "BAAI",
        "license": "Gemma License",
        "languages": ["ru", "en"],
        "context_length": 8192,
        "parameters_b": 2.51,
        "mteb_rerank": 71.5,
        "specialization": "Тяжелый LLM-based реранкер максимальной точности",
        "source_url": "https://huggingface.co/BAAI/bge-reranker-v2-gemma",
    },
    {
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "name": "jina-reranker-v2-base-multilingual",
        "organization": "Jina AI",
        "license": "CC BY-NC 4.0",
        "languages": ["ru", "en"],
        "context_length": 1024,
        "parameters_b": 0.278,
        "mteb_rerank": 66.8,
        "specialization": "Высокоскоростной реранкер с поддержкой русского и английского языков",
        "source_url": "https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual",
    },
    {
        "model_id": "BAAI/bge-reranker-large",
        "name": "bge-reranker-large",
        "organization": "BAAI",
        "license": "MIT",
        "languages": ["ru", "en"],
        "context_length": 512,
        "parameters_b": 0.560,
        "mteb_rerank": 67.1,
        "specialization": "Классический надежный cross-encoder для стандартных чанков 512 токенов",
        "source_url": "https://huggingface.co/BAAI/bge-reranker-large",
    },
]


def parse_context_tokens(ctx_raw: Any) -> int:
    """Нормализует строковые представления контекста в целое число токенов."""
    if isinstance(ctx_raw, (int, float)):
        return int(ctx_raw)
    ctx_str = str(ctx_raw).strip().upper()
    base_part = ctx_str.split("(")[0].replace(" ", "").replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*([KM]?)", base_part)
    if not match:
        return 8192
    val = float(match.group(1))
    unit = match.group(2)
    return (
        int(val * 1024 * 1024)
        if unit == "M"
        else (int(val * 1024) if unit == "K" else int(val))
    )


def parse_llm_size(size_str: str) -> tuple[float | None, float | None]:
    """Извлекает общее и активное число параметров в млрд (B) для плотных и MoE-моделей."""
    if not size_str or "—" in size_str or size_str.strip() == "-":
        return None, None
    parts = [p.strip() for p in size_str.split("/")]

    def _to_b(part: str) -> float | None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([TB])", part, re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        return val * 1000.0 if m.group(2).upper() == "T" else val

    total = _to_b(parts[0])
    active = _to_b(parts[1]) if len(parts) > 1 else None
    return total, active


def parse_dimension(dim_raw: Any) -> int | None:
    """Извлекает максимальную размерность эмбеддинга из строк вида 3584.0, 32 (до 1024)."""
    if not dim_raw:
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", str(dim_raw))
    if not numbers:
        return None
    return int(max(float(n) for n in numbers))


def parse_languages(lang_str: str) -> list[str]:
    """Приводит описания языков к ISO 639-1."""
    s = lang_str.lower()
    langs: list[str] = []
    if any(k in s for k in ["мульти", "высокая", "средняя"]):
        langs.extend(["ru", "en"])
    if ("ru" in s or "русск" in s) and "ru" not in langs:
        langs.append("ru")
    if ("en" in s or "английск" in s) and "en" not in langs:
        langs.append("en")
    if any(
        k in s
        for k in ["только en", "нет (только en)", "низкая (en)", "только английский"]
    ):
        langs = ["en"]
    if "только ru" in s or "только русский" in s:
        langs = ["ru"]
    return langs if langs else ["ru", "en"]


def estimate_embedding_params_b(
    name: str, is_cloud: bool
) -> tuple[float | None, ProvenanceStatus]:
    """Восстанавливает размер параметров векторизаторов по архитектурному семейству."""
    if is_cloud:
        return None, "unknown"
    name_l = name.lower()
    m_b = re.search(r"(\d+(?:\.\d+)?)b", name_l)
    if m_b:
        return float(m_b.group(1)), "declared"
    m_m = re.search(r"(\d+(?:\.\d+)?)m", name_l)
    if m_m:
        return round(float(m_m.group(1)) / 1000.0, 3), "declared"
    if "gemma2" in name_l:
        return 9.24, "estimated"
    if any(k in name_l for k in ["bge-m3", "jina-embeddings-v3", "e5-large"]):
        return 0.565, "catalog"
    if "large" in name_l:
        return 0.335, "estimated"
    return 0.278 if ("base" in name_l or "mpnet" in name_l) else 0.120, "estimated"


class CatalogAdapter:
    """Адаптер статического каталога 4-го семестра для наполнения единой Базы Знаний."""

    def __init__(
        self,
        llm_path: Path | str = DEFAULT_LLM_PATH,
        emb_path: Path | str = DEFAULT_EMB_PATH,
    ):
        self._raw_llm = self._read_json(Path(llm_path))
        self._raw_emb = self._read_json(Path(emb_path))
        self._raw_rerankers = DEFAULT_RERANKER_CATALOG

    @staticmethod
    def _read_json(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def normalize_llm(
        self, raw: dict[str, Any], req: RequirementProfile | None = None
    ) -> ModelCandidate:
        req = req or RequirementProfile()
        warnings: list[str] = []
        license_str = raw.get("license", "Проприетарная")
        hw_str = raw.get("hardware", "")
        source_url = (
            raw.get("sourceUrl") or f"https://huggingface.co/{raw.get('id', 'model')}"
        )
        is_prop = (
            "проприетар" in license_str.lower()
            or "облачный api" in hw_str.lower()
            or "openrouter.ai" in source_url.lower()
        )

        total_b, active_b = parse_llm_size(raw.get("size", "—"))
        if total_b is not None:
            params_item = EvidenceItem(
                value=total_b, source="catalog:llmCatalog:size", status="catalog"
            )
        else:
            params_item = EvidenceItem(
                value=None, source="catalog:llmCatalog:size", status="unknown"
            )
            warnings.append(
                "данные о количестве параметров не указаны (закрытый облачный API)"
            )

        active_item = (
            EvidenceItem(
                value=active_b,
                source="catalog:llmCatalog:size_active",
                status="catalog",
            )
            if active_b
            else None
        )

        ctx_tokens = parse_context_tokens(raw.get("context", "8192"))
        ctx_item = EvidenceItem(
            value=ctx_tokens, source="catalog:llmCatalog:context", status="catalog"
        )
        is_ctx_ok = ctx_tokens >= req.max_context_length

        elo_m = re.search(r"\d+(?:\.\d+)?", str(raw.get("elo", "")))
        quality_item = (
            EvidenceItem(
                value=float(elo_m.group(0)),
                source="catalog:llmCatalog:elo",
                status="benchmark",
            )
            if elo_m
            else None
        )

        spec = raw.get("specialization", "")
        domain_match = check_domain_match(req.domain, f"{raw.get('name', '')} {spec}")

        formats = [] if is_prop else ["safetensors", "gguf"]
        quants = ["Q4_K_M", "Q8_0", "FP16"] if not is_prop else []
        engines: list[EngineType] = (
            [] if is_prop else ["vllm", "sglang", "llama.cpp", "ollama"]
        )

        source_url = (
            raw.get("sourceUrl") or f"https://huggingface.co/{raw.get('id', 'model')}"
        )
        model_id = (
            source_url.split("huggingface.co/")[-1].strip("/")
            if "huggingface.co/" in source_url
            else raw.get("id", "unknown")
        )

        return ModelCandidate(
            model_id=model_id,
            name=raw.get("name", model_id),
            organization=raw.get("developer", "Unknown"),
            role="llm",
            license=license_str,
            languages=parse_languages(raw.get("languages", "")),
            context_length=ctx_item,
            parameters_count_b=params_item,
            active_parameters_count_b=active_item,
            is_local_compatible=not is_prop,
            is_context_sufficient=is_ctx_ok,
            supported_formats=formats,
            supported_quantizations=quants,
            compatible_engines=engines,
            quality_score=quality_item,
            specialization=spec,
            domain_match=domain_match,
            warnings=warnings,
            source_url=source_url,
        )

    def normalize_embedding(
        self, raw: dict[str, Any], req: RequirementProfile | None = None
    ) -> ModelCandidate:
        req = req or RequirementProfile()
        warnings: list[str] = []
        name = raw.get("name", "embedding")
        license_str = raw.get("license", "Unknown")
        hw_str = str(raw.get("hardwareRequirements", ""))
        is_cloud = "api" in license_str.lower() or "неприменимо" in hw_str.lower()

        params_b, params_st = estimate_embedding_params_b(name, is_cloud)
        if params_st == "unknown":
            warnings.append("requires_verification: закрытый API без открытых весов")
        elif params_st == "estimated":
            warnings.append(
                "requires_verification: количество параметров оценено по архитектуре энкодера"
            )

        params_item = EvidenceItem(
            value=params_b, source="catalog:embeddingCatalog:params", status=params_st
        )

        ctx_tokens = parse_context_tokens(raw.get("contextWindow", 512))
        ctx_item = EvidenceItem(
            value=ctx_tokens,
            source="catalog:embeddingCatalog:contextWindow",
            status="catalog",
        )
        is_ctx_ok = ctx_tokens >= min(req.max_context_length, 8192)

        dim_val = parse_dimension(raw.get("dimension"))
        dim_item = (
            EvidenceItem(
                value=dim_val,
                source="catalog:embeddingCatalog:dimension",
                status="catalog",
            )
            if dim_val
            else None
        )
        if dim_val is None:
            warnings.append("requires_verification: размерность вектора не указана")

        mteb_m = re.search(r"\d+(?:\.\d+)?", str(raw.get("mtebScore", "")))
        quality_item = (
            EvidenceItem(
                value=float(mteb_m.group(0)),
                source="catalog:embeddingCatalog:mteb",
                status="benchmark",
            )
            if (mteb_m and "N/A" not in str(raw.get("mtebScore", "")))
            else None
        )

        engines: list[EngineType] = (
            []
            if is_cloud
            else (["tei", "ollama", "vllm"] if params_b >= 4.0 else ["tei", "ollama"])
        )

        link = raw.get("link", "")
        model_id = (
            link.split("huggingface.co/")[-1].strip("/")
            if "huggingface.co/" in link
            else f"{raw.get('family', 'org')}/{name}".replace(" ", "_")
        )

        return ModelCandidate(
            model_id=model_id,
            name=name,
            organization=raw.get("family", "Unknown"),
            role="embedding",
            license=license_str,
            languages=parse_languages(raw.get("multilingual", "")),
            context_length=ctx_item,
            parameters_count_b=params_item,
            embedding_dimension=dim_item,
            is_local_compatible=not is_cloud,
            is_context_sufficient=is_ctx_ok,
            supported_formats=[] if is_cloud else ["safetensors", "onnx"],
            supported_quantizations=["FP16", "INT8"] if not is_cloud else [],
            compatible_engines=engines,
            quality_score=quality_item,
            specialization=f"Эмбеддинг (dim={dim_val or 'unknown'})",
            domain_match=True,
            warnings=warnings,
            source_url=link
            if link.startswith("http")
            else f"https://huggingface.co/{model_id}",
        )

    def normalize_reranker(
        self, raw: dict[str, Any], req: RequirementProfile | None = None
    ) -> ModelCandidate:
        req = req or RequirementProfile()
        is_ctx_ok = raw["context_length"] >= min(req.max_context_length, 512)

        return ModelCandidate(
            model_id=raw["model_id"],
            name=raw["name"],
            organization=raw["organization"],
            role="reranker",
            license=raw["license"],
            languages=raw["languages"],
            context_length=EvidenceItem(
                value=raw["context_length"],
                source="catalog:rerankers:context",
                status="catalog",
            ),
            parameters_count_b=EvidenceItem(
                value=raw["parameters_b"],
                source="catalog:rerankers:params",
                status="catalog",
            ),
            is_local_compatible=True,
            is_context_sufficient=is_ctx_ok,
            supported_formats=["safetensors", "onnx"],
            supported_quantizations=["FP16", "INT8"],
            compatible_engines=["tei", "vllm"]
            if raw["parameters_b"] > 1.5
            else ["tei"],
            quality_score=EvidenceItem(
                value=raw["mteb_rerank"],
                source="catalog:rerankers:mteb",
                status="benchmark",
            ),
            specialization=raw.get("specialization", "Cross-encoder"),
            domain_match=True,
            warnings=[],
            source_url=raw["source_url"],
        )

    def get_all_catalog_models(self) -> list[ModelCandidate]:
        """Возвращает полный нормализованный список эталонных моделей из локальных каталогов."""
        models: list[ModelCandidate] = []
        models.extend(self.normalize_llm(i) for i in self._raw_llm)
        models.extend(self.normalize_embedding(i) for i in self._raw_emb)
        models.extend(self.normalize_reranker(i) for i in self._raw_rerankers)
        return models
