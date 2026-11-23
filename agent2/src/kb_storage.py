import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .schemas import ModelCandidate, RequirementProfile, RoleType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "kb"
DEFAULT_KB_PATH = KB_DIR / "knowledge_base.sqlite3"


def check_domain_match(domain: str, text_corpus: str) -> bool:
    """Проверяет соответствие модели предметному домену запроса."""
    if not domain or domain.lower() == "general":
        return True
    domain_keywords = {
        "code": ["code", "coder", "программирован", "ide", "swe-bench", "разработк"],
        "technical": ["научн", "техническ", "рассужден", "математик", "логик"],
        "medical": ["медицин", "health", "клиническ"],
    }
    keywords = domain_keywords.get(domain.lower(), [domain.lower()])
    corpus_l = text_corpus.lower()
    return any(k in corpus_l for k in keywords)


class UnifiedKBStorage:
    """Единая персистентная База Знаний (SQLite WAL) с фоновым обновлением и offline-доступом."""

    def __init__(self, db_path: Path | str = DEFAULT_KB_PATH):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            cols = {
                row["name"]: row["notnull"]
                for row in conn.execute("PRAGMA table_info(models_kb)").fetchall()
            }
            if cols.get("context_length") == 1 or cols.get("parameters_count_b") == 1:
                conn.execute("DROP TABLE IF EXISTS models_kb")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS models_kb (
                    model_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    is_local_compatible INTEGER NOT NULL,
                    context_length INTEGER,
                    parameters_count_b REAL,
                    provenance_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_models_role_ctx ON models_kb(role, context_length);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_sync_at TEXT,
                    status TEXT NOT NULL,
                    models_synced INTEGER DEFAULT 0,
                    last_error TEXT
                )
                """
            )
            conn.commit()

    def upsert_models(self, models: list[ModelCandidate]) -> int:
        """Добавляет новые или обновляет существующие модели с защитой проверенного каталога."""
        if not models:
            return 0

        now_iso = datetime.now(UTC).isoformat()
        affected = 0

        with self._connect() as conn:
            for m in models:
                existing = conn.execute(
                    "SELECT provenance_status FROM models_kb WHERE model_id = ?",
                    (m.model_id,),
                ).fetchone()

                if (
                    existing
                    and existing["provenance_status"] == "catalog"
                    and m.context_length.status != "catalog"
                ):
                    continue

                ctx_db = (
                    int(m.context_length.value)
                    if m.context_length.value is not None
                    else None
                )
                params_db = (
                    float(m.parameters_count_b.value)
                    if m.parameters_count_b.value is not None
                    else None
                )

                conn.execute(
                    """
                    INSERT INTO models_kb (
                        model_id, role, is_local_compatible, context_length,
                        parameters_count_b, provenance_status, payload_json, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        role = excluded.role,
                        is_local_compatible = excluded.is_local_compatible,
                        context_length = excluded.context_length,
                        parameters_count_b = excluded.parameters_count_b,
                        provenance_status = excluded.provenance_status,
                        payload_json = excluded.payload_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        m.model_id,
                        m.role,
                        1 if m.is_local_compatible else 0,
                        ctx_db,
                        params_db,
                        m.context_length.status,
                        m.model_dump_json(),
                        now_iso,
                    ),
                )
                affected += 1
            conn.commit()
        return affected

    def query_models(
        self,
        role: RoleType,
        req: RequirementProfile,
        strict_context: bool = True,
        strict_domain: bool = False,
    ) -> list[ModelCandidate]:
        """Мгновенный поиск и фильтрация кандидатов по актуальной локальной БЗ."""
        if role == "llm":
            target_min_ctx = req.max_context_length
        elif role == "embedding":
            target_min_ctx = min(req.max_context_length, 8192)
        else:
            target_min_ctx = min(req.max_context_length, 512)

        sql = "SELECT payload_json FROM models_kb WHERE role = ?"
        params: list[Any] = [role]

        if req.privacy_required:
            sql += " AND is_local_compatible = 1"
        if strict_context:
            sql += " AND context_length IS NOT NULL AND context_length >= ?"
            params.append(target_min_ctx)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()

        filtered: list[ModelCandidate] = []
        lang_req = req.language.lower().strip() if req.language else ""

        for row in rows:
            c = ModelCandidate.model_validate_json(row["payload_json"])

            if lang_req:
                model_langs = [l.lower() for l in c.languages]
                if lang_req not in model_langs and "multilingual" not in model_langs:
                    continue

            c.domain_match = check_domain_match(
                req.domain, f"{c.name} {c.specialization}"
            )
            if strict_domain and req.domain != "general" and not c.domain_match:
                continue

            c.is_context_sufficient = (
                c.context_length.value is not None
                and int(c.context_length.value) >= target_min_ctx
            )
            filtered.append(c)

        filtered.sort(
            key=lambda x: (
                1 if x.domain_match else 0,
                1 if x.context_length.status == "catalog" else 0,
                float(x.quality_score.value)
                if x.quality_score and isinstance(x.quality_score.value, (int, float))
                else 0.0,
                int(x.context_length.value)
                if x.context_length.value is not None
                else -1,
            ),
            reverse=True,
        )
        return filtered

    def prune_stale_hf_models(self, max_age_days: int = 60) -> int:
        """Удаляет устаревшие модели из HF Hub, не обновлявшиеся дольше max_age_days (каталог не трогает)."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM models_kb
                WHERE provenance_status != 'catalog'
                  AND datetime(updated_at) < datetime('now', ?)
                """,
                (f"-{max_age_days} days",),
            )
            conn.commit()
            return cur.rowcount

    def get_sync_metadata(self) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sync_metadata WHERE id = 1").fetchone()
            return dict(row) if row else None

    def update_sync_metadata(
        self, status: str, models_synced: int = 0, last_error: str | None = None
    ) -> None:
        now_iso = datetime.now(UTC).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_metadata (id, last_sync_at, status, models_synced, last_error)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    last_sync_at = CASE
                        WHEN excluded.status = 'success' THEN excluded.last_sync_at
                        ELSE sync_metadata.last_sync_at
                    END,
                    status = excluded.status,
                    models_synced = excluded.models_synced,
                    last_error = excluded.last_error
                """,
                (
                    now_iso if status == "success" else None,
                    status,
                    models_synced,
                    last_error,
                ),
            )
            conn.commit()

    def count_by_role(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, COUNT(*) AS cnt FROM models_kb GROUP BY role"
            ).fetchall()
        counts = {"llm": 0, "embedding": 0, "reranker": 0}
        for r in rows:
            counts[r["role"]] = int(r["cnt"])
        return counts
