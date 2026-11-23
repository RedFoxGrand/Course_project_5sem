# Агент 5: Рефлексивный агент долгосрочной памяти OSTIS

Модуль реализует завершающий узел конвейера проектирования (`reflect_and_store_experience`) в интеллектуальной среде **ostis-ann**.

**Основные задачи модуля:**
* приём и валидация финального состояния проектирования (`DesignState`);
* предотвращение дублирования одинаковых решений в базе знаний по хэшу `SHA-256`;
* поиск и связывание похожих проектов (например, при переносе с RTX 5090 на RTX 4090);
* смысловая категоризация задачи через LLM (`Structured Output`) с автоматическим локальным резервом;
* разделение подтверждённых фактов (`confirmed_fact`) и расчётных оценок (`hypothesis` / `requires_verification`);
* трансляция результата (`ReflectionRecord`) в семантическую сеть на языке `SCs` (15 сущностей и 11 отношений онтологии OSTIS) и сохранение через адаптер;
* агрегация сквозной трассы работы агентов (`trace`) и отдача данных для экрана истории в UI.

---

## 1. Структура файлов (что переносить в общий проект)

Модуль разделён на ядро (идёт в продакшен) и симуляторы остальных агентов (нужны только для автономной проверки).

| Файл | Назначение | Нужен при интеграции? |
| :--- | :--- | :---: |
| `agent.py` | Ядро `ReflectionAgent`, категоризаторы LLM, хэширование и анализ гипотез | **Да** |
| `ostis_ontology.py` | Генератор `SCs` (`ScsOntologyGenerator`), интерфейсы `LocalDictionaryOstisGateway` и `HttpOstisGateway` | **Да** |
| `schemas.py` | Валидационные контракты Pydantic v2 (защитный слой с `from_attributes=True, extra="ignore"`) | **Да** *(или слить с общим `schemas.py`)* |
| `agent1_data.py` – `agent4_data.py` | Заглушки, имитирующие словари от Агентов 1–4 для сценариев RTX 5090 / RTX 4090 | **Нет** *(только для тестов)* |
| `workflow.py` | Изолированный граф переходов для автономного прогона | **Нет** *(заменяется общим `LangGraph`)* |
| `app.py` | Локальный FastAPI-сервер и HTML-дашборд для проверки истории и трассы | **Частично** *(взять эндпоинты `/api/ostis/cases` и `/api/trace`)* |

---

## 2. Подключение в общий граф LangGraph

Все модели данных в `schemas.py` настроены с параметром `ConfigDict(from_attributes=True, extra="ignore")`. Это значит, что Агент 5 может принимать как обычные словари `dict`, так и экземпляры Pydantic-моделей от Агентов 1–4 без конфликтов импортов.

### Пример регистрации узла в `StateGraph`

```python
from langgraph.graph import StateGraph, END
from agent import ReflectionAgent, OpenAICompatibleCategorizer
from ostis_ontology import HttpOstisGateway, LocalDictionaryOstisGateway

# 1. Инициализация адаптера БЗ OSTIS
# Для разработки без поднятого sc-machine используйте LocalDictionaryOstisGateway("./ostis_kb")
ostis_gateway = HttpOstisGateway(base_url="http://ostis-adapter:8080", timeout=10.0)

# 2. Инициализация категоризатора (LLM Structured Output)
# Если передать llm_categorizer=None, агент использует локальный детерминированный категоризатор
categorizer = OpenAICompatibleCategorizer(
    base_url="http://llm-proxy:8000/v1",
    api_key="YOUR_API_KEY",
    model="gpt-4o-mini",
)

reflection_agent = ReflectionAgent(
    ostis_gateway=ostis_gateway,
    llm_categorizer=categorizer,
)

# 3. Адаптер-обёртка для узла LangGraph
# Используйте эту обёртку, если в общем DesignState поля components и memory_breakdown
# лежат внутри объекта deployment_plan (от Агента 4)
async def reflect_and_store_experience_node(state: dict) -> dict:
    payload = dict(state)
    plan = state.get("deployment_plan")

    if plan is not None:
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
        payload.setdefault("components", plan_dict.get("configs") or plan_dict.get("components"))
        payload.setdefault("memory_breakdown", plan_dict.get("memory_estimate") or plan_dict.get("memory_breakdown"))
        payload.setdefault("primary_set_id", plan_dict.get("primary_set_id", "set_primary"))
        payload.setdefault("alternative_set_id", plan_dict.get("alternative_set_id"))
        payload.setdefault("decision_rationale", plan_dict.get("explanation", ""))
        payload.setdefault("warnings", plan_dict.get("warnings", []))

    return await reflection_agent.reflect_and_store_experience(payload)

# 4. Добавление узла в общий граф
# workflow.add_node("reflect_and_store_experience", reflect_and_store_experience_node)
# workflow.add_edge("save_project", "reflect_and_store_experience")
# workflow.add_edge("reflect_and_store_experience", END)
```

> **Важно по редьюсеру `trace` в общем `DesignState`:**
> Метод `reflect_and_store_experience` возвращает **полный** обновлённый список `state["trace"]` (включая события предыдущих агентов + свои новые). Если в общем `DesignState` на поле `trace` навешан LangGraph-редьюсер `Annotated[list, operator.add]`, в функции-обёртке узла возвращайте только срез новых событий: `result["trace"][len(state.get("trace", [])):]`, чтобы избежать дублирования строк в трассе.

---

## 3. Контракт входных и выходных данных (`dict`)

### Что Агент 5 читает из `state`

**Обязательные ключи** *(при их отсутствии агент не падает, а пишет ошибку в `state["errors"]` и `state["trace"]`)*:
* `requirements` (`dict` / `RequirementProfile`): `language`, `domain`, `max_context_length`, `parallel_sessions`, `privacy_required`, `priority`, `raw_query`.
* `hardware` (`dict` / `HardwareProfile`): `gpu_model`, `gpu_count`, `vram_gb`, `ram_gb`, `cpu_model`, `os_name`.
* `components` (`list[dict]`): список моделей комплекта (`min_length=1`; если не хватает какой-либо из 3 ролей MVP `llm`/`embedding`/`reranker`, агент автоматически помечает запись статусом `requires_verification`). Поля элемента:
  * `role`: `"llm" | "embedding" | "reranker"` *(регистр нормализуется автоматически)*;
  * `model_id`: строка;
  * `engine`: `"vllm" | "sglang" | "llama.cpp" | "ollama" | "tei"`;
  * `weight_format`, `quantization`, `placement` (`"gpu" | "cpu" | "hybrid"`), `max_model_len`, `max_num_seqs`, `keep_loaded`;
  * `source_url`: URL источника *(пустые строки, `"—"` или битые ссылки из Excel-парсеров не роняют валидацию — они очищаются и фиксируются как гипотеза)*.
* `memory_breakdown` (`dict`): 6 слагаемых (`llm_weights`, `llm_kv_cache`, `embedding_resident_memory`, `reranker_resident_memory`, `runtime_overhead`, `safety_reserve`), каждое вида `{"value": float, "source": str, "status": "catalog"|"declared"|"benchmark"|"estimated"|"unknown"}`, плюс `total_vram_gb` и `total_ram_gb`.

**Опциональные ключи:**
* `project_id` (`str`, по умолчанию `"prj_unknown"`);
* `catalog_version` (`str`, по умолчанию `"2026.10.1"`);
* `primary_set_id`, `alternative_set_id`, `decision_rationale`;
* `rejected_options` (`list[dict]`): элементы вида `{"candidate_id": str, "role": str, "reason": str}`;
* `warnings` (`list[str]`);
* `trace` (`list[dict]`);
* `errors` (`list[str]`).

### Что Агент 5 записывает в `state`

Возвращается обновлённый словарь `dict`, в котором появляются/обновляются ключи:
* `reflection` (`dict | None`): сериализованная запись `ReflectionRecord` (или `None` при критической нехватке входных данных);
* `scs_code` (`str`): готовый граф опыта на языке `SCs` для загрузки в `sc-machine` *(не перезаписывается при срабатывании дедупликации)*;
* `trace` (`list[dict]`): список событий `AgentTraceEvent` с добавленными шагами Агента 5 (`prevent_duplication`, `categorize_solution`, `store_ostis_experience`);
* `errors` (`list[str]`): некритичные и критичные ошибки обработки (если возникли).

---

## 4. Требования к внешнему HTTP-сервису OSTIS

При переходе с `LocalDictionaryOstisGateway` на `HttpOstisGateway(base_url=...)` ожидается, что Python/HTTP-адаптер к `sc-machine` предоставляет 4 эндпоинта:

| Метод и путь | Назначение | Ожидаемый ответ |
| :--- | :--- | :--- |
| `GET /api/ostis/hash/{deduplication_hash}` | Проверка полного дубликата перед записью | `200 OK` (словарь `ReflectionRecord`) или `404 Not Found` |
| `GET /api/ostis/similar?language=...&domain=...` | Поиск последнего похожего кейса для связи `nrel_similar_design_case` | `200 OK` (словарь `ReflectionRecord`) или `404 Not Found` |
| `POST /api/ostis/cases` | Сохранение кейса. Тело: `{"record": dict, "scs": str}` | `200 OK` с `{"sc_addr": "..."}` |
| `GET /api/ostis/cases` | Получение истории накопленных случаев для UI | `200 OK` с `{"items": [...]}` |

---

## 5. Отказоустойчивость (Гарантии для общей системы)

Агент спроектирован так, чтобы не прерывать выполнение общего конвейера `LangGraph`:
* **Отсутствие ключей или ошибки валидации:** не выбрасывают необработанных исключений наружу — фиксируются в `state["errors"]` и `state["trace"]`.
* **Таймаут или сбой LLM:** при ошибке `OpenAICompatibleCategorizer` агент автоматически переключается на `DeterministicLocalCategorizer`, фиксирует предупреждение в `errors` и завершает сохранение опыта.
* **Недоступность сервера OSTIS:** при сетевой ошибке `HttpOstisGateway` сформированные `reflection` и `scs_code` всё равно возвращаются в `state`, а ошибка сети логируется в `errors` и `trace`.
* **Регистр и «грязные» строки из парсеров:** значения `Literal` (`"vLLM"`, `"GPU"`, `"Catalog"`) автоматически нормализуются к нижнему регистру до валидации; прочерки `"—"` и `None` в источниках и ссылках безопасно обрабатываются.

---

## 6. Интеграция API для экрана истории и трассы (FastAPI)

Для подключения блока UI Агента 5 к общему бэкенду FastAPI необходимо пробросить два GET-роута (реализация-образец находится в `app.py`):
* `GET /api/ostis/cases` — вызывает `await ostis_gateway.list_all()`, возвращает список сохранённых кейсов с полем `scs_code`, оценками памяти, гипотезами и ссылкой `similar_project_id` (для отображения diff между RTX 5090 и RTX 4090).
* `GET /api/trace` — возвращает сквозной список событий `trace` (с полями `from_node`, `to_node`, `agent`, `step`, `details`, `updated_keys`, `candidates_count`).

---

## 7. Локальный запуск для проверки перед интеграцией

**Требования:** Python 3.12+

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install fastapi uvicorn httpx pydantic
python app.py
```

После запуска:
1. В консоли выведется результат 3 прогонов (Сценарий RTX 5090 $\rightarrow$ повторный запуск с проверкой дедупликации $\rightarrow$ перенос на RTX 4090) и полная трасса переходов по 5 агентам.
2. В папке `./isolated_ostis_kb/` появятся сгенерированные файлы онтологии `prj_rtx5090_001.scs` и `prj_rtx4090_002.scs`.
3. В браузере по адресу `http://127.0.0.1:8005` откроется дашборд с карточками накопленного опыта OSTIS и таблицей сквозной трассы `Trace`.