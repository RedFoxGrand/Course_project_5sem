# Мультиагентный программный комплекс «Нейро Мастер»

Интеллектуальная среда проектирования локальных RAG-ассистентов на базе **LangGraph**, **Pydantic v2**, **SQLite (WAL)**, **FastAPI** и базы знаний **OSTIS (`SCs`)**.

Система принимает текстовое описание прикладной задачи и профиль аппаратного обеспечения, формирует структурированные требования, подбирает жизнеспособный комплект моделей (`LLM + Embedding + Reranker`), рассчитывает параметры совместного инференса и потребление VRAM/RAM, а затем фиксирует проектный опыт, гипотезы и SCs-онтологию в базе знаний OSTIS.

---

## Архитектура и состав агентов

Конвейер реализован в виде единого графа состояний `LangGraph` (`orchestrator/graph.py`) с изолированными модулями агентов и слоем адаптеров данных (`orchestrator/adapters.py`):

```text
START
  └─► Agent 1 (RequirementsAnalyst) ──[критичные вопросы?]──► save_project ──► END
        └─► Agent 2 (ModelResearcher / SQLite WAL KB) ──[тройка неполная?]──► save_project ──► END
              └─► Agent 3 (ModelSelector) ──[нет комплектов?]──► save_project ──► END
                    └─► Agent 4 (InferenceConfigurator) ──[не помещается?]──► save_project ──► END
                          └─► Agent 5 (ReflectionAgent / OSTIS SCs)
                                └─► save_project ──► END
```

| Модуль | Роль в системе | Ключевые механизмы |
| :--- | :--- | :--- |
| **`agent1/`** | Аналитик требований (`RequirementsAnalyst`) | Извлечение параметров из текста, валидация оборудования по YAML-правилам, генерация уточняющих вопросов, перерасчёт при смене GPU |
| **`agent2/`** | Исследователь моделей (`ModelResearcher`) | Единая локальная БЗ SQLite (WAL), офлайн-поиск с многокритериальным ранжированием, защита статического каталога, фоновая синхронизация с Hugging Face Hub, контроль целостности тройки RAG |
| **`agent3/`** | Подборщик комплекта (`ModelSelector`) | Дифференцированная фильтрация по ролям, предварительная оценка памяти с учётом квантования, подбор `Primary` и `Economic` комплектов (LLM Structured Output + детерминированный fallback) |
| **`agent4/`** | Конфигуратор инференса (`InferenceConfigurator`) | Детерминированный расчёт совместной VRAM/RAM (веса, KV-cache, runtime, batch, safety reserve), правила совместимости `vLLM` / `llama.cpp` / `Ollama` / `TEI`, CPU-offload, генерация CLI-команд запуска |
| **`agent5/`** | Рефлексивный агент (`ReflectionAgent`) | SHA-256 дедупликация решений, поиск похожих проектов (`nrel_similar_design_case`), разделение фактов и гипотез (`confirmed_fact` / `hypothesis` / `requires_verification`), генерация SCs-кода для OSTIS |
| **`orchestrator/`** | Слой интеграции | Единое состояние `SharedDesignState`, адаптеры контрактов между агентами, унификация журнала трассировки (`trace`), маршрутизация графа |

---

## Структура проекта

```text
.
├── agent1/                  # Агент 1: анализ требований и правила валидации
├── agent2/                  # Агент 2: база знаний SQLite WAL, каталоги и HF-адаптер
├── agent3/                  # Агент 3: подбор и ранжирование комплектов моделей
├── agent4/                  # Агент 4: планирование совместного инференса и памяти
├── agent5/                  # Агент 5: рефлексия и генератор SCs-онтологии OSTIS
├── orchestrator/
│   ├── state.py             # Единая схема состояния SharedDesignState
│   ├── adapters.py          # Адаптеры форматов (1->2->3->4->5) и нормализация trace
│   └── graph.py             # Сквозной граф LangGraph
├── api.py                   # Объединённый FastAPI-сервер и HTML-дашборд OSTIS
├── run_demo.py              # CLI-демонстрация сквозного конвейера (RTX 5090 -> RTX 4090)
├── test_integration.py      # Сквозные интеграционные тесты
├── requirements.txt         # Единые зависимости проекта
├── .gitignore
└── README.md
```

---

## Установка

Требуется **Python 3.12+**.

```bash
python -m venv .venv

# Windows (PowerShell / cmd):
.\.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

---

## Запуск объединённой системы

### 1. Сквозной CLI-прогон (сценарии RTX 5090 и перенос на RTX 4090)

```bash
python run_demo.py
```

Скрипт последовательно прогоняет оба сценария через всех 5 агентов, выводит в консоль выбранные планы развёртывания, команды запуска, сквозную трассу и сохраняет:
* полные состояния проектов в `data/projects/`;
* сгенерированные SCs-файлы онтологии в `isolated_ostis_kb/`.

### 2. Единый веб-сервер и дашборд OSTIS (FastAPI)

```bash
python api.py
```

После запуска доступны:
* **HTML-дашборд OSTIS и трассы:** `http://127.0.0.1:8000/`
* **Swagger UI (интерактивное тестирование API):** `http://127.0.0.1:8000/docs`

**Основные REST-эндпоинты:**
* `GET /health` — проверка состояния сервиса;
* `POST /api/v1/projects/analyze` — запуск полного 5-агентного конвейера;
* `POST /api/v1/projects/{project_id}/clarify` — доотправка ответов на критичные вопросы Агента 1;
* `PATCH /api/v1/projects/{project_id}/hardware` — пересчёт существующего проекта под новое оборудование (например, перенос с RTX 5090 на RTX 4090);
* `GET /api/v1/projects/{project_id}` — чтение сохранённого проекта;
* `GET /api/v1/projects/{project_id}/export?format=json|yaml` — экспорт результата в JSON или YAML;
* `GET /api/ostis/cases` — список накопленных в БЗ OSTIS кейсов с SCs-кодом;
* `GET /api/trace` — сквозная трасса выполнения агентов по проектам.

---

## Автономное тестирование отдельных агентов

Каждый модуль сохраняет полную изоляцию и может проверяться отдельно из активированного корневого `.venv`:

```bash
# Тесты Агента 1 (7 тестов)
cd agent1 && python -m pytest -v && cd ..

# Тесты Агента 2 (6 тестов)
cd agent2 && python -m pytest -v && cd ..

# Тесты агента 3 (детерминированный автономный smoke-test без API-ключа)
python -c "from agent3.agent3_selector import agent3_select_model_sets; from agent3.schemas import RequirementProfile, HardwareProfile; from agent3.mock_data import MOCK_CANDIDATES; r = agent3_select_model_sets(MOCK_CANDIDATES, RequirementProfile(max_context_length=4096), HardwareProfile(), None); print('Agent 3 OK, sets:', [s.set_name for s in r['model_sets']])"

# Тесты Агента 4 (19 тестов)
cd agent4 && python -m pytest -v && cd ..
```

Дополнительные скрипты автономной проверки:
* `cd agent2 && python scripts/verify_scenarios.py` — проверка 5 сценариев поиска по SQLite-БЗ;
* `cd agent2 && python scripts/sync_catalog.py` — пересборка локальной БЗ и принудительная синхронизация с Hugging Face Hub;
* `cd agent4 && python -m inference_agent examples/rtx5090_input.json --out plan` — автономный CLI-расчёт плана инференса.
