import re
from typing import Any
from openai import OpenAI
from app.domain.interfaces import ILLMProvider
from app.domain.schemas import (
    AssistantProjectRequest,
    ClarificationQuestion,
    HardwareProfile,
    LLMRequirementExtraction,
)


class OpenAICompatibleRequirementsAdapter(ILLMProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        use_mock: bool = True,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._use_mock = use_mock
        self._timeout_seconds = timeout_seconds

    def extract_requirements(
        self,
        request: AssistantProjectRequest,
        hardware: HardwareProfile,
        rules: dict[str, Any],
    ) -> LLMRequirementExtraction:
        if self._use_mock:
            return self._heuristic_structured_extraction(request, hardware, rules)

        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
        )
        system_prompt = (
            "Вы — Агент 1 (Аналитик требований) интеллектуальной среды проектирования ИИ-ассистентов. "
            "Проанализируйте описание задачи, уточнения пользователя и профиль оборудования. "
            "Верните строго структурированный JSON: "
            "language — двухбуквенный код ISO 639-1 (например ru, en, de; не используйте слово multilingual, "
            "для русскоязычной/мультиязычной задачи с русским укажите ru); "
            "domain — строго одно из значений: general, code, technical, medical; "
            "privacy_required — bool (True, если данные нельзя передавать во внешние API); "
            "max_context_length — целое число токенов (> 0); "
            "parallel_sessions — число параллельных сессий (1..64); "
            "priority_quality, priority_speed, priority_memory — целые числа 1..5; "
            "allow_cpu_offload — bool (допускается ли перенос embedding/reranker на CPU). "
            "Если язык или предметную область невозможно определить из описания и уточнений, "
            "добавьте критичный вопрос в missing_critical_info."
        )
        user_content = (
            f"Описание задачи: {request.task_description}\n"
            f"Ответы на уточнения: {request.user_clarifications}\n"
            f"Оборудование: {hardware.model_dump_json()}"
        )
        completion = client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=LLMRequirementExtraction,
            temperature=0.0,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM не вернула валидный Structured Output")
        return parsed

    def _heuristic_structured_extraction(
        self,
        request: AssistantProjectRequest,
        hardware: HardwareProfile,
        rules: dict[str, Any],
    ) -> LLMRequirementExtraction:
        combined_text = (
            request.task_description
            + " "
            + " ".join(f"{k}: {v}" for k, v in request.user_clarifications.items())
        ).lower()

        language = request.user_clarifications.get("language")
        if language:
            language = language.strip().lower()
        else:
            if any(w in combined_text for w in ["русск", "русскоязыч", " ru ", "на ru"]):
                language = "ru"
            elif any(w in combined_text for w in ["англ", "english", " en ", "на en"]):
                language = "en"
            elif any(w in combined_text for w in ["немецк", "german", " de "]):
                language = "de"
            elif any(w in combined_text for w in ["мультиязыч", "multilingual"]):
                language = "multilingual"

        domain = request.user_clarifications.get("domain")
        if domain:
            domain = domain.strip().lower()
        else:
            for dom_name in ["code", "technical", "medical", "general"]:
                keywords = rules.get("domain_keywords", {}).get(dom_name, [])
                if any(kw in combined_text for kw in keywords):
                    domain = dom_name
                    break

        privacy_required = True
        if "privacy_required" in request.user_clarifications:
            val = request.user_clarifications["privacy_required"].strip().lower()
            privacy_required = val not in {"false", "0", "no", "нет"}
        elif any(
            w in combined_text
            for w in ["внешние api разрешены", "допускаются облачные", "облачный api", "не строго"]
        ):
            privacy_required = False
        elif any(
            w in combined_text
            for w in ["локальн", "нельзя передавать", "приватн", "закрыт", "внутренн"]
        ):
            privacy_required = True

        sessions = rules["defaults"]["parallel_sessions"]
        word_numbers = {
            "один": 1,
            "одной": 1,
            "два": 2,
            "двух": 2,
            "три": 3,
            "трех": 3,
            "четыре": 4,
            "четырех": 4,
            "пять": 5,
            "пяти": 5,
            "десять": 10,
            "десяти": 10,
        }
        for word, val in word_numbers.items():
            if word in combined_text and "параллельн" in combined_text:
                sessions = val
                break
        digit_match = re.search(r"(\d+)\s*(?:параллельн|сесси|пользовател)", combined_text)
        if digit_match:
            sessions = int(digit_match.group(1))
        if "parallel_sessions" in request.user_clarifications:
            m_sess = re.search(r"\d+", request.user_clarifications["parallel_sessions"])
            if m_sess:
                sessions = int(m_sess.group(0))

        if "снижение параллелизма" in combined_text and sessions > 2:
            sessions = max(1, sessions // 2)

        context_len = rules["defaults"]["max_context_length"]
        ctx_k_match = re.search(r"\b(\d+)\s*[kк]\b", combined_text)
        ctx_m_match = re.search(r"\b(\d+)\s*[mм]\s*(?:токен|контекст)", combined_text)
        ctx_exact = re.search(r"\b(512|1024|2048|4096|8192|16384|32768|65536|131072|262144|1048576)\b", combined_text)
        if ctx_m_match:
            context_len = int(ctx_m_match.group(1)) * 1024 * 1024
        elif ctx_k_match:
            context_len = int(ctx_k_match.group(1)) * 1024
        elif ctx_exact:
            context_len = int(ctx_exact.group(1))
        if "max_context_length" in request.user_clarifications:
            m_ctx = re.search(r"\d+", request.user_clarifications["max_context_length"])
            if m_ctx:
                context_len = int(m_ctx.group(0))

        p_quality = rules["defaults"]["priority_quality"]
        p_speed = rules["defaults"]["priority_speed"]
        p_memory = rules["defaults"]["priority_memory"]

        if "приоритет качество" in combined_text or "качество ответа" in combined_text:
            p_quality = 5
        if "интерактивной скорости" in combined_text or "быстр" in combined_text:
            p_speed = 4
        if 0 < hardware.vram_gb <= 24:
            p_memory = max(p_memory, 4)

        cpu_denied = any(
            phrase in combined_text
            for phrase in ["без cpu", "без использования cpu", "только gpu", "не использовать cpu"]
        )
        cpu_allowed_in_text = any(
            phrase in combined_text
            for phrase in [
                "использование cpu",
                "перенос на cpu",
                "cpu для embedding",
                "cpu для reranker",
                "cpu offload",
                "допускается cpu",
                "разрешить cpu",
            ]
        )
        allow_cpu_offload = False if cpu_denied else (hardware.allow_cpu_offload or cpu_allowed_in_text)

        missing: list[ClarificationQuestion] = []
        if not language:
            missing.append(
                ClarificationQuestion(
                    field_name="language",
                    question="Уточните целевой язык документов и ответов ассистента (например: ru, en, de).",
                    importance="critical",
                    reason_rule_id="RULE_MISSING_LANGUAGE",
                )
            )
        if not domain:
            missing.append(
                ClarificationQuestion(
                    field_name="domain",
                    question="Укажите предметную область документов (general, technical, code, medical).",
                    importance="critical",
                    reason_rule_id="RULE_MISSING_DOMAIN",
                )
            )

        return LLMRequirementExtraction(
            language=language,
            domain=domain,
            privacy_required=privacy_required,
            max_context_length=context_len,
            parallel_sessions=sessions,
            priority_quality=p_quality,
            priority_speed=p_speed,
            priority_memory=p_memory,
            allow_cpu_offload=allow_cpu_offload,
            missing_critical_info=missing,
            extraction_rationale=(
                f"Извлечены параметры: language={language}, domain={domain}, "
                f"sessions={sessions}, max_context_length={context_len}, privacy_required={privacy_required}."
            ),
        )
