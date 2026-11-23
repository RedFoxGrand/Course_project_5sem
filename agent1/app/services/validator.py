import re
from typing import Any
from app.domain.schemas import (
    AssistantProjectRequest,
    ClarificationQuestion,
    HardwareProfile,
    LLMRequirementExtraction,
    RequirementProfile,
)


class HardwareAndRequestValidator:
    def __init__(self, rules: dict[str, Any]) -> None:
        self._rules = rules

    def apply_hardware_clarifications(
        self,
        hardware: HardwareProfile,
        answers: dict[str, str],
    ) -> HardwareProfile:
        updated = hardware.model_copy(deep=True)
        if not answers:
            return updated

        if "vram_gb" in answers:
            match = re.search(r"\d+(?:[.,]\d+)?", answers["vram_gb"])
            if match:
                updated.vram_gb = float(match.group(0).replace(",", "."))

        if "ram_gb" in answers:
            match = re.search(r"\d+(?:[.,]\d+)?", answers["ram_gb"])
            if match:
                val = float(match.group(0).replace(",", "."))
                if val > 0:
                    updated.ram_gb = val

        if "gpu_count" in answers:
            match = re.search(r"\d+", answers["gpu_count"])
            if match:
                updated.gpu_count = int(match.group(0))

        if "gpu_name" in answers:
            val_raw = answers["gpu_name"].strip()
            val_low = val_raw.lower()
            if val_low in {"cpu", "нет", "none", "cpu-only", "только cpu"}:
                updated.gpu_name = None
                updated.gpu_count = 0
                updated.vram_gb = 0.0
                updated.allow_cpu_offload = True
            elif val_raw:
                updated.gpu_name = val_raw
                if updated.gpu_count == 0:
                    updated.gpu_count = 1

        if "allow_cpu_offload" in answers:
            val_low = answers["allow_cpu_offload"].strip().lower()
            if val_low in {"true", "1", "yes", "да", "разрешить"}:
                updated.allow_cpu_offload = True
            elif val_low in {"false", "0", "no", "нет", "запретить"}:
                updated.allow_cpu_offload = False

        if "os" in answers and answers["os"].strip():
            updated.os = answers["os"].strip()

        return updated

    def validate(
        self,
        request: AssistantProjectRequest,
        hardware: HardwareProfile,
    ) -> tuple[list[ClarificationQuestion], list[str], list[str]]:
        questions: list[ClarificationQuestion] = []
        warnings: list[str] = []
        applied_rules: list[str] = []

        constraints = self._rules.get("constraints", {})
        templates = self._rules.get("clarification_templates", {})
        known_gpus = self._rules.get("known_gpus", {})

        min_desc = constraints.get("min_description_length", 15)
        if len(request.task_description.strip()) < min_desc and not request.user_clarifications:
            applied_rules.append("RULE_SHORT_TASK_DESCRIPTION")
            questions.append(
                ClarificationQuestion(
                    field_name="task_description",
                    question=templates.get("short_description", "Уточните описание задачи."),
                    importance="critical",
                    reason_rule_id="RULE_SHORT_TASK_DESCRIPTION",
                )
            )

        has_gpu = hardware.gpu_count > 0 or bool(hardware.gpu_name and hardware.gpu_name.strip())
        if has_gpu:
            applied_rules.append("RULE_GPU_VRAM_CHECK")
            if hardware.gpu_name and hardware.gpu_name.strip() and hardware.gpu_count == 0:
                applied_rules.append("RULE_GPU_COUNT_ZERO_WITH_NAME")
                warnings.append(
                    f"Указана модель GPU ({hardware.gpu_name}), но gpu_count=0; расчёт предполагает 1 GPU."
                )
            if hardware.vram_gb <= 0:
                questions.append(
                    ClarificationQuestion(
                        field_name="vram_gb",
                        question=templates.get("missing_gpu_vram", "Укажите объём VRAM."),
                        importance="critical",
                        reason_rule_id="RULE_GPU_VRAM_ZERO",
                    )
                )
            else:
                min_vram = constraints.get("min_vram_for_gpu_gb", 4.0)
                if hardware.vram_gb < min_vram:
                    applied_rules.append("RULE_LOW_GPU_VRAM")
                    warnings.append(
                        templates.get(
                            "low_vram_warning",
                            "Объём VRAM ({actual} ГБ) ниже рекомендуемого минимума ({min_vram} ГБ).",
                        ).format(actual=hardware.vram_gb, min_vram=min_vram)
                    )
        else:
            applied_rules.append("RULE_CPU_ONLY_CHECK")
            if not hardware.allow_cpu_offload:
                questions.append(
                    ClarificationQuestion(
                        field_name="gpu_name",
                        question=templates.get(
                            "missing_hardware_mode", "Укажите GPU или разрешите CPU."
                        ),
                        importance="critical",
                        reason_rule_id="RULE_NO_ACCELERATOR",
                    )
                )

        if hardware.ram_gb < constraints.get("min_ram_gb", 8.0):
            applied_rules.append("RULE_LOW_SYSTEM_RAM")
            warnings.append(
                f"Объём оперативной памяти ({hardware.ram_gb} ГБ) ниже рекомендуемого минимума "
                f"({constraints.get('min_ram_gb', 8.0)} ГБ)."
            )

        if hardware.gpu_name and hardware.vram_gb > 0:
            effective_gpu_count = max(hardware.gpu_count, 1)
            for known_name, expected_vram in known_gpus.items():
                if known_name.lower() in hardware.gpu_name.lower():
                    applied_rules.append(f"RULE_KNOWN_GPU_{known_name.replace(' ', '_')}")
                    matches_single = abs(hardware.vram_gb - expected_vram) <= 1.0
                    matches_total = abs(hardware.vram_gb - expected_vram * effective_gpu_count) <= 1.0
                    if not (matches_single or matches_total):
                        warnings.append(
                            templates.get("vram_mismatch_warning", "").format(
                                actual=hardware.vram_gb,
                                gpu=known_name,
                                expected=expected_vram,
                            )
                        )

        return questions, warnings, applied_rules

    def normalize_profile(
        self,
        extracted: LLMRequirementExtraction,
        hardware: HardwareProfile,
        hw_warnings: list[str],
    ) -> tuple[RequirementProfile, HardwareProfile, list[str]]:
        defaults = self._rules.get("defaults", {})
        constraints = self._rules.get("constraints", {})
        lang_aliases = self._rules.get("language_aliases", {})
        supported_langs = set(self._rules.get("supported_languages", ["ru", "en"]))
        domain_aliases = self._rules.get("domain_aliases", {})
        supported_domains = set(
            self._rules.get("supported_domains", ["general", "code", "technical", "medical"])
        )

        warnings = list(hw_warnings)
        applied_rules: list[str] = []

        raw_lang = (extracted.language or defaults.get("language", "ru")).strip().lower()
        if raw_lang in lang_aliases:
            mapped_lang = lang_aliases[raw_lang]
            if raw_lang in {"multilingual", "мультиязычный"}:
                applied_rules.append("RULE_NORMALIZE_MULTILINGUAL_TO_ISO")
                warnings.append(
                    f"Запрошен мультиязычный режим; для совместимости с каталогом Агента 2 выбран базовый язык '{mapped_lang}'."
                )
            language = mapped_lang
        elif re.fullmatch(r"[a-z]{2}", raw_lang):
            language = raw_lang
            if language not in supported_langs:
                applied_rules.append("RULE_RARE_LANGUAGE_ISO")
        else:
            applied_rules.append("RULE_FALLBACK_LANGUAGE")
            language = defaults.get("language", "ru")
            warnings.append(
                f"Язык '{raw_lang}' не распознан как ISO 639-1; применён язык по умолчанию '{language}'."
            )

        raw_domain = (extracted.domain or defaults.get("domain", "general")).strip().lower()
        if raw_domain in supported_domains:
            domain = raw_domain
        elif raw_domain in domain_aliases:
            applied_rules.append("RULE_NORMALIZE_DOMAIN_FOR_AGENT2")
            domain = domain_aliases[raw_domain]
            warnings.append(
                f"Домен '{raw_domain}' приведён к каноническому домену '{domain}' для Агента 2."
            )
        else:
            applied_rules.append("RULE_FALLBACK_DOMAIN")
            domain = "general"
            warnings.append(
                f"Домен '{raw_domain}' не входит в поддерживаемый перечень; применён 'general'."
            )

        min_ctx = constraints.get("min_context_len", 512)
        max_ctx = constraints.get("max_context_len", 1048576)
        raw_ctx = extracted.max_context_length or defaults.get("max_context_length", 8192)
        if raw_ctx <= 0:
            context_length = defaults.get("max_context_length", 8192)
            applied_rules.append("RULE_INVALID_CONTEXT_FALLBACK")
        else:
            context_length = int(raw_ctx)
            if context_length < min_ctx:
                applied_rules.append("RULE_CONTEXT_BELOW_MIN")
                warnings.append(
                    f"Запрошенная длина контекста ({context_length}) меньше минимальной ({min_ctx}); увеличена до {min_ctx}."
                )
                context_length = min_ctx
            elif context_length > max_ctx:
                applied_rules.append("RULE_CONTEXT_ABOVE_MAX")
                warnings.append(
                    f"Запрошенная длина контекста ({context_length}) превышает верхний предел ({max_ctx})."
                )

        max_sessions = constraints.get("max_parallel_sessions", 64)
        raw_sessions = extracted.parallel_sessions or defaults.get("parallel_sessions", 1)
        parallel_sessions = max(1, min(int(raw_sessions), max_sessions))
        if raw_sessions != parallel_sessions:
            applied_rules.append("RULE_CLAMP_PARALLEL_SESSIONS")
            warnings.append(
                f"Число параллельных сессий скорректировано с {raw_sessions} до {parallel_sessions}."
            )

        allow_cpu_offload = (
            bool(extracted.allow_cpu_offload)
            if extracted.allow_cpu_offload is not None
            else bool(hardware.allow_cpu_offload)
        )
        synced_hardware = hardware.model_copy(deep=True)
        if allow_cpu_offload and not synced_hardware.allow_cpu_offload:
            synced_hardware.allow_cpu_offload = True
            applied_rules.append("RULE_SYNC_HARDWARE_CPU_OFFLOAD")

        profile = RequirementProfile(
            language=language,
            domain=domain,
            max_context_length=context_length,
            privacy_required=(
                extracted.privacy_required
                if extracted.privacy_required is not None
                else True
            ),
            parallel_sessions=parallel_sessions,
            priority_quality=extracted.priority_quality or defaults.get("priority_quality", 3),
            priority_speed=extracted.priority_speed or defaults.get("priority_speed", 3),
            priority_memory=extracted.priority_memory or defaults.get("priority_memory", 3),
            allow_cpu_offload=allow_cpu_offload,
            warnings=warnings,
        )
        return profile, synced_hardware, applied_rules
