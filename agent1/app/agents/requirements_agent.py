from app.domain.interfaces import ILLMProvider, IRulesRepository
from app.domain.schemas import (
    AgentTraceEvent,
    ClarificationQuestion,
)
from app.services.validator import HardwareAndRequestValidator
from app.workflow.state import DesignState


class RequirementsAnalystAgent:
    def __init__(
        self,
        rules_repo: IRulesRepository,
        llm_provider: ILLMProvider,
    ) -> None:
        self._rules_repo = rules_repo
        self._llm_provider = llm_provider

    def invoke(self, state: DesignState) -> dict:
        rules = self._rules_repo.load_rules()
        validator = HardwareAndRequestValidator(rules)

        request = state["request"]
        hardware = validator.apply_hardware_clarifications(
            state["hardware"], request.user_clarifications
        )
        trace = list(state.get("trace", []))
        errors = list(state.get("errors", []))

        det_questions, hw_warnings, applied_rules = validator.validate(request, hardware)

        try:
            extracted = self._llm_provider.extract_requirements(request, hardware, rules)
        except Exception as exc:
            errors.append(f"Ошибка анализа требований: {str(exc)}")
            trace.append(
                AgentTraceEvent(
                    agent="RequirementsAnalyst",
                    step="analyze_requirements",
                    details=f"Сбой вызова LLM: {str(exc)}",
                    status="error",
                    applied_rules=list(dict.fromkeys(applied_rules)),
                ).model_dump()
            )
            return {
                "hardware": hardware,
                "requirements": None,
                "clarification_questions": [],
                "errors": errors,
                "trace": trace,
            }

        combined_questions: dict[str, ClarificationQuestion] = {}
        for q in det_questions:
            combined_questions[q.field_name] = q
        for q in extracted.missing_critical_info:
            if q.field_name not in request.user_clarifications:
                combined_questions[q.field_name] = q

        final_questions = list(combined_questions.values())

        if any(q.importance == "critical" for q in final_questions):
            applied_rules.extend(q.reason_rule_id for q in final_questions)
            trace.append(
                AgentTraceEvent(
                    agent="RequirementsAnalyst",
                    step="analyze_requirements",
                    details=(
                        f"Обнаружены недостающие критичные сведения ({len(final_questions)} шт.). "
                        "Требуется уточнение у пользователя."
                    ),
                    status="needs_clarification",
                    applied_rules=list(dict.fromkeys(applied_rules)),
                ).model_dump()
            )
            return {
                "hardware": hardware,
                "requirements": None,
                "clarification_questions": final_questions,
                "trace": trace,
                "errors": errors,
            }

        profile, synced_hardware, norm_rules = validator.normalize_profile(
            extracted=extracted,
            hardware=hardware,
            hw_warnings=hw_warnings,
        )
        applied_rules.extend(norm_rules)

        trace.append(
            AgentTraceEvent(
                agent="RequirementsAnalyst",
                step="analyze_requirements",
                details=(
                    f"Сформирован профиль задачи: language={profile.language}, domain={profile.domain}, "
                    f"max_context_length={profile.max_context_length}, sessions={profile.parallel_sessions}, "
                    f"privacy_required={profile.privacy_required}. "
                    f"Оборудование: GPU={synced_hardware.gpu_name or 'CPU'}, VRAM={synced_hardware.vram_gb} ГБ."
                ),
                status="completed",
                applied_rules=list(dict.fromkeys(applied_rules)),
            ).model_dump()
        )

        return {
            "hardware": synced_hardware,
            "requirements": profile.model_dump(),
            "clarification_questions": [],
            "trace": trace,
            "errors": errors,
        }
