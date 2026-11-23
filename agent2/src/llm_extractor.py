import re
from typing import Any

from .schemas import EngineType, ExtractedCardMetadata


class FreeTextCardExtractor:
    """Извлечение скрытых характеристик из свободного текста README с помощью LLM (Structured Output)."""

    def __init__(self, llm_instance: Any = None):
        """
        Если передан экземпляр LLM (например, ChatOpenAI или ChatOllama из LangChain),
        настраиваем structured_output. Если нет — используем резервный детерминированный парсер.
        """
        self.llm = llm_instance
        if self.llm and hasattr(self.llm, "with_structured_output"):
            self.structured_llm = self.llm.with_structured_output(ExtractedCardMetadata)
        else:
            self.structured_llm = None

    def extract_from_markdown(self, markdown_text: str) -> ExtractedCardMetadata:
        """Анализирует текст README.md карточки модели."""
        if not markdown_text or not markdown_text.strip():
            return ExtractedCardMetadata()

        text_sample = markdown_text[:3500]

        if self.structured_llm:
            try:
                prompt = (
                    "Проанализируй фрагмент карточки нейросетевой модели и извлеки характеристики:\n"
                    "1. Доступные квантования весов (GGUF, Q4_K_M, AWQ, GPTQ, FP8 и т.д.).\n"
                    "2. Совместимые движки инференса, упомянутые в инструкциях (vLLM, SGLang, Ollama, llama.cpp, TEI).\n"
                    "3. Название и значение основного бенчмарка качества (MMLU, GPQA, MTEB, HumanEval).\n\n"
                    f"Текст карточки:\n{text_sample}"
                )
                result: ExtractedCardMetadata = self.structured_llm.invoke(prompt)
                return result
            except Exception: # noqa: BLE001, S110
                pass

        return self._heuristic_fallback(text_sample)

    @staticmethod
    def _heuristic_fallback(text: str) -> ExtractedCardMetadata:
        quants: list[str] = []
        for p in [
            r"\bQ[2-8]_[KMS_0-9]+\b",
            r"\bAWQ\b",
            r"\bGPTQ\b",
            r"\bFP8\b",
            r"\bINT4\b",
            r"\bINT8\b",
        ]:
            matches = re.findall(p, text, re.IGNORECASE)
            quants.extend([m.upper() for m in matches])

        engines: list[EngineType] = []
        tl = text.lower()
        if "vllm" in tl:
            engines.append("vllm")
        if "sglang" in tl:
            engines.append("sglang")
        if "llama.cpp" in tl or "gguf" in tl:
            engines.append("llama.cpp")
        if "ollama" in tl:
            engines.append("ollama")
        if "tei" in tl or "text-embeddings-inference" in tl:
            engines.append("tei")

        bench_name = None
        bench_val = None
        m = re.search(
            r"(MMLU|GPQA|HumanEval|ARC|MTEB)\s*[:=-]?\s*(\d+(?:\.\d+)?)\s*%?",
            text,
            re.IGNORECASE,
        )
        if m:
            bench_name = m.group(1).upper()
            bench_val = float(m.group(2))

        return ExtractedCardMetadata(
            supported_quantizations=sorted(set(quants)),
            compatible_engines=sorted(set(engines)),
            benchmark_name=bench_name,
            benchmark_value=bench_val,
        )
