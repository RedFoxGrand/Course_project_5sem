#test_agent3.py
import os
from langgraph.graph import StateGraph, START, END

from schemas import RequirementProfile, HardwareProfile
from agent3_selector import create_agent3_node
from state import DesignState
from mock_data import MOCK_CANDIDATES

# from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def main():
    reqs = RequirementProfile(
        language="ru", 
        domain="technical", 
        max_context_length=4000, 
        privacy_required=True
    )
    
    hardware = HardwareProfile(
        cpu_cores=16, 
        ram_gb=64.0, 
        gpu_count=1, 
        gpu_vram_gb=24.0,
        os_name="Linux"
    )

    # инициализация клиента локальной языковой модели через ollama (закомментировано)
    # llm_client = ChatOllama(
    #     model="qwen2.5:3b",
    #     temperature=0.0
    # )

    # чтение api-ключа из внешнего текстового файла для обеспечения безопасности
    try:
        with open("api_key.txt", "r", encoding="utf-8") as key_file:
            os.environ["OPENAI_API_KEY"] = key_file.read().strip()
    except FileNotFoundError:
        print("Ошибка: Файл api_key.txt не найден в корне проекта. Создайте его и поместите туда ключ.")
        return

    # инициализация клиента облачной языковой модели
    llm_client = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        # model="google/gemma-4-26b-a4b-it:free",
        model="nvidia/nemotron-3.5-super-vl:free",
        # model="nvidia/nemotron-3.5-lightning:free",
        temperature=0.0,
    )

    # инициализация графа состояний
    workflow = StateGraph(DesignState)
    
    # добавление узла подбора моделей
    agent3_node = create_agent3_node(llm_client)
    workflow.add_node("model_selector", agent3_node)
    
    # настройка маршрутизации
    workflow.add_edge(START, "model_selector")
    workflow.add_edge("model_selector", END)
    
    # сборка исполняемого графа
    app = workflow.compile()

    # формирование начального состояния
    initial_state = {
        "project_id": "test_run_01",
        "hardware": hardware,
        "requirements": reqs,
        "candidates": MOCK_CANDIDATES,
        "model_sets": [],
        "rejected_reasons": [],
        "warnings": [],
        "trace": [],
        "errors": []
    }

    print("Запуск тестирования подсистемы подбора моделей через LangGraph")
    try:
        # запуск графа
        final_state = app.invoke(initial_state)
        
        if final_state.get("errors"):
            print(f"В процессе выполнения возникли ошибки: {final_state['errors']}")
            return

        print("\nРезультаты интеллектуального формирования конфигураций:")
        for i, model_set in enumerate(final_state.get("model_sets", []), 1):
            print(f"\nКонфигурация {i}: {model_set.set_name}")
            print(f"  Языковая модель: {model_set.llm.name}")
            print(f"  Векторная модель: {model_set.embedding.name}")
            print(f"  Модель ранжирования: {model_set.reranker.name}")
            print(f"  Расчетная VRAM: {model_set.estimated_total_vram_gb:.2f} ГБ из {hardware.gpu_vram_gb} ГБ")
            print(f"  Обоснование: {model_set.rationale}")
            
        # выгрузка и форматирование причин отклонения неоптимальных конфигураций
        rejected = final_state.get("rejected_reasons")
        if rejected:
            print("\nПричины отклонения альтернативных конфигураций:")
            for reason in rejected:
                print(f"  - {reason}")
                
        # выгрузка и форматирование системных предупреждений о неполноте данных
        warnings = final_state.get("warnings")
        if warnings:
            print("\nСистемные предупреждения:")
            for w in warnings:
                print(f"  ! {w}")
                
    except Exception as e:
        print(f"В процессе выполнения возникла критическая ошибка: {e}")

if __name__ == "__main__":
    main()