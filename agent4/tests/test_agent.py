from copy import deepcopy
import json
from pathlib import Path

import pytest
from inference_agent import InferenceConfigurator, configure_inference, export_json, export_yaml

DEMO = Path(__file__).resolve().parents[1] / "examples" / "rtx5090_input.json"

def input_state():
    return json.loads(DEMO.read_text(encoding="utf-8"))

def test_dictionary_protocol_and_no_mutation():
    state = input_state()
    before = deepcopy(state)
    out = configure_inference(state)
    assert type(out) is dict
    assert type(out["deployment_plan"]) is dict
    assert out["request"] == state["request"]
    assert out["trace"][0] == state["trace"][0]
    assert state == before

def test_5090_chooses_estimated_resident_configuration():
    out = configure_inference(input_state())
    plan = out["deployment_plan"]
    assert plan["status"] == "planned"
    assert plan["primary"]["memory"]["fit_by_estimate"] is True
    assert plan["primary"]["memory"]["required_vram_gb"] <= 32
    assert plan["primary"]["memory"]["required_ram_gb"] <= 64
    assert plan["primary"]["memory"]["requires_verification"] is True
    assert plan["primary"]["context_length"] == 8192
    assert plan["primary"]["parallel_sessions"] == 5
    assert plan["fallback"] and plan["fallback"]["id"] != plan["primary"]["id"]

def test_4090_recalculates_without_modifying_request():
    state = input_state()
    state["hardware"]["gpus"][0]["name"] = "NVIDIA RTX 4090"
    state["hardware"]["gpus"][0]["vram_gb"] = 24
    out = configure_inference(state)
    p = out["deployment_plan"]["primary"]
    assert p is not None
    assert p["memory"]["required_vram_gb"] <= 24
    assert p["context_length"] == state["requirements"]["context_length"]

def test_cpu_placement_when_gpu_small():
    state = input_state()
    state["hardware"]["gpus"][0]["vram_gb"] = 12
    out = configure_inference(state)
    p = out["deployment_plan"]["primary"]
    assert p is not None
    assert p["embedding"]["device"] == "cpu" or p["reranker"]["device"] == "cpu" or p["llm"]["quantization"] == "awq"

def test_not_fit_returns_failure_not_fake_success():
    state = input_state()
    state["hardware"]["ram_gb"] = 3
    state["hardware"]["gpus"][0]["vram_gb"] = 2
    out = configure_inference(state)
    plan = out["deployment_plan"]
    assert plan["status"] == "no_feasible_solution"
    assert plan["primary"] is None
    assert plan["rejected"]

def test_missing_model_metadata_raises_warning():
    state = input_state()
    state["model_sets"][0]["embedding"].pop("parameters_b")
    state["model_sets"][1]["embedding"].pop("parameters_b")
    out = configure_inference(state)
    plan = out["deployment_plan"]
    assert plan["primary"] is not None
    assert plan["primary"]["status"] == "requires_verification"
    assert plan["primary"]["memory"]["fit_by_estimate"] is None

def test_engine_support_is_not_assumed():
    state = input_state()
    for s in state["model_sets"]:
        s["reranker"]["engines"] = []
    out = configure_inference(state)
    assert out["deployment_plan"]["status"] == "no_feasible_solution"
    assert any("reranker" in r["reason"] for r in out["deployment_plan"]["rejected"])

def test_reduction_needs_explicit_permission():
    state = input_state()
    state["requirements"]["context_length"] = 32768
    state["requirements"]["parallel_sessions"] = 12
    out = configure_inference(state)
    p = out["deployment_plan"]["primary"]
    if p:
        assert p["context_length"] == 32768
        assert p["parallel_sessions"] == 12
    state["requirements"].update({"allow_reduce_context": True,
                                  "min_context_length": 4096,
                                  "allow_reduce_parallelism": True,
                                  "min_parallel_sessions": 2})
    out = configure_inference(state)
    assert out["deployment_plan"]["primary"] is not None

def test_multigpu_does_not_sum_vram():
    state = input_state()
    state["hardware"]["gpus"] = [{"name": "A", "vram_gb": 16, "count": 2}]
    plan = configure_inference(state)["deployment_plan"]
    assert plan["primary"] is not None
    assert plan["primary"]["memory"]["available_vram_gb"] == 16
    assert any("Multi-GPU" in msg or "multi-GPU" in msg for msg in plan["warnings"])

def test_ranker_cannot_choose_invalid_strategy():
    state = input_state()
    ranker = lambda d: {"strategy_id": "not-approved", "explanation": "fake"}
    plan = InferenceConfigurator(ranker=ranker).run(state)["deployment_plan"]
    assert plan["status"] == "planned"
    assert any("LLM ranking ignored" in s for s in plan["warnings"])

def test_invalid_state_records_error():
    out = configure_inference({"project_id": "x", "trace": [], "errors": []})
    assert out["deployment_plan"] is None
    assert out["errors"]
    assert out["trace"][-1]["event"] == "validation_failed"

def test_export_json_yaml_roundtrip(tmp_path):
    out = configure_inference(input_state())
    j = tmp_path / "p.json"
    y = tmp_path / "p.yaml"
    export_json(out, j)
    export_yaml(out, y)
    assert json.loads(j.read_text(encoding="utf-8"))["status"] == "planned"
    assert "primary:" in y.read_text(encoding="utf-8")

def test_cpu_only_llamacpp_configuration():
    state = input_state()
    state["hardware"]["gpus"] = []
    state["hardware"]["ram_gb"] = 64
    for model_set in state["model_sets"]:
        model_set["llm"]["engines"] = ["llama.cpp"]
        model_set["llm"]["formats"] = ["gguf"]
        model_set["llm"]["quantizations"] = ["q4_k_m"]
        model_set["llm"]["metadata"] = {"local_path": "/models/demo.gguf"}
    p = configure_inference(state)["deployment_plan"]["primary"]
    assert p is not None
    assert p["llm"]["device"] == "cpu"
    assert p["llm"]["engine"] == "llama.cpp"
    assert p["llm"]["command"][0] == "llama-server"
    assert p["memory"]["required_vram_gb"] == 0


def test_engine_fallback_from_vllm_to_llamacpp():
    state = input_state()
    state["hardware"]["gpus"][0]["vram_gb"] = 6
    for model_set in state["model_sets"]:
        model_set["llm"]["engines"] = ["vllm", "llama.cpp"]
        model_set["llm"]["formats"] = ["safetensors", "gguf"]
        model_set["llm"]["quantizations"] = ["fp16", "q4_k_m"]
    plan = configure_inference(state)["deployment_plan"]
    assert plan["primary"] is not None
    assert plan["primary"]["llm"]["engine"] == "llama.cpp"


def test_unknown_weight_with_already_exceeded_memory_is_rejected():
    state = input_state()
    state["hardware"]["gpus"][0]["vram_gb"] = 1
    state["hardware"]["ram_gb"] = 12
    for model_set in state["model_sets"]:
        model_set["embedding"].pop("parameters_b")
    plan = configure_inference(state)["deployment_plan"]
    assert plan["primary"] is None  # known fixed reserves alone exceed VRAM/RAM


def test_invalid_model_set_records_input_validation_error():
    state = input_state()
    state["model_sets"][0]["llm"].pop("id")
    out = configure_inference(state)
    assert out["deployment_plan"] is None
    assert "validation" in out["errors"][-1]

def test_language_is_checked_when_card_provides_supported_languages():
    state = input_state()
    for ms in state["model_sets"]:
        for role in ("llm", "embedding", "reranker"):
            ms[role]["languages"] = ["en"]
    plan = configure_inference(state)["deployment_plan"]
    assert plan["status"] == "no_feasible_solution"
    assert any("язык" in item["reason"] for item in plan["rejected"])


def test_local_only_rejects_explicitly_nonlocal_model():
    state = input_state()
    for ms in state["model_sets"]:
        ms["llm"]["local_available"] = False
    plan = configure_inference(state)["deployment_plan"]
    assert plan["status"] == "no_feasible_solution"
    assert any("локальное" in r["reason"] for r in plan["rejected"])


def test_language_and_licenses_warning_when_not_declared():
    state = input_state()
    state["requirements"]["license_allowlist"] = ["apache-2.0"]
    plan = configure_inference(state)["deployment_plan"]
    assert plan["primary"]["status"] == "requires_verification"
    assert any("Лицензии" in s for s in plan["warnings"])
