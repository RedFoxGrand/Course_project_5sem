import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from schemas import ReflectionRecord


def to_sc_idtf(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw.lower()).strip("_")
    return slug or "unnamed_node"


class ScsOntologyGenerator:
    @staticmethod
    def generate(record: ReflectionRecord) -> str:
        p_id = to_sc_idtf(record.project_id)
        case_node = f"design_case_{p_id}"
        task_node = f"task_class_{to_sc_idtf(record.category.task_class)}"
        req_node = f"requirement_profile_{p_id}"
        hw_node = f"hardware_profile_{to_sc_idtf(record.hardware.fingerprint)}"
        cat_node = f"experience_category_{to_sc_idtf(record.category.language_category)}_{to_sc_idtf(record.category.domain_category)}_{to_sc_idtf(record.category.hardware_category)}"
        set_node = f"model_set_{to_sc_idtf(record.primary_set_id)}"
        decision_node = f"accepted_decision_{p_id}"
        rationale_node = f"decision_rationale_{p_id}"

        lines: list[str] = [
            f"{case_node} <- concept_ai_assistant_design_case;",
            f'  => nrel_main_idtf: [{record.category.normalized_summary}] (* <- lang_ru;; *);',
            f'  => nrel_project_id: [{record.project_id}];',
            f'  => nrel_catalog_version: [{record.catalog_version}];',
            f'  => nrel_deduplication_hash: [{record.deduplication_hash}];;',
            f"{task_node} <- concept_task_class;;",
            f"{req_node} <- concept_requirement_profile;",
            f'  => nrel_language: [{record.requirements.language}];',
            f'  => nrel_domain: [{record.requirements.domain}];',
            f"  => nrel_max_context_length: [{record.requirements.max_context_length}];;",
            f"{hw_node} <- concept_hardware_profile;",
            f'  => nrel_gpu_model: [{record.hardware.gpu_model}];',
            f"  => nrel_vram_gb: [{record.hardware.vram_gb}];",
            f"  => nrel_ram_gb: [{record.hardware.ram_gb}];;",
            f"{cat_node} <- concept_experience_category;;",
            f"{case_node} => nrel_has_task: {task_node};;",
            f"{case_node} => nrel_has_requirement_profile: {req_node};;",
            f"{case_node} => nrel_executed_for_hardware: {hw_node};;",
            f"{case_node} => nrel_belongs_to_category: {cat_node};;",
        ]

        if record.similar_project_id:
            sim_node = f"design_case_{to_sc_idtf(record.similar_project_id)}"
            lines.append(f"{case_node} => nrel_similar_design_case: {sim_node};;")

        lines.extend([
            f"{set_node} <- concept_model_set;;",
            f"{decision_node} <- concept_accepted_decision;",
            f"  => nrel_model_set: {set_node};",
            f"  => nrel_epistemic_status: status_{record.overall_epistemic_status};;",
            f"{rationale_node} <- concept_decision_rationale;",
            f'  => nrel_text: [{record.decision_rationale}];;',
            f"{case_node} => nrel_accepted_decision: {decision_node};;",
            f"{decision_node} => nrel_has_rationale: {rationale_node};;",
        ])

        for comp in record.components:
            m_node = f"model_{to_sc_idtf(comp.model_id)}"
            r_node = f"role_{to_sc_idtf(comp.role)}"
            e_node = f"engine_{to_sc_idtf(comp.engine)}"
            cfg_node = f"inference_config_{p_id}_{to_sc_idtf(comp.role)}"

            lines.extend([
                f"{m_node} <- concept_neural_network_model;;",
                f"{r_node} <- concept_model_role;;",
                f"{e_node} <- concept_inference_engine;;",
                f"{cfg_node} <- concept_inference_configuration;",
                f'  => nrel_weight_format: [{comp.weight_format}];',
                f'  => nrel_quantization: [{comp.quantization}];',
                f'  => nrel_placement: [{comp.placement}];',
                f"  => nrel_max_model_len: [{comp.max_model_len}];",
                f"  => nrel_max_num_seqs: [{comp.max_num_seqs}];;",
                f"{decision_node} => nrel_includes_model: {m_node};;",
                f"{m_node} => nrel_performs_role: {r_node};;",
                f"{m_node} => nrel_runs_on_engine: {e_node};;",
                f"{decision_node} => nrel_uses_inference_parameter: {cfg_node};;",
            ])

        for idx, url in enumerate(record.sources):
            src_node = f"open_source_{p_id}_{idx}"
            lines.extend([
                f"{src_node} <- concept_open_source_reference;",
                f'  => nrel_source_url: [{url}];;',
                f"{decision_node} => nrel_based_on_source: {src_node};;",
            ])

        for idx, rej in enumerate(record.rejected_options):
            rej_node = f"rejected_variant_{p_id}_{idx}_{to_sc_idtf(rej.candidate_id)}"
            lines.extend([
                f"{rej_node} <- concept_rejected_variant;",
                f'  => nrel_candidate_id: [{rej.candidate_id}];',
                f'  => nrel_rejection_reason: [{rej.reason}];;',
                f"{case_node} => nrel_has_rejected_variant: {rej_node};;",
            ])

        for unc in record.uncertainties:
            u_node = f"statement_{p_id}_{to_sc_idtf(unc.statement_id)}"
            node_class = (
                "concept_uncertainty"
                if unc.epistemic_status != "confirmed_fact"
                else "concept_verified_statement"
            )
            lines.extend([
                f"{u_node} <- {node_class};",
                f"  => nrel_epistemic_status: status_{unc.epistemic_status};",
                f"  => nrel_provenance_status: provenance_{unc.provenance_status};",
                f'  => nrel_description: [{unc.description}];;',
                f"{case_node} => nrel_has_statement: {u_node};;",
            ])

        return "\n".join(lines)


class AbstractOstisGateway(ABC):
    @abstractmethod
    async def get_by_hash(self, deduplication_hash: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def find_similar(self, language: str, domain: str) -> dict[str, Any] | None:
        pass

    @abstractmethod
    async def store(self, record_dict: dict[str, Any], scs_text: str) -> str:
        pass

    @abstractmethod
    async def list_all(self) -> list[dict[str, Any]]:
        pass


class LocalDictionaryOstisGateway(AbstractOstisGateway):
    def __init__(self, base_dir: str | Path = "./isolated_ostis_kb") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        self._scs_map: dict[str, str] = {}

    async def get_by_hash(self, deduplication_hash: str) -> dict[str, Any] | None:
        for rec in self._records.values():
            if rec.get("deduplication_hash") == deduplication_hash:
                return dict(rec)
        return None

    async def find_similar(self, language: str, domain: str) -> dict[str, Any] | None:
        matches = [
            rec for rec in self._records.values()
            if rec.get("requirements", {}).get("language") == language
            and rec.get("requirements", {}).get("domain") == domain
        ]
        return dict(matches[-1]) if matches else None

    async def store(self, record_dict: dict[str, Any], scs_text: str) -> str:
        project_id = str(record_dict["project_id"])
        self._records[project_id] = dict(record_dict)
        self._scs_map[project_id] = scs_text

        scs_path = self.base_dir / f"{project_id}.scs"
        scs_path.write_text(scs_text, encoding="utf-8")
        return f"sc_addr_{to_sc_idtf(project_id)}"

    async def list_all(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for pid, rec in self._records.items():
            entry = dict(rec)
            entry["scs_code"] = self._scs_map.get(pid, "")
            result.append(entry)
        return result


class HttpOstisGateway(AbstractOstisGateway):
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_by_hash(self, deduplication_hash: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/ostis/hash/{deduplication_hash}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return dict(resp.json())

    async def find_similar(self, language: str, domain: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/api/ostis/similar",
                params={"language": language, "domain": domain},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return dict(resp.json())

    async def store(self, record_dict: dict[str, Any], scs_text: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/ostis/cases",
                json={"record": record_dict, "scs": scs_text},
            )
            resp.raise_for_status()
            return str(resp.json()["sc_addr"])

    async def list_all(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/ostis/cases")
            resp.raise_for_status()
            return list(resp.json()["items"])