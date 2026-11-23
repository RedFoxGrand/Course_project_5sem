"""CLI: python -m inference_agent examples/rtx5090_input.json --out plan"""
import argparse
import json
from pathlib import Path
from .agent import configure_inference
from .export import export_json, export_yaml


def main():
    parser = argparse.ArgumentParser(description="Agent 4: joint inference configurator")
    parser.add_argument("input", type=Path, help="JSON file with DesignState dictionary")
    parser.add_argument("--out", type=Path, default=Path("deployment_plan"), help="Output name (no suffix)")
    args = parser.parse_args()
    state = json.loads(args.input.read_text(encoding="utf-8"))
    result = configure_inference(state)
    if result["deployment_plan"] is None:
        parser.exit(2, "Validation failed: " + "; ".join(result["errors"]) + "\n")
    json_file = args.out.with_suffix(".json")
    yaml_file = args.out.with_suffix(".yaml")
    export_json(result, json_file)
    export_yaml(result, yaml_file)
    plan = result["deployment_plan"]
    print("Status:", plan["status"])
    if plan["primary"]:
        p = plan["primary"]
        print("Primary:", p["id"])
        print("Required VRAM/RAM (GiB):", p["memory"]["required_vram_gb"],
              "/", p["memory"]["required_ram_gb"])
    print("Saved:", json_file, yaml_file)


if __name__ == "__main__":
    main()
