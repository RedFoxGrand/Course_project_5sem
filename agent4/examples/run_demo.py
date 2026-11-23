"""Run: python examples/run_demo.py examples/rtx5090_input.json"""
import json
import sys
from pathlib import Path

from inference_agent import configure_inference, export_json, export_yaml

def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/rtx5090_input.json")
    state = json.loads(path.read_text(encoding="utf-8"))
    out = configure_inference(state)
    export_json(out, "deployment_plan.json")
    export_yaml(out, "deployment_plan.yaml")
    plan = out["deployment_plan"]
    print("STATUS:", plan["status"] if plan else "invalid input")
    if plan and plan["primary"]:
        p = plan["primary"]
        print("PRIMARY:", p["id"])
        print("VRAM (GiB):", p["memory"]["required_vram_gb"])
        print("RAM  (GiB):", p["memory"]["required_ram_gb"])
        print("FALLBACK:", plan["fallback"]["id"] if plan["fallback"] else "none")
    if out["errors"]:
        print("ERRORS:", out["errors"])
    print("Written: deployment_plan.json, deployment_plan.yaml")

if __name__ == "__main__":
    main()
