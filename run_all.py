"""Run all four lab steps sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    ROOT / "pseudocode" / "01_langsmith_rag_pipeline.py",
    ROOT / "pseudocode" / "02_prompt_hub_ab_routing.py",
    ROOT / "pseudocode" / "03_ragas_evaluation.py",
    ROOT / "pseudocode" / "04_guardrails_validator.py",
]


def main() -> int:
    for step in STEPS:
        print("\n" + "=" * 72)
        print(f"Running {step.name}")
        print("=" * 72)
        completed = subprocess.run([sys.executable, str(step)], cwd=ROOT)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())