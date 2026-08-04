"""Full pipeline entry point: generate data -> train -> evaluate.

Usage:
    python main.py [--config config/config.yaml] [--skip-generate]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(script: str, extra_args: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra_args]
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {script} (exit code {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full hand gesture pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    cfg_args = ["--config", args.config]

    if not args.skip_generate:
        run_step("generate_data.py", cfg_args)
    run_step("train.py", cfg_args)
    run_step("evaluate.py", cfg_args)


if __name__ == "__main__":
    main()
