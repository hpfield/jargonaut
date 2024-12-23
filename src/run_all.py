#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path
import re
import shutil
import time
import yaml

GENERATE_QA_SCRIPT = "generate_qa.py"
FINETUNE_SCRIPT = "finetune.py"
BUILD_EMBEDDINGS_SCRIPT = "build_embeddings.py"
DEMO_SERVER_SCRIPT = "demo_server.py"

STAGE_OUTPUT_DIRS = {
    "generate_qa": "outputs/generate_qa",
    "finetune": "outputs/finetune",
    "build_embeddings": "outputs/build_embeddings",
    "demo_server": "outputs/demo_server",
}

# Regex for Hydra-run folder naming convention: YY-MM-DD_HH-MM-SS
TIMESTAMP_REGEX = re.compile(r"^\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the pipeline scripts in sequence, updating config paths automatically."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "generate_qa", "finetune", "build_embeddings", "demo_server"],
        help="Which stage to run. 'all' runs everything in sequence."
    )
    return parser.parse_args()

def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config_path: Path, cfg: dict):
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

def get_latest_output_dir(stage_name: str) -> Path:
    """
    Finds the newest directory in outputs/<stage_name>/ that matches your timestamp format,
    and returns it. Raises an error if none found.
    """
    base_dir = Path(STAGE_OUTPUT_DIRS[stage_name])
    if not base_dir.exists():
        raise FileNotFoundError(f"Stage directory not found: {base_dir}")

    # Gather subdirs that match the YY-MM-DD_HH-MM-SS pattern
    candidates = []
    for item in base_dir.iterdir():
        if item.is_dir() and TIMESTAMP_REGEX.match(item.name):
            candidates.append(item)
    if not candidates:
        raise FileNotFoundError(f"No timestamped output directory found in {base_dir}")

    # Sort by modification time descending, pick the newest
    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]

def run_stage(script_name: str):
    """
    Runs a Python script (e.g. generate_qa.py) via subprocess, blocking until it finishes.
    Raises an exception on non-zero exit code.
    """
    print(f"=== Running {script_name} ...")
    result = subprocess.run(["python", script_name], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

def main():
    args = parse_args()

    config_path = Path("config") / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at {config_path}")

    cfg = load_config(config_path)

    pipeline_order = []
    if args.stage == "all":
        pipeline_order = ["generate_qa", "finetune", "build_embeddings", "demo_server"]
    else:
        pipeline_order = [args.stage]

    for stage_name in pipeline_order:
        if stage_name == "generate_qa":
            run_stage(GENERATE_QA_SCRIPT)
            # After finishing, find newest output dir for generate_qa
            latest_dir = get_latest_output_dir("generate_qa")
            # Update the config so the next stage can find train_dataset.json & val_dataset.json
            cfg["paths"]["train_dataset_path"] = str(latest_dir / "train_dataset.json")
            cfg["paths"]["val_dataset_path"] = str(latest_dir / "val_dataset.json")

        elif stage_name == "finetune":
            run_stage(FINETUNE_SCRIPT)
            # Get newest finetune output
            latest_dir = get_latest_output_dir("finetune")
            # Update config with the finetuned model path
            cfg["paths"]["finetuned_model_path"] = str(latest_dir / "finetuned_model")

        elif stage_name == "build_embeddings":
            run_stage(BUILD_EMBEDDINGS_SCRIPT)
            # Get newest build_embeddings output
            latest_dir = get_latest_output_dir("build_embeddings")
            # Update config with doc_embeddings.pkl
            cfg["paths"]["doc_embeddings_path"] = str(latest_dir / "doc_embeddings.pkl")

        elif stage_name == "demo_server":
            run_stage(DEMO_SERVER_SCRIPT)

        save_config(config_path, cfg)
        print(f"Updated {config_path} for stage '{stage_name}' with new paths.")

    print("=== Pipeline complete. ===")

if __name__ == "__main__":
    main()