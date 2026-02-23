"""Analyse training runs and visualise loss curves grouped by epoch count."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.infer.generate_json import infer_and_generate_json
from scripts.scan_outputs_dir.manipulate_outputs_dir import (
    build_adapter_id,
    check_hf_repo_exists,
    list_subdir_names,
)


def _process_subdir(
    subdir: str,
    base_model_id: str,
    hf_repo_id: str,
    output_dir_base: str,
    force: bool,
) -> None:
    """Run inference for a single subdirectory, skipping if the HF repo does not exist.

    Args:
        subdir: The subdirectory name to process.
        base_model_id: The base model identifier.
        hf_repo_id: The Hugging Face repository ID.
        output_dir_base: Base directory where inference outputs are stored.
        force: If True, run inference even when inference.json already exists.
    """
    adapter_id = build_adapter_id(base_model_id, subdir)

    if not check_hf_repo_exists(hf_repo_id, adapter_id):
        print(f"[INFO] Skipping {adapter_id}: not found on HF Hub")
        return

    if not force and Path(output_dir_base, subdir, "inference.json").exists():
        print(f"[INFO] Skipping {adapter_id}: inference.json already exists")
        return

    _t0 = time.monotonic()

    infer_and_generate_json(adapter_id=adapter_id, subdir=subdir)

    _elapsed = int(time.monotonic() - _t0)

    print(f"[INFO] {adapter_id} finished in {_elapsed // 3600:02d}:{_elapsed % 3600 // 60:02d}:{_elapsed % 60:02d}")


def main() -> None:
    """Run inference for all subdirectories in the output directory."""
    load_dotenv()

    base_model_id = os.getenv("SFT_BASE_MODEL")
    output_dir_base = os.getenv("OUTPUT_DIR_BASE")
    hf_repo_id = os.getenv("HF_REPO_ID")

    assert base_model_id, "SFT_BASE_MODEL env var is not set"
    assert output_dir_base, "OUTPUT_DIR_BASE env var is not set"
    assert hf_repo_id, "HF_REPO_ID env var is not set"

    force = "--all" in sys.argv

    subdir_list = list_subdir_names(output_dir_base)
    print(f"[INFO] Found {len(subdir_list)} candidate subdirectories")

    tuple(map(lambda s: _process_subdir(s, base_model_id, hf_repo_id, output_dir_base, force), subdir_list))


if __name__ == "__main__":
    main()
