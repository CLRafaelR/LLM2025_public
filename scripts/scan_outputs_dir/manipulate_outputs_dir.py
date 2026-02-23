from __future__ import annotations

import os
import re
from huggingface_hub import repo_exists

_SUBDIR_PATTERN = re.compile(r"^\d{8}_T\d{6}$")


def list_subdir_names(output_dir_base: str) -> list[str]:
    """List timestamp-formatted subdirectory names under output_dir_base.

    Args:
        output_dir_base: Path to the base output directory.

    Returns:
        Sorted list of subdirectory names matching YYYYMMDD_THHMMSS pattern.
    """
    return sorted(filter(_SUBDIR_PATTERN.match, os.listdir(output_dir_base)))


def build_adapter_id(base_model_id: str, subdir: str) -> str:
    """Build the adapter ID from base model ID and subdir name.

    Args:
        base_model_id: HuggingFace model ID in ``owner/name`` format.
        subdir: Timestamp subdirectory name (YYYYMMDD_THHMMSS).

    Returns:
        Adaptor ID string in ``modelname-subdir`` format.
    """
    return f"{base_model_id.split('/')[1]}-{subdir}"


def check_hf_repo_exists(hf_repo_id: str, adapter_id: str) -> bool:
    """Check whether the HuggingFace model repository exists.

    Args:
        hf_repo_id: HuggingFace repository namespace (e.g. ``owner/repo``).
        adapter_id: Adaptor identifier to append to the repo namespace.

    Returns:
        True if the repository exists, False otherwise.
    """
    return repo_exists(f"{hf_repo_id}/{adapter_id}", repo_type="model")
