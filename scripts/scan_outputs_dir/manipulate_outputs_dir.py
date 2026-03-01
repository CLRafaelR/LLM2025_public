from __future__ import annotations

import os
import re
from huggingface_hub import repo_exists
from functools import reduce

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


_CONFIG_KEYS: tuple[tuple[str, str], ...] = (
    ("- Base model:", "Base model"),
    ("- Method:", "Method"),
    ("- Max sequence length:", "Max sequence length"),
    ("- Epochs:", "Epochs"),
    ("- Learning rate:", "Learning rate"),
    ("- LoRA:", "LoRA"),
    ("- Lora target modules:", "Lora target modules"),
    ("- Lora dropout:", "Lora dropout"),
    ("- Per device batch size:", "Per device batch size"),
    ("- Gradient accumulation step:", "Gradient accumulation step"),
)

_THINK_STYLE_MAP: dict[str, str] = {
    "Reasoning CoT text was stored inside `<think>...</think>` tags and included in the training data as-is.": "full",
    "Reasoning CoT text was removed, and only empty `<think>\\n</think>` tags were retained in the training data.": "tag-only",
    "Reasoning CoT text was removed entirely, and no `<think>` tags were added to the training data.": "remove",
}


def parse_training_config(current_dir: str) -> dict[str, str] | None:
    """Parse the Training Configuration section from README.md.

    Reads the README.md in ``current_dir`` and extracts key-value pairs
    from the ``## Training Configuration`` section.  Returns None for
    Unsloth auto-generated READMEs that lack this section.

    Args:
        current_dir: Path to the training run directory.

    Returns:
        Dict mapping config key names to their values, or None if the
        section is absent or the file cannot be read.
    """
    readme_path = os.path.join(current_dir, "README.md")
    try:
        with open(readme_path) as f:
            text = f.read()
    except Exception:
        return None

    section_match = re.search(
        r"## Training Configuration\n(.*?)(?:\n##|\Z)",
        text,
        re.DOTALL,
    )
    if not section_match:
        return None

    section_text = section_match.group(1)

    def _extract_key(acc: dict[str, str], kv: tuple[str, str]) -> dict[str, str]:
        prefix, key = kv
        line_match = re.search(
            rf"^{re.escape(prefix)}\s*(.+)$",
            section_text,
            re.MULTILINE,
        )
        if line_match:
            return {**acc, key: line_match.group(1).strip()}
        return acc

    raw = reduce(_extract_key, _CONFIG_KEYS, {})
    if not raw:
        return None

    think_match = re.search(
        r"^- Treatment of reasoning CoT:\s*(.+)$",
        text,
        re.MULTILINE,
    )
    think_raw = think_match.group(1).strip() if think_match else ""
    return {**raw, "think_style": _THINK_STYLE_MAP.get(think_raw, think_raw)}
