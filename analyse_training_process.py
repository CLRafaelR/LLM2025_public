"""Analyse training runs and visualise loss curves grouped by epoch count."""

from __future__ import annotations

import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from scripts.scan_outputs_dir.manipulate_outputs_dir import (
    list_subdir_names,
    build_adapter_id,
    check_hf_repo_exists,
    parse_training_config,
)

_THINK_STYLE_ORDER: tuple[str, ...] = ("full", "tag-only", "remove")


def find_max_checkpoint(current_dir: str) -> int | None:
    """Find the highest checkpoint number in a training output directory.

    Args:
        current_dir: Path to the training run directory.

    Returns:
        The maximum checkpoint step number, or None if no checkpoints exist.
    """
    _ckpt_re = re.compile(r"checkpoint-(\d+)$")
    numbers = [
        int(m.group(1))  #
        for entry in os.listdir(current_dir)
        if (m := _ckpt_re.match(entry))
    ]
    return max(numbers) if numbers else None


def extract_loss_trace(current_dir: str, max_ckpt: int) -> pd.DataFrame | None:
    """Extract per-step training loss from trainer_state.json.

    Only entries that contain both ``"epoch"`` and ``"loss"`` keys are
    included; eval-only entries (which contain ``"eval_loss"`` but not
    ``"loss"``) are excluded.

    Args:
        current_dir: Path to the training run directory.
        max_ckpt: Checkpoint step number to read trainer_state.json from.

    Returns:
        DataFrame with columns ``epoch`` and ``loss``, or None on error.
    """
    state_path = os.path.join(current_dir, f"checkpoint-{max_ckpt}", "trainer_state.json")
    try:
        with open(state_path) as f:
            log_history = json.load(f)["log_history"]
    except Exception:
        return None

    rows = [
        {"epoch": entry["epoch"], "loss": entry["loss"]}
        for entry in log_history
        if "epoch" in entry and "loss" in entry
    ]
    return pd.DataFrame(rows) if rows else None


def build_training_spec(adapter_id: str, config: dict[str, str]) -> pd.DataFrame:
    """Build a single-row DataFrame combining adapter ID and config values.

    Args:
        adapter_id: The adapter identifier string.
        config: Parsed training configuration dictionary.

    Returns:
        Single-row DataFrame with ``adapter_id`` as first column followed
        by all config keys as additional columns.
    """
    return pd.DataFrame([{"adapter_id": adapter_id, **config}])


def process_single_subdir(
    subdir: str,
    base_model_id: str,
    hf_repo_id: str,
    output_dir_base: str,
) -> pd.DataFrame | None:
    """Process one training run directory into a combined DataFrame.

    Chains together all data-loading helpers.  Returns None if any step
    fails (missing checkpoint, missing HF repo, unparseable README, etc.).

    Args:
        subdir: Timestamp subdirectory name.
        base_model_id: HuggingFace base model ID.
        hf_repo_id: HuggingFace repository namespace.
        output_dir_base: Path to the parent output directory.

    Returns:
        DataFrame with loss trace columns plus training config columns,
        or None if this run should be skipped.
    """
    adapter_id = build_adapter_id(base_model_id, subdir)

    if not check_hf_repo_exists(hf_repo_id, adapter_id):
        return None

    current_dir = os.path.join(output_dir_base, subdir)

    max_ckpt = find_max_checkpoint(current_dir)
    if max_ckpt is None:
        return None

    loss_trace = extract_loss_trace(current_dir, max_ckpt)
    if loss_trace is None:
        return None

    config = parse_training_config(current_dir)
    if config is None:
        return None

    training_spec = build_training_spec(adapter_id, config)

    return loss_trace.assign(
        **{col: training_spec[col].iloc[0] for col in training_spec.columns},
    )


def collect_all_training_data(
    subdir_list: list[str],
    base_model_id: str,
    hf_repo_id: str,
    output_dir_base: str,
) -> pd.DataFrame:
    """Collect and concatenate training data from all valid subdirectories.

    Args:
        subdir_list: List of timestamp subdirectory names to process.
        base_model_id: HuggingFace base model ID.
        hf_repo_id: HuggingFace repository namespace.
        output_dir_base: Path to the parent output directory.

    Returns:
        Concatenated DataFrame of all valid training runs.

    Raises:
        ValueError: If no valid training runs are found.
    """
    results = list(
        filter(
            lambda x: x is not None,
            map(
                lambda subdir: process_single_subdir(
                    subdir,
                    base_model_id,
                    hf_repo_id,
                    output_dir_base,
                ),
                subdir_list,
            ),
        )
    )
    if not results:
        raise ValueError("No valid training runs found.")
    return pd.concat(results, ignore_index=True)


def plot_loss_curves(df: pd.DataFrame, output_dir_base: str) -> None:
    """Plot per-adapter loss curves faceted by epoch count and save to PNG.

    Args:
        df: Combined DataFrame with columns ``epoch``, ``loss``,
            ``adapter_id``, and ``Epochs``.
        output_dir_base: Directory in which to save the output PNG.
    """
    col_order = sorted(df["Epochs"].unique(), key=int)
    row_order = [s for s in _THINK_STYLE_ORDER if s in df["think_style"].unique()]
    with sns.axes_style("whitegrid", {"grid.color": ".85", "grid.linewidth": 0.8}):
        g = sns.relplot(
            data=df,
            x="epoch",
            y="loss",
            col="Epochs",
            row="think_style",
            hue="adapter_id",
            style="Lora target modules",
            kind="line",
            col_order=col_order,
            row_order=row_order,
            markers=True,
            dashes=False,
        )
        leg = g.legend
        handles = leg.legend_handles
        labels = [t.get_text() for t in leg.texts]
        split = labels.index("Lora target modules")
        leg.remove()
        g.figure.legend(
            handles[1:split],
            labels[1:split],
            title="adapter_id",
            loc="center right",
            bbox_to_anchor=(1.0, 0.5),
        )
        g.figure.legend(
            handles[split + 1 :],
            labels[split + 1 :],
            title="Lora target modules",
            loc="lower center",
            bbox_to_anchor=(0.5, 0),
            ncol=len(handles[split + 1 :]),
        )
        g.figure.subplots_adjust(bottom=0.12)
        g.set_axis_labels("Epoch", "Loss")
        g.set_titles(col_template="Epochs: {col_name}")

    output_path = os.path.join(output_dir_base, "training_loss_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Saved: {output_path}")
    plt.show()


def main() -> None:
    """Load environment, collect training data, and render loss curve plots."""
    load_dotenv()

    base_model_id = os.getenv("SFT_BASE_MODEL")
    output_dir_base = os.getenv("OUTPUT_DIR_BASE")
    hf_repo_id = os.getenv("HF_REPO_ID")

    assert base_model_id, "SFT_BASE_MODEL env var is not set"
    assert output_dir_base, "OUTPUT_DIR_BASE env var is not set"
    assert hf_repo_id, "HF_REPO_ID env var is not set"

    subdir_list = list_subdir_names(output_dir_base)
    print(f"[INFO] Found {len(subdir_list)} candidate subdirectories")

    df = collect_all_training_data(subdir_list, base_model_id, hf_repo_id, output_dir_base)
    print(f"[INFO] Collected {len(df)} data points from {df['adapter_id'].nunique()} runs")

    plot_loss_curves(df, output_dir_base)


if __name__ == "__main__":
    main()
