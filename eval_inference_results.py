#!/usr/bin/env python3
"""Batch evaluation script for all inference results under OUTPUT_DIR_BASE.

Iterates all qualifying subdirectories (HF repo exists + inference.json exists),
runs evaluation for each, saves per-subdirectory validation_errors.json, and
produces a consolidated all_validation.md report with rankings.

Usage:
    uv run python eval_inference_results.py
"""

import contextlib
import io
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from scripts.analyse.local_eval import evaluate_with_task_id
from scripts.scan_outputs_dir.manipulate_outputs_dir import (
    build_adapter_id,
    check_hf_repo_exists,
    list_subdir_names,
    parse_training_config,
)
from scripts.analyse.local_eval import _get_rating_text


def _extract_report_section(captured: str) -> str:
    """Extract the evaluation report block from captured stdout.

    Args:
        captured: Full stdout text from evaluate_with_task_id.

    Returns:
        Portion of text starting at the first '====...' separator line,
        or the full captured text if no separator is found.
    """
    lines = captured.splitlines(keepends=True)
    separator_indices = [i for i, ln in enumerate(lines) if ln.startswith("=" * 10)]
    if not separator_indices:
        return captured
    return "".join(lines[separator_indices[0] :])


def _format_training_config(config: dict[str, str] | None) -> str:
    """Format training config dict as a markdown bullet list.

    Args:
        config: Dict returned by ``parse_training_config``, or ``None`` if
            the README was absent or lacked a Training Configuration section.

    Returns:
        Markdown bullet list string, or a fallback message if config is None.
    """
    if config is None:
        return "*(実験条件の情報が見つかりませんでした)*\n"
    return "".join(f"- **{key}**: {value}\n" for key, value in config.items())


def _evaluate_subdir(
    input_path: str,
    output_dir_base: str,
    base_model_id: str,
    hf_repo_id: str,
    subdir: str,
) -> tuple[str, str, dict, dict[str, str] | None] | None:
    """Evaluate one subdirectory and return results.

    Skips the subdirectory if the HF repo does not exist or inference.json
    is absent.

    Args:
        input_path: Path to public_150.json ground-truth file.
        output_dir_base: Base directory that contains timestamp subdirs.
        base_model_id: HuggingFace base model ID (``owner/name`` format).
        hf_repo_id: HuggingFace repo namespace.
        subdir: Timestamp subdirectory name.

    Returns:
        ``(adapter_id, report_text, result_dict, training_config)`` on success,
        ``None`` if the subdirectory is skipped.
    """
    adapter_id = build_adapter_id(base_model_id, subdir)

    if not check_hf_repo_exists(hf_repo_id, adapter_id):
        print(f"⏭  Skipping {adapter_id} (HF repo not found)", flush=True)
        return None

    subdir_path = Path(output_dir_base) / subdir
    inference_path = subdir_path / "inference.json"

    if not inference_path.exists():
        print(f"⏭  Skipping {adapter_id} (inference.json not found)", flush=True)
        return None

    print(f"\n{'=' * 60}", flush=True)
    print(f"🔍 Evaluating: {adapter_id}", flush=True)
    print(f"{'=' * 60}", flush=True)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = evaluate_with_task_id(
            str(input_path),
            str(inference_path),
            save_errors=True,
            error_output_dir=str(subdir_path),
        )

    captured = buffer.getvalue()
    sys.stdout.write(captured)
    sys.stdout.flush()

    report_text = _extract_report_section(captured)
    training_config = parse_training_config(str(subdir_path))
    return adapter_id, report_text, result, training_config


_RANKING_HEADER = (
    "| 順位 | Adapter_id | 総合得点 | 評定"
    " | ベースモデル | ファインチューニング手法 | 最大系列長 | エポック数 | 学習率"
    " | Loraパラメタ | チューニング対象 | ドロップアウト率"
    " | デバイス1個当たりのバッチサイズ | 累積勾配ステップ数"
    " | `<think>...</think>`取り扱い |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
)


def _ranking_row(rank_entry: tuple[int, tuple[str, str, dict, dict[str, str] | None]]) -> str:
    """Build one markdown table row for the ranking section.

    Args:
        rank_entry: ``(rank, (adapter_id, report_text, result_dict, training_config))``.

    Returns:
        Markdown table row string ending with ``\\n``.
    """
    rank, (adapter_id, _, result, config) = rank_entry
    cfg = config or {}
    return (
        f"| {rank} | {adapter_id} | {result['overall_rate']:.2f}%"
        f" | {_get_rating_text(result['overall_rate'])}"
        f" | {cfg.get('Base model', '-')}"
        f" | {cfg.get('Method', '-')}"
        f" | {cfg.get('Max sequence length', '-')}"
        f" | {cfg.get('Epochs', '-')}"
        f" | {cfg.get('Learning rate', '-')}"
        f" | {cfg.get('LoRA', '-')}"
        f" | {cfg.get('Lora target modules', '-')}"
        f" | {cfg.get('Lora dropout', '-')}"
        f" | {cfg.get('Per device batch size', '-')}"
        f" | {cfg.get('Gradient accumulation step', '-')}"
        f" | {cfg.get('think_style', '-')} |\n"
    )


def _build_ranking(sorted_results: list[tuple[str, str, dict, dict[str, str] | None]]) -> str:
    """Build the markdown ranking section (top 10) as a table.

    Args:
        sorted_results: List of ``(adapter_id, report_text, result_dict, training_config)``
            sorted by overall_rate descending.

    Returns:
        Markdown string for the ranking section.
    """
    rows = "".join(map(_ranking_row, enumerate(sorted_results[:10], start=1)))
    return f"# ランキング\n\n{_RANKING_HEADER}{rows}"


def _build_report_md(sorted_results: list[tuple[str, str, dict, dict[str, str] | None]]) -> str:
    """Build the full all_validation.md content.

    Args:
        sorted_results: List of ``(adapter_id, report_text, result_dict, training_config)``
            sorted by overall_rate descending.

    Returns:
        Complete markdown string for all_validation.md.
    """
    sections = [
        _build_ranking(sorted_results),
        "\n# 個別結果\n",
    ]
    sections += [
        f"\n## 【{rank}位】{adapter_id}\n\n### 結果\n\n{report_text}\n### 実験条件\n\n{_format_training_config(config)}"
        for rank, (adapter_id, report_text, _, config) in enumerate(sorted_results, start=1)
    ]
    return "".join(sections)


def main() -> None:
    """Entry point: load env, iterate subdirs, write report."""
    load_dotenv()

    input_path = os.getenv("INPUT_PATH")
    output_dir_base = os.getenv("OUTPUT_DIR_BASE")
    base_model_id = os.getenv("SFT_BASE_MODEL")
    hf_repo_id = os.getenv("HF_REPO_ID")

    missing = [
        name
        for name, val in [
            ("INPUT_PATH", input_path),
            ("OUTPUT_DIR_BASE", output_dir_base),
            ("SFT_BASE_MODEL", base_model_id),
            ("HF_REPO_ID", hf_repo_id),
        ]
        if not val
    ]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    subdirs = list_subdir_names(output_dir_base)
    print(f"Found {len(subdirs)} subdirectories under {output_dir_base}")

    raw_results = list(
        filter(
            None,
            map(
                lambda subdir: _evaluate_subdir(input_path, output_dir_base, base_model_id, hf_repo_id, subdir),
                subdirs,
            ),
        )
    )

    if not raw_results:
        print("No qualifying subdirectories found. Exiting.")
        return

    sorted_results = sorted(raw_results, key=lambda x: x[2]["overall_rate"], reverse=True)

    report_path = Path(output_dir_base) / "all_validation.md"
    report_md = _build_report_md(sorted_results)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n✅ Report written to: {report_path}")


if __name__ == "__main__":
    main()
