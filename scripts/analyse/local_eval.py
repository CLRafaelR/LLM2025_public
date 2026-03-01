#!/usr/bin/env python3
"""
ローカル簡易評価スクリプト（汎用版）

推論結果のフォーマット妥当性を検証し、簡易スコアを算出します。
詳細な評価基準が公表されていないため、フォーマットに沿っているかのみを判定しており、
実際のスコアとはずれる可能性があります。

ただし、簡易評価でスコアが低い場合は、そもそもフォーマットに沿った出力ができていない
ので、実際のスコアも低くなる可能性が高いです。
フォーマット別の成功率も確認できるので、弱点を強化する学習戦略を考えることができます。

■ 改善点（オリジナル版からの変更）
  - コードフェンス（```json等）を自動除去してから評価
    → LB評価との整合性が向上（相関係数: -0.05 → 0.48）
  - コマンドライン引数でファイル指定可能

■ 使い方
  python local_eval.py public_150.json inference.json

  または、Colabで:
  !python local_eval.py /content/public_150.json /content/inference.json

■ 必要なパッケージ
  pip install pyyaml toml pandas

■ 推奨される使い方
  - 簡易スコア 90%以上 → 提出候補として検討
  - 簡易スコア 80-90% → 要検討
  - 簡易スコア 80%未満 → 見直し推奨

Author: Person K (original), improved by community
"""

import json
import os
import re
import argparse
import pandas as pd
from collections import defaultdict
import yaml
import toml
import xml.etree.ElementTree as ET
import io


def strip_code_fence(text):
    """
    コードフェンスを除去する。

    LB評価ではコードフェンスが自動除去されている可能性があるため、
    ローカル評価でも除去してから評価することで整合性を向上させる。

    対応パターン:
      - ```json ... ```
      - ```yaml ... ```
      - ```toml ... ```
      - ```xml ... ```
      - ```csv ... ```
      - ``` ... ``` (言語指定なし)
    """
    text = text.strip()

    # パターン1: 完全なコードフェンス（開始と終了あり）
    pattern = r"^```\w*\n?(.*?)```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # パターン2: 開始タグのみ（閉じタグなし）
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            return "\n".join(lines[1:]).strip()

    return text


def validate_format(text, fmt):
    """
    フォーマットの妥当性を検証する。

    Args:
        text: 検証するテキスト
        fmt: 期待するフォーマット（JSON, YAML, TOML, XML, CSV）

    Returns:
        (is_valid, error_message): 妥当性と、エラー時のメッセージ
    """
    # コードフェンスを除去
    text = strip_code_fence(text)

    try:
        if fmt == "JSON":
            json.loads(text)
        elif fmt == "YAML":
            yaml.safe_load(text)
        elif fmt == "TOML":
            toml.loads(text)
        elif fmt == "XML":
            ET.fromstring(text)
        elif fmt == "CSV":
            if not text.strip():
                raise ValueError("Empty CSV")
            pd.read_csv(io.StringIO(text))
        else:
            return False, f"Unknown format: {fmt}"
        return True, None
    except Exception as e:
        return False, str(e)


def _get_rating_text(overall_rate: float) -> str:
    """Return rating emoji and text for a given overall_rate.

    Args:
        overall_rate: Evaluation score as a percentage (0–100).

    Returns:
        Rating string with emoji.
    """
    if overall_rate >= 90:
        return "✅ 提出候補として検討可能"
    if overall_rate >= 80:
        return "△ 要検討。弱点フォーマットの改善を推奨"
    return "⚠️ 見直し推奨。フォーマットエラーが多い"


def evaluate_with_task_id(
    public_file,
    inference_file,
    save_errors=True,
    error_output_dir=".",
):
    """
    推論結果を評価する。

    Args:
        public_file: テストデータファイル（public_150.json）のパス
        inference_file: 推論結果ファイル（inference.json）のパス
        save_errors: エラー詳細をJSONファイルに保存するかどうか

    Returns:
        dict: 評価結果
    """
    print(f"📂 Loading ground truth: {public_file}")
    with open(public_file, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # task_id -> output_type の辞書を作成
    id_to_type = {}
    for entry in ground_truth:
        tid = str(entry.get("task_id"))
        otype = entry.get("output_type")
        if tid and otype:
            id_to_type[tid] = otype.upper()

    print(f"   Loaded {len(id_to_type)} task definitions.")

    print(f"📂 Loading inference: {inference_file}")
    with open(inference_file, "r", encoding="utf-8") as f:
        inferences = json.load(f)

    print(f"   Loaded {len(inferences)} predictions.")

    # 集計用
    stats = defaultdict(lambda: {"total": 0, "valid": 0})
    errors = []

    for item in inferences:
        tid = str(item.get("task_id"))
        generated_text = item.get("generation", "")

        target_fmt = id_to_type.get(tid)

        if not target_fmt:
            print(f"   ⚠️ Warning: Task ID {tid} not found in ground truth.")
            continue

        is_valid, error_msg = validate_format(generated_text, target_fmt)

        stats[target_fmt]["total"] += 1
        if is_valid:
            stats[target_fmt]["valid"] += 1
        else:
            errors.append(
                {"task_id": tid, "format": target_fmt, "error": error_msg, "generated_snippet": generated_text[:100]}
            )

    # レポート出力
    print("\n" + "=" * 60)
    print("🏆 EVALUATION REPORT (with code fence removal)")
    print("=" * 60)

    total_valid = 0
    total_count = 0

    for fmt, data in sorted(stats.items()):
        rate = (data["valid"] / data["total"]) * 100 if data["total"] > 0 else 0
        bar = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        status = "✓" if rate >= 80 else "△" if rate >= 60 else "✗"
        print(f"[{fmt:4}] {bar} {rate:5.1f}%  ({data['valid']:3}/{data['total']:3}) {status}")
        total_valid += data["valid"]
        total_count += data["total"]

    print("-" * 60)
    overall_rate = (total_valid / total_count) * 100 if total_count > 0 else 0
    print(f"OVERALL: {overall_rate:.2f}%  ({total_valid}/{total_count})")
    print("=" * 60)

    # 判定と推奨
    _get_rating_text(overall_rate=overall_rate)

    # エラー詳細の保存
    if errors and save_errors:
        error_file = os.path.join(error_output_dir, "validation_errors.json")
        print(f"\n📝 Found {len(errors)} errors. Saving to '{error_file}'...")
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)

        # エラーサンプルを表示
        print(f"\n⚠️ Error samples (first 5):")
        for err in errors[:5]:
            err_msg = err["error"][:40].replace("\n", " ")
            print(f"   [{err['format']}] {err['task_id']}: {err_msg}...")

    return {
        "overall_rate": overall_rate,
        "total_valid": total_valid,
        "total_count": total_count,
        "stats": dict(stats),
        "error_count": len(errors),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ローカル簡易評価スクリプト（コードフェンス除去版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python local_eval.py public_150.json inference.json
  python local_eval.py /content/public_150.json /content/inference.json

推奨される使い方:
  - 簡易スコア 90%以上 → 提出候補として検討
  - 簡易スコア 80-90% → 要検討
  - 簡易スコア 80%未満 → 見直し推奨
        """,
    )
    parser.add_argument("public_file", help="テストデータファイル（public_150.json）")
    parser.add_argument("inference_file", help="推論結果ファイル（inference.json）")
    parser.add_argument("--no-save-errors", action="store_true", help="エラー詳細をファイルに保存しない")

    args = parser.parse_args()

    print("=" * 60)
    print("🔍 ローカル簡易評価スクリプト")
    print("   (コードフェンス自動除去版)")
    print("=" * 60)

    evaluate_with_task_id(args.public_file, args.inference_file, save_errors=not args.no_save_errors)


if __name__ == "__main__":
    main()
