#!/usr/bin/env python3
"""
# Originally came from the following URL:
# https://matsuokenllmcommunity.slack.com/archives/C09837A235K/p1770475527794839?thread_ts=1770474745.272879&cid=C09837A235K

SFTデータセットをHuggingFaceからダウンロードするスクリプト

使い方:
    python scripts/download_sft_datasets.py

出力先:
    inputs/{folder_name}/train.parquet
    inputs/{folder_name}/train.json
"""

import os
import json
import numpy as np
from datasets import load_dataset


# ダウンロード対象のデータセット定義
# (HuggingFace repo ID, ローカルフォルダ名, 備考)
DATASETS = [
    ("u-10bei/structured_data_with_cot_dataset_512_v2", "1-1_512_v2", "現在使用中"),
    ("u-10bei/structured_data_with_cot_dataset_512_v4", "1-2_512_v4", ""),
    ("u-10bei/structured_data_with_cot_dataset_512_v5", "1-3_512_v5", ""),
    ("u-10bei/structured_data_with_cot_dataset_512", "1-4_512", ""),
    ("u-10bei/structured_data_with_cot_dataset_v2", "1-5_v2", ""),
    ("u-10bei/structured_data_with_cot_dataset", "1-6_base", ""),
    ("daichira/structured-3k-mix-sft", "2-1_3k_mix", "長文系"),
    ("daichira/structured-5k-mix-sft", "2-2_5k_mix", "長文系"),
    ("daichira/structured-hard-sft-4k", "2-3_hard_4k", "長文系・高難度"),
]

# 出力先ベースディレクトリ
BASE_DIR = "inputs/sft"


def convert_to_serializable(obj):
    """numpy配列やその他の型をJSON serializableな形式に変換"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def download_dataset(repo_id: str, folder_name: str, note: str = "") -> bool:
    """
    HuggingFaceからデータセットをダウンロードし、parquetとjsonで保存する

    Args:
        repo_id: HuggingFace リポジトリID (例: "u-10bei/structured_data_with_cot_dataset_512_v2")
        folder_name: 保存先フォルダ名 (例: "1-1_512_v2")
        note: 備考（ログ表示用）

    Returns:
        成功したらTrue
    """
    out_dir = os.path.join(BASE_DIR, folder_name)
    parquet_path = os.path.join(out_dir, "train.parquet")
    json_path = os.path.join(out_dir, "train.json")

    os.makedirs(out_dir, exist_ok=True)

    note_str = f" ({note})" if note else ""
    print(f"\n{'=' * 60}")
    print(f"Downloading: {repo_id}{note_str}")
    print(f"  → {out_dir}")
    print(f"{'=' * 60}")

    try:
        # HuggingFaceからデータセットをロード
        ds = load_dataset(repo_id, split="train")
        print(f"  ✅ Loaded: {len(ds)} rows")
        print(f"  Columns: {ds.column_names}")

        # Parquet形式で保存（効率的な読み込み用）
        ds.to_parquet(parquet_path)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  ✅ Saved parquet: {parquet_path} ({parquet_size:.1f} MB)")

        # JSON形式で保存（人間が読める形式）
        records = ds.to_pandas().to_dict(orient="records")
        records = convert_to_serializable(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        json_size = os.path.getsize(json_path) / (1024 * 1024)
        print(f"  ✅ Saved json: {json_path} ({json_size:.1f} MB)")

        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """メイン処理: 全データセットをダウンロード"""
    print("=" * 60)
    print("SFT Datasets Downloader")
    print("=" * 60)
    print(f"\nTarget datasets: {len(DATASETS)}")
    print(f"Output directory: {BASE_DIR}/")

    os.makedirs(BASE_DIR, exist_ok=True)

    success_count = 0
    failed = []

    for repo_id, folder_name, note in DATASETS:
        if download_dataset(repo_id, folder_name, note):
            success_count += 1
        else:
            failed.append(folder_name)

    # サマリー表示
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total: {len(DATASETS)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")

    print("\nDone!")


if __name__ == "__main__":
    main()
