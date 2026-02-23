#!/usr/bin/env python3
"""
# Originally came from the following URL:
# https://matsuokenllmcommunity.slack.com/archives/C09837A235K/p1770475527794839?thread_ts=1770474745.272879&cid=C09837A235K

DPOデータセットをHuggingFaceからダウンロードするスクリプト

使い方:
    python scripts/download_dpo_dataset.py

出力先:
    inputs/dpo/train.parquet
    inputs/dpo/train.json
"""

import os
import json
import numpy as np
from datasets import load_dataset


# DPOデータセット
DPO_REPO_ID = "u-10bei/dpo-dataset-qwen-cot"
OUTPUT_DIR = "inputs/dpo"


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


def main():
    """DPOデータセットをダウンロード"""
    print("=" * 60)
    print("DPO Dataset Downloader")
    print("=" * 60)
    print(f"\nRepo: {DPO_REPO_ID}")
    print(f"Output: {OUTPUT_DIR}/")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parquet_path = os.path.join(OUTPUT_DIR, "train.parquet")
    json_path = os.path.join(OUTPUT_DIR, "train.json")

    try:
        # HuggingFaceからデータセットをロード
        print("\nDownloading from HuggingFace...")
        ds = load_dataset(DPO_REPO_ID, split="train")
        print(f"  ✅ Loaded: {len(ds)} rows")
        print(f"  Columns: {ds.column_names}")

        # Parquet形式で保存
        ds.to_parquet(parquet_path)
        parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
        print(f"  ✅ Saved parquet: {parquet_path} ({parquet_size:.1f} MB)")

        # JSON形式で保存
        records = ds.to_pandas().to_dict(orient="records")
        records = convert_to_serializable(records)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        json_size = os.path.getsize(json_path) / (1024 * 1024)
        print(f"  ✅ Saved json: {json_path} ({json_size:.1f} MB)")

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)

    except Exception as e:
        print(f"  ❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
